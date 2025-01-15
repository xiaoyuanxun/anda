use anda_core::BoxError;
use futures::TryStreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{sync::Arc, vec};

pub mod lancedb;
pub use lancedb::*;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct Knowledge {
    pub id: String,
    pub text: String,
    pub meta: Value,
}

#[derive(Debug, Clone)]
pub struct KnowledgeInput<const DIM: usize> {
    pub text: String,
    pub meta: Value,
    pub vec: [f32; DIM],
}

#[derive(Clone)]
pub struct KnowledgeStore<const DIM: usize> {
    pub name: Path,
    table: Arc<Table>,
    embedder: Option<Arc<dyn EmbeddingFeaturesDyn>>,
    columns: Vec<String>,
}

impl<const DIM: usize> KnowledgeStore<DIM> {
    pub async fn init(
        db: &mut LanceVectorStore,
        name: Path,
        index_cache_size: Option<u32>,
    ) -> Result<Self, BoxError> {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("meta", DataType::Utf8, false),
            Field::new(
                "vec",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    DIM as i32,
                ),
                false,
            ),
        ]);

        let columns = vec!["id".to_string(), "text".to_string(), "meta".to_string()];
        let table = db
            .init_table(
                name.clone(),
                Arc::new(schema),
                Some("id".to_string()),
                Some("text".to_string()),
                index_cache_size,
            )
            .await?;

        Ok(Self {
            name,
            table: Arc::new(table),
            embedder: db.embedder.clone(),
            columns,
        })
    }

    pub async fn create_index(&self) -> Result<(), BoxError> {
        self.table
            .create_index(&["text"], Index::FTS(FtsIndexBuilder::default()))
            .execute()
            .await?;
        // cannot create vector index if no data (requires 256 rows), ignore error
        let _ = self
            .table
            .create_index(&["vec"], Index::Auto)
            .execute()
            .await;
        // println!("{:?}", res);
        Ok(())
    }

    pub async fn optimize(&self) -> Result<(), BoxError> {
        let _ = self.table.optimize(OptimizeAction::All).await?;
        Ok(())
    }

    pub async fn add(&self, docs: Vec<KnowledgeInput<DIM>>) -> Result<(), BoxError> {
        if docs.is_empty() {
            return Ok(());
        }

        let schema = self.table.schema().await?;
        let mut ids: Vec<String> = Vec::with_capacity(docs.len());
        let mut texts: Vec<String> = Vec::with_capacity(docs.len());
        let mut metas: Vec<String> = Vec::with_capacity(docs.len());
        let mut vecs: Vec<Option<Vec<Option<f32>>>> = Vec::with_capacity(docs.len());
        for doc in docs {
            ids.push(xid::new().to_string());
            texts.push(doc.text);
            metas.push(serde_json::to_string(&doc.meta)?);
            vecs.push(Some(doc.vec.into_iter().map(Some).collect()));
        }
        // Create a RecordBatch stream.
        let batches = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(texts)),
                Arc::new(StringArray::from(metas)),
                Arc::new(
                    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(vecs, DIM as i32),
                ),
            ],
        )?;
        let batches = RecordBatchIterator::new(vec![batches].into_iter().map(Ok), schema);
        self.table.add(batches).execute().await?;
        Ok(())
    }

    pub async fn top_n(&self, query: String, n: usize) -> Result<Vec<Knowledge>, BoxError> {
        let mut res = if let Some(embedder) = &self.embedder {
            let prompt_embedding = embedder.embed_query(query.clone()).await?;
            self.table
                .vector_search(prompt_embedding.vec.clone())?
                .full_text_search(FullTextSearchQuery::new(query))
                .select(Select::Columns(self.columns.clone()))
                .limit(n)
                .execute()
                .await?
        } else {
            self.table
                .query()
                .full_text_search(FullTextSearchQuery::new(query))
                .select(Select::Columns(self.columns.clone()))
                .limit(n)
                .execute()
                .await?
        };

        let mut writer = arrow_json::ArrayWriter::new(Vec::new());
        while let Some(batch) = res.try_next().await? {
            writer.write(&batch)?;
        }
        let mut data = writer.into_inner();
        if data.is_empty() {
            data.extend_from_slice(b"[]");
        }
        if data.last() != Some(&b']') {
            data.push(b']');
        }
        let docs = serde_json::from_slice(&data)?;
        Ok(docs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn test_knowledge_store() {
        let os = InMemory::new();
        let mut store = LanceVectorStore::new_with_object_store(
            "ic://object_store".to_string(),
            Arc::new(os),
            Some(1024 * 64),
            None,
        )
        .await
        .unwrap();

        let namespace: Path = "anda".into();
        let ks = KnowledgeStore::<384>::init(&mut store, namespace.clone(), Some(1024))
            .await
            .unwrap();

        ks.create_index().await.unwrap();

        let lt = store.table(&namespace).unwrap();
        assert_eq!(ks.name.as_ref(), lt.table.name());
        assert_eq!(&lt.id_field, "id");

        ks.add(vec![
            KnowledgeInput {
                text: "Hello".to_string(),
                meta: serde_json::json!({ "author": "a" }),
                vec: [0.1; 384],
            },
            KnowledgeInput {
                text: "Anda".to_string(),
                meta: serde_json::json!({ "author": "b" }),
                vec: [0.1; 384],
            },
        ])
        .await
        .unwrap();

        ks.create_index().await.unwrap();
        ks.optimize().await.unwrap();

        let res1 = store
            .top_n(namespace.clone(), "hello".to_string(), 10)
            .await
            .unwrap();
        assert_eq!(res1, vec!["Hello".to_string()]);

        let res2 = ks.top_n("hello".to_string(), 10).await.unwrap();
        println!("{:?}", res2);
        assert_eq!(res2.len(), 1);
        assert_eq!(res2[0].text, "Hello");

        let res3 = ks.top_n("anda".to_string(), 10).await.unwrap();
        println!("{:?}", res3);
        assert_eq!(res3.len(), 1);
        assert_eq!(res3[0].text, "Anda");

        let res = store
            .top_n_ids(namespace.clone(), "hello".to_string(), 10)
            .await
            .unwrap();
        println!("{:?}", res);
        assert_eq!(res[0], res2[0].id);
    }
}
