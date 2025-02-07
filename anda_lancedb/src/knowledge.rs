use anda_core::{BoxError, Knowledge, KnowledgeFeatures, KnowledgeInput, VectorSearchFeatures};
use anda_engine::unix_ms;
use std::{sync::Arc, vec};

use crate::lancedb::*;

#[derive(Clone)]
pub struct KnowledgeStore {
    name: Path,
    dim: i32,
    table: Arc<Table>,
    embedder: Option<Arc<dyn EmbeddingFeaturesDyn>>,
}

pub fn xid_from_timestamp(unix_secs: u32) -> xid::Id {
    let mut id = [0u8; 12];
    id[0..4].copy_from_slice(&unix_secs.to_be_bytes());
    xid::Id(id)
}

impl KnowledgeStore {
    pub fn name(&self) -> &Path {
        &self.name
    }

    pub async fn init(
        db: &mut LanceVectorStore,
        name: Path,
        dim: u16,
        index_cache_size: Option<u32>,
    ) -> Result<Self, BoxError> {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("user", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("meta", DataType::Utf8, false),
            Field::new(
                "vec",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float16, false)),
                    dim as i32,
                ),
                false,
            ),
        ]);

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
            dim: dim as i32,
            table: Arc::new(table),
            embedder: db.embedder(),
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
        Ok(())
    }

    pub async fn optimize(&self) -> Result<(), BoxError> {
        let _ = self.table.optimize(OptimizeAction::All).await?;
        Ok(())
    }
}

impl VectorSearchFeatures for KnowledgeStore {
    async fn top_n(&self, query: &str, n: usize) -> Result<Vec<String>, BoxError> {
        if n == 0 {
            return Ok(vec![]);
        }
        let docs = hybrid_search(
            &self.table,
            self.embedder.clone(),
            ["text".to_string()],
            query.to_string(),
            n,
            None,
        )
        .await?;

        Ok(docs.into_iter().flatten().collect())
    }

    async fn top_n_ids(&self, query: &str, n: usize) -> Result<Vec<String>, BoxError> {
        if n == 0 {
            return Ok(vec![]);
        }

        let ids = hybrid_search(
            &self.table,
            self.embedder.clone(),
            ["id".to_string()],
            query.to_string(),
            n,
            None,
        )
        .await?;
        Ok(ids.into_iter().flatten().collect())
    }
}

impl KnowledgeFeatures for KnowledgeStore {
    async fn knowledge_top_n(
        &self,
        query: &str,
        n: usize,
        user: Option<String>,
    ) -> Result<Vec<Knowledge>, BoxError> {
        if n == 0 {
            return Ok(vec![]);
        }

        let filter = user.map(|user| format!("user = {:?}", user.to_ascii_lowercase()));
        let docs = hybrid_search(
            &self.table,
            self.embedder.clone(),
            [
                "id".to_string(),
                "user".to_string(),
                "text".to_string(),
                "meta".to_string(),
            ],
            query.to_string(),
            n,
            filter,
        )
        .await?;
        let docs: Vec<Knowledge> = docs
            .into_iter()
            .map(|doc| Knowledge {
                id: doc[0].to_owned(),
                user: doc[1].to_owned(),
                text: doc[2].to_owned(),
                meta: serde_json::from_str(&doc[3]).unwrap_or(serde_json::Value::Null),
            })
            .collect();

        Ok(docs)
    }

    async fn knowledge_latest_n(
        &self,
        last_seconds: u32,
        n: usize,
        user: Option<String>,
    ) -> Result<Vec<Knowledge>, BoxError> {
        if last_seconds == 0 || n == 0 {
            return Ok(vec![]);
        }

        let timestamp = (unix_ms() / 1000).saturating_sub(last_seconds as u64);
        let id = xid_from_timestamp(timestamp as u32).to_string();
        let filter = if let Some(user) = user {
            format!("(id > {id:?}) AND (user = {:?})", user.to_ascii_lowercase())
        } else {
            format!("id > {id:?}")
        };
        let docs = hybrid_search(
            &self.table,
            self.embedder.clone(),
            [
                "id".to_string(),
                "user".to_string(),
                "text".to_string(),
                "meta".to_string(),
            ],
            "".to_string(),
            n,
            Some(filter),
        )
        .await?;
        let docs: Vec<Knowledge> = docs
            .into_iter()
            .map(|doc| Knowledge {
                id: doc[0].to_owned(),
                user: doc[1].to_owned(),
                text: doc[2].to_owned(),
                meta: serde_json::from_str(&doc[3]).unwrap_or(serde_json::Value::Null),
            })
            .collect();

        Ok(docs)
    }

    async fn knowledge_add(&self, docs: Vec<KnowledgeInput>) -> Result<(), BoxError> {
        if docs.is_empty() {
            return Ok(());
        }

        let schema = self.table.schema().await?;
        let mut ids: Vec<String> = Vec::with_capacity(docs.len());
        let mut users: Vec<String> = Vec::with_capacity(docs.len());
        let mut texts: Vec<String> = Vec::with_capacity(docs.len());
        let mut metas: Vec<String> = Vec::with_capacity(docs.len());
        let mut vecs: Vec<Option<Vec<Option<half::f16>>>> = Vec::with_capacity(docs.len());
        for doc in docs {
            if doc.vec.len() != self.dim as usize {
                return Err(format!(
                    "invalid vector length, expected {}, got {}",
                    self.dim,
                    doc.vec.len()
                )
                .into());
            }

            ids.push(xid::new().to_string());
            users.push(doc.user.to_ascii_lowercase());
            texts.push(doc.text);
            metas.push(serde_json::to_string(&doc.meta)?);
            vecs.push(Some(
                doc.vec
                    .into_iter()
                    .map(|v| Some(half::f16::from_f32(v)))
                    .collect(),
            ));
        }
        // Create a RecordBatch stream.
        let batches = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(users)),
                Arc::new(StringArray::from(texts)),
                Arc::new(StringArray::from(metas)),
                Arc::new(
                    FixedSizeListArray::from_iter_primitive::<Float16Type, _, _>(vecs, self.dim),
                ),
            ],
        )?;
        let batches = RecordBatchIterator::new(vec![batches].into_iter().map(Ok), schema);
        self.table.add(batches).execute().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candid::Principal;
    use ed25519_consensus::SigningKey;
    use ic_agent::{identity::BasicIdentity, Identity};
    use ic_cose_types::types::object_store::CHUNK_SIZE;
    use ic_object_store::{
        agent::build_agent,
        client::{Client, ObjectStoreClient},
    };

    #[tokio::test(flavor = "current_thread")]
    async fn test_knowledge_store() {
        let os = InMemory::new();
        let mut store = LanceVectorStore::new_with_object_store(
            "test://object_store".to_string(),
            Arc::new(os),
            Some(CHUNK_SIZE),
            None,
        )
        .await
        .unwrap();

        const DIM: u16 = 384;
        let namespace: Path = "anda".into();
        let ks = KnowledgeStore::init(&mut store, namespace.clone(), DIM, Some(1024))
            .await
            .unwrap();

        ks.create_index().await.unwrap();

        let lt = store.table(&namespace).unwrap();
        assert_eq!(ks.name.as_ref(), lt.table.name());
        assert_eq!(&lt.id_field, "id");

        ks.knowledge_add(vec![
            KnowledgeInput {
                user: "Anda".to_string(),
                text: "Hello".to_string(),
                meta: serde_json::json!({}),
                vec: vec![0.1; DIM as usize],
            },
            KnowledgeInput {
                user: "Dom".to_string(),
                text: "Anda".to_string(),
                meta: serde_json::json!({}),
                vec: vec![0.1; DIM as usize],
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

        let res2 = ks.knowledge_top_n("hello", 10, None).await.unwrap();
        println!("{:?}", res2);
        assert_eq!(res2.len(), 1);
        assert_eq!(res2[0].text, "Hello");

        let res3 = ks.knowledge_top_n("anda", 10, None).await.unwrap();
        println!("{:?}", res3);
        assert_eq!(res3.len(), 1);
        assert_eq!(res3[0].text, "Anda");

        let res = store
            .top_n_ids(namespace.clone(), "hello".to_string(), 10)
            .await
            .unwrap();
        println!("{:?}", res);
        assert_eq!(res[0], res2[0].id);

        let res = ks.knowledge_latest_n(1, 10, None).await.unwrap();
        println!("latest_n\n{:?}", res);
        assert_eq!(res.len(), 2);

        let res = ks
            .knowledge_latest_n(1, 10, Some("Anda".to_string()))
            .await
            .unwrap();
        println!("latest_n Anda:\n{:?}", res);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].user, "anda");

        let res = ks
            .knowledge_latest_n(1, 10, Some("Dom".to_string()))
            .await
            .unwrap();
        println!("latest_n Dom:\n{:?}", res);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].user, "dom");
    }

    #[tokio::test(flavor = "current_thread")]
    #[ignore]
    async fn test_with_ic_object_store() {
        // create a object store client with encryption on ICP canister
        // more details: https://github.com/ldclabs/ic-cose/tree/main/src/ic_object_store
        let secret = [8u8; 32];
        let canister = Principal::from_text("6at64-oyaaa-aaaap-anvza-cai").unwrap();
        let sk = SigningKey::from(secret);
        let id = BasicIdentity::from_signing_key(sk);
        println!("id: {:?}", id.sender().unwrap().to_text());
        // jjn6g-sh75l-r3cxb-wxrkl-frqld-6p6qq-d4ato-wske5-op7s5-n566f-bqe

        let agent = build_agent("http://localhost:4943", Arc::new(id))
            .await
            .unwrap();
        let cli = Arc::new(Client::new(Arc::new(agent), canister, Some(secret)));
        let os = ObjectStoreClient::new(cli.clone());

        let mut store = LanceVectorStore::new_with_object_store(
            "test://object_store".to_string(),
            Arc::new(os),
            Some(CHUNK_SIZE),
            None,
        )
        .await
        .unwrap();

        const DIM: u16 = 1024;
        let namespace: Path = "anda".into();
        let ks = KnowledgeStore::init(&mut store, namespace.clone(), DIM, Some(1024))
            .await
            .unwrap();

        ks.create_index().await.unwrap();

        let lt = store.table(&namespace).unwrap();
        assert_eq!(ks.name.as_ref(), lt.table.name());
        assert_eq!(&lt.id_field, "id");

        let res = ks.top_n("great", 10).await.unwrap();
        println!("{:?}", res);

        if res.is_empty() {
            println!("add some data");
            ks.knowledge_add(vec![
                KnowledgeInput {
                    user: "Anda".to_string(),
                    text: "Albert Einstein was a great theoretical physicist.".to_string(),
                    meta: serde_json::json!({}),
                    vec: vec![0.1; DIM as usize],
                },
                KnowledgeInput {
                    user: "Anda".to_string(),
                    text: "The Great Wall of China is one of the Seven Wonders of the World."
                        .to_string(),
                    meta: serde_json::json!({}),
                    vec: vec![0.2; DIM as usize],
                },
            ])
            .await
            .unwrap();

            // create_index or optimize the table at some time
            ks.create_index().await.unwrap();

            let res = ks.top_n("great", 10).await.unwrap();
            println!("{:?}", res);
        }
    }
}
