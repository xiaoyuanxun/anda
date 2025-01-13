use anda_core::{BoxError, BoxPinFut, Path};
use anda_engine::{model::EmbeddingFeaturesDyn, store::VectorSearchFeaturesDyn};
use arrow_array::StringArray;
use futures::TryStreamExt;
use lance_index::scalar::FullTextSearchQuery;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use std::{collections::BTreeMap, sync::Arc};

pub struct LanceVectorStore {
    db: lancedb::Connection,
    tables: BTreeMap<Path, LanceVectorTable>,
}

/// Type on which vector searches can be performed for a lanceDb table.
#[derive(Clone)]
pub struct LanceVectorTable {
    table: lancedb::Table,
    /// Column name in `table` that contains the id of a record.
    id_field: String,
    columns: Vec<String>,
    embedder: Option<Arc<dyn EmbeddingFeaturesDyn>>,
}

// TODO
impl LanceVectorStore {}

impl VectorSearchFeaturesDyn for LanceVectorStore {
    fn top_n(
        &self,
        namespace: Path,
        query: String,
        n: usize,
    ) -> BoxPinFut<Result<Vec<u8>, BoxError>> {
        let table = self
            .tables
            .get(&namespace)
            .ok_or_else(|| format!("table {} not found", namespace));
        let table = match table {
            Ok(table) => table,
            Err(err) => return Box::pin(futures::future::ready(Err(err.into()))),
        };

        let embedder = table.embedder.clone();
        let columns = table.columns.clone();
        let table = table.table.clone();

        Box::pin(async move {
            let mut res = if let Some(embedder) = embedder {
                let prompt_embedding = embedder.embed_query(query.clone()).await?;
                table
                    .vector_search(prompt_embedding.vec.clone())?
                    // .column("vec")
                    .full_text_search(FullTextSearchQuery::new(query))
                    .select(Select::Columns(columns))
                    .limit(n)
                    .execute()
                    .await?
            } else {
                table
                    .query()
                    .full_text_search(FullTextSearchQuery::new(query))
                    .select(Select::Columns(columns))
                    .limit(n)
                    .execute()
                    .await?
            };

            let mut writer = arrow_json::ArrayWriter::new(Vec::new());
            while let Some(batch) = res.try_next().await? {
                writer.write(&batch)?;
            }
            let data = writer.into_inner();

            Ok(data)
        })
    }

    fn top_n_ids(
        &self,
        namespace: Path,
        query: String,
        n: usize,
    ) -> BoxPinFut<Result<Vec<String>, BoxError>> {
        let table = self
            .tables
            .get(&namespace)
            .ok_or_else(|| format!("table {} not found", namespace));
        let table = match table {
            Ok(table) => table,
            Err(err) => return Box::pin(futures::future::ready(Err(err.into()))),
        };

        let embedder = table.embedder.clone();
        let id_field = table.id_field.clone();
        let table = table.table.clone();
        Box::pin(async move {
            let mut res = if let Some(embedder) = embedder {
                let prompt_embedding = embedder.embed_query(query.clone()).await?;
                table
                    .vector_search(prompt_embedding.vec.clone())?
                    // .column("vec")
                    .full_text_search(FullTextSearchQuery::new(query))
                    .select(Select::Columns(vec![id_field.clone()]))
                    .limit(n)
                    .execute()
                    .await?
            } else {
                table
                    .query()
                    .full_text_search(FullTextSearchQuery::new(query))
                    .select(Select::Columns(vec![id_field.clone()]))
                    .limit(n)
                    .execute()
                    .await?
            };

            let mut ids: Vec<String> = Vec::new();
            while let Some(batch) = res.try_next().await? {
                let id_array = batch
                    .column_by_name(&id_field)
                    .ok_or_else(|| "id field not found")?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| "id column is not a string array")?;
                for s in id_array.into_iter() {
                    if let Some(s) = s {
                        ids.push(s.to_string());
                    }
                }
            }

            Ok(ids)
        })
    }
}
