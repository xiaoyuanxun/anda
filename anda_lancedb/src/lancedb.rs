use anda_core::{BoxError, BoxPinFut};
use futures::TryStreamExt;
use std::{collections::BTreeMap, sync::Arc};

pub use anda_engine::{model::EmbeddingFeaturesDyn, store::VectorSearchFeaturesDyn};
pub use arrow_array::{
    types::Float32Type, FixedSizeListArray, RecordBatch, RecordBatchIterator, StringArray,
};
pub use arrow_schema::{DataType, Field, Schema, SchemaRef};
pub use lance_index::scalar::FullTextSearchQuery;
pub use lance_io::object_store::{
    ObjectStore as LanceObjectStore, ObjectStoreParams, ObjectStoreProvider,
};
pub use lancedb::{
    connection::{ConnectBuilder, Connection, CreateTableMode},
    index::{scalar::FtsIndexBuilder, Index},
    query::{ExecutableQuery, QueryBase, Select},
    table::OptimizeAction,
    Table,
};
pub use object_store::{memory::InMemory, path::Path, DynObjectStore, ObjectStore};

#[derive(Clone)]
pub struct LanceVectorStore {
    db: Connection,
    tables: BTreeMap<Path, LanceVectorTable>,
    pub embedder: Option<Arc<dyn EmbeddingFeaturesDyn>>,
}

/// Type on which vector searches can be performed for a lanceDb table.
#[derive(Clone)]
pub struct LanceVectorTable {
    pub table: Table,
    pub id_field: String,
    pub columns: Vec<String>,
}

impl LanceVectorStore {
    pub async fn new_with_object_store(
        uri: String,
        object_store: Arc<DynObjectStore>,
        object_block_size: Option<usize>,
        embedder: Option<Arc<dyn EmbeddingFeaturesDyn>>,
    ) -> Result<Self, BoxError> {
        let db = ConnectBuilder::new(&uri)
            .object_store((Arc::new(object_store), object_block_size))
            .execute()
            .await?;
        Ok(Self::new(db, embedder))
    }

    pub fn new(db: Connection, embedder: Option<Arc<dyn EmbeddingFeaturesDyn>>) -> Self {
        Self {
            db,
            tables: BTreeMap::new(),
            embedder,
        }
    }

    /// Create a new table in the database if not exists.
    /// Or open the existing table.
    pub async fn init_table(
        &mut self,
        name: Path,
        schema: SchemaRef,
        id_field: Option<String>, // default to "id"
        columns: Option<Vec<String>>,
        index_cache_size: Option<u32>,
    ) -> Result<Table, BoxError> {
        let id_field = id_field.unwrap_or_else(|| "id".to_string());
        let index_cache_size = index_cache_size.unwrap_or(1024);
        let mut columns = columns.unwrap_or_default();
        let id = schema
            .field_with_name(&id_field)
            .map_err(|err| format!("id field {} not found: {}", id_field, err))?;
        if id.data_type() != &DataType::Utf8 {
            return Err(format!("id field {} must be of type Utf8", id_field).into());
        }
        if columns.is_empty() {
            for f in schema.fields() {
                match f.data_type() {
                    DataType::FixedSizeList(_, _) => continue,
                    _ => {
                        columns.push(f.name().to_string());
                    }
                }
            }
        }

        if !columns.contains(&id_field) {
            columns.push(id_field.clone());
        }

        let table_name = name.as_ref().to_ascii_lowercase();
        let table = match self
            .db
            .create_empty_table(&table_name, schema)
            .mode(CreateTableMode::Create)
            .execute()
            .await
        {
            Ok(res) => res,
            Err(err) => {
                if err.to_string().contains("already exists") {
                    self.db
                        .open_table(&table_name)
                        .index_cache_size(index_cache_size)
                        .execute()
                        .await
                        .map_err(|err| format!("failed to open table {}: {}", name, err))?
                } else {
                    return Err(format!("failed to create table {}: {}", name, err).into());
                }
            }
        };

        self.tables.insert(
            name,
            LanceVectorTable {
                table: table.clone(),
                id_field,
                columns,
            },
        );

        Ok(table)
    }

    pub fn table(&self, name: &Path) -> Result<LanceVectorTable, BoxError> {
        let table = self
            .tables
            .get(name)
            .ok_or_else(|| format!("table {} not found", name))?;
        Ok(table.clone())
    }
}

impl VectorSearchFeaturesDyn for LanceVectorStore {
    fn top_n(
        &self,
        namespace: Path,
        query: String,
        n: usize,
    ) -> BoxPinFut<Result<Vec<u8>, BoxError>> {
        let table = match self.table(&namespace) {
            Ok(table) => table,
            Err(err) => return Box::pin(futures::future::ready(Err(err))),
        };
        let embedder = self.embedder.clone();
        let columns = table.columns.clone();
        let table = table.table.clone();

        Box::pin(async move {
            let mut res = if let Some(embedder) = embedder {
                let prompt_embedding = embedder.embed_query(query.clone()).await?;
                table
                    .vector_search(prompt_embedding.vec.clone())?
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
            let mut data = writer.into_inner();
            if data.is_empty() {
                data.extend_from_slice(b"[]");
            }
            if data.last() != Some(&b']') {
                data.push(b']');
            }

            Ok(data)
        })
    }

    fn top_n_ids(
        &self,
        namespace: Path,
        query: String,
        n: usize,
    ) -> BoxPinFut<Result<Vec<String>, BoxError>> {
        let table = match self.table(&namespace) {
            Ok(table) => table,
            Err(err) => return Box::pin(futures::future::ready(Err(err))),
        };

        let embedder = self.embedder.clone();
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
                    .ok_or("id field not found")?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or("id column is not a string array")?;
                for s in id_array.into_iter().flatten() {
                    ids.push(s.to_string());
                }
            }

            Ok(ids)
        })
    }
}
