use anda_core::{BoxError, BoxPinFut};
use futures::TryStreamExt;
use std::{collections::BTreeMap, sync::Arc};

pub use anda_engine::{model::EmbeddingFeaturesDyn, store::VectorSearchFeaturesDyn};
pub use arrow_array::{
    types::Float16Type, FixedSizeListArray, RecordBatch, RecordBatchIterator, StringArray,
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
pub use object_store::{
    local::LocalFileSystem, memory::InMemory, path::Path, DynObjectStore, ObjectStore,
};

#[derive(Clone)]
pub struct LanceVectorStore {
    db: Connection,
    tables: BTreeMap<Path, LanceVectorTable>,
    embedder: Option<Arc<dyn EmbeddingFeaturesDyn>>,
}

/// Type on which vector searches can be performed for a lanceDb table.
#[derive(Clone)]
pub struct LanceVectorTable {
    pub table: Table,
    pub id_field: String,
    pub text_field: String,
}

pub async fn hybrid_search<const N: usize>(
    table: &Table,
    embedder: Option<Arc<dyn EmbeddingFeaturesDyn>>,
    select_columns: [String; N],
    query: String,
    n: usize,
    filter: Option<String>,
) -> Result<Vec<[String; N]>, BoxError> {
    let mut res = if query.is_empty() {
        let mut q = table
            .query()
            .select(Select::Columns(select_columns.to_vec()))
            .limit(n);
        if let Some(filter) = filter {
            q = q.only_if(filter);
        }
        q.execute().await?
    } else if let Some(embedder) = embedder {
        let prompt_embedding = embedder.embed_query(query.clone()).await?;
        let mut q = table
            .vector_search(prompt_embedding.vec.clone())?
            .full_text_search(FullTextSearchQuery::new(query))
            .select(Select::Columns(select_columns.to_vec()))
            .limit(n);
        if let Some(filter) = filter {
            q = q.only_if(filter);
        }
        q.execute().await?
    } else {
        let mut q = table
            .query()
            .full_text_search(FullTextSearchQuery::new(query))
            .select(Select::Columns(select_columns.to_vec()))
            .limit(n);
        if let Some(filter) = filter {
            q = q.only_if(filter);
        }
        q.execute().await?
    };

    let mut docs: Vec<[String; N]> = Vec::new();
    while let Some(batch) = res.try_next().await? {
        let mut rows: Vec<[String; N]> = vec![[const { String::new() }; N]; batch.num_rows()];
        for (col_idx, col) in select_columns.iter().enumerate() {
            let col_array = batch
                .column_by_name(col)
                .ok_or_else(|| format!("field {col} not found"))?
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| format!("column {col} is not a string array"))?;
            for (row_idx, item) in rows.iter_mut().enumerate().take(batch.num_rows()) {
                item[col_idx] = col_array.value(row_idx).to_string();
            }
        }
        docs.append(&mut rows);
    }

    Ok(docs)
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
        id_field: Option<String>,   // default to "id"
        text_field: Option<String>, // default to "text"
        index_cache_size: Option<u32>,
    ) -> Result<Table, BoxError> {
        let id_field = id_field.unwrap_or_else(|| "id".to_string());
        let text_field = text_field.unwrap_or_else(|| "text".to_string());
        let index_cache_size = index_cache_size.unwrap_or(1024);
        let id = schema
            .field_with_name(&id_field)
            .map_err(|err| format!("id field {} not found: {}", id_field, err))?;
        if id.data_type() != &DataType::Utf8 {
            return Err(format!("id field {} must be of type Utf8", id_field).into());
        }
        let text = schema
            .field_with_name(&text_field)
            .map_err(|err| format!("text field {} not found: {}", text_field, err))?;
        if text.data_type() != &DataType::Utf8 {
            return Err(format!("text field {} must be of type Utf8", text_field).into());
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
                text_field,
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

    pub fn embedder(&self) -> Option<Arc<dyn EmbeddingFeaturesDyn>> {
        self.embedder.clone()
    }
}

impl VectorSearchFeaturesDyn for LanceVectorStore {
    fn top_n(
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
        let text_field = table.text_field.clone();
        let table = table.table.clone();
        Box::pin(async move {
            let docs = hybrid_search(&table, embedder, [text_field], query, n, None).await?;

            Ok(docs.into_iter().flatten().collect())
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
            let ids = hybrid_search(&table, embedder, [id_field], query, n, None).await?;

            Ok(ids.into_iter().flatten().collect())
        })
    }
}
