use anda_core::{BoxError, Path};
use anda_db::{
    collection::{Collection, CollectionConfig},
    database::{AndaDB, DBConfig},
    error::DBError,
    schema::Schema,
};
use anda_engine::model::EmbeddingFeaturesDyn;
use object_store::DynObjectStore;
use std::{collections::BTreeMap, sync::Arc};

pub use anda_db::storage::StorageConfig;

pub struct AndaKDB {
    pub db: AndaDB,
    collections: BTreeMap<Path, Arc<Collection>>,
    embedder: Option<Arc<dyn EmbeddingFeaturesDyn>>,
}

impl AndaKDB {
    pub async fn new(
        name: String,
        object_store: Arc<DynObjectStore>,
        storage: StorageConfig,
        embedder: Option<Arc<dyn EmbeddingFeaturesDyn>>,
    ) -> Result<Self, DBError> {
        let config = DBConfig {
            name,
            description: "Anda AI Agent knowledge store".to_string(),
            storage,
            lock: None,
        };

        // connect to the database (create if it doesn't exist)
        let db = AndaDB::connect(object_store, config).await?;
        Ok(AndaKDB {
            db,
            collections: BTreeMap::new(),
            embedder,
        })
    }

    pub async fn open_or_create_collection<F>(
        &mut self,
        schema: Schema,
        config: CollectionConfig,
        init_fn: F,
    ) -> Result<Arc<Collection>, DBError>
    where
        F: AsyncFnOnce(&mut Collection) -> Result<(), DBError>,
    {
        let name = config.name.clone();
        let collection = self
            .db
            .open_or_create_collection(schema, config, init_fn)
            .await?;
        self.collections
            .insert(Path::from(name), collection.clone());
        Ok(collection)
    }

    pub fn collection(&self, name: &Path) -> Result<Arc<Collection>, BoxError> {
        let col = self
            .collections
            .get(name)
            .ok_or_else(|| format!("collection {} not found", name))?;
        Ok(col.clone())
    }

    pub fn embedder(&self) -> Option<Arc<dyn EmbeddingFeaturesDyn>> {
        self.embedder.clone()
    }
}
