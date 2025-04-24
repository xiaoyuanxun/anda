use anda_core::{BoxError, Knowledge, KnowledgeFeatures, KnowledgeInput, VectorSearchFeatures};
use anda_db::{
    collection::{Collection, CollectionConfig},
    error::DBError,
    index::HnswConfig,
    schema::{Fe, Ft, Schema},
};
use anda_db_tfs::jieba_tokenizer;
use anda_engine::model::EmbeddingFeaturesDyn;
use std::{collections::BTreeMap, sync::Arc};

use crate::db::*;

#[derive(Clone)]
pub struct KnowledgeStore {
    name: String,
    dimension: usize,
    collection: Arc<Collection>,
    embedder: Option<Arc<dyn EmbeddingFeaturesDyn>>,
}

pub fn xid_from_timestamp(unix_secs: u32) -> xid::Id {
    let mut id = [0u8; 12];
    id[0..4].copy_from_slice(&unix_secs.to_be_bytes());
    xid::Id(id)
}

impl KnowledgeStore {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub async fn init(db: &mut AndaKDB, name: String, dimension: usize) -> Result<Self, DBError> {
        let mut schema = Schema::builder();
        schema
            .with_xid("xid", false)?
            .add_field(
                Fe::new("user".to_string(), Ft::Text)?
                    .with_required()
                    .with_description("user name".to_string()),
            )?
            .add_field(
                Fe::new("meta".to_string(), Ft::Map(BTreeMap::new()))?
                    .with_description("knowledge metadata".to_string()),
            )?
            .with_segments("segments", true)?;
        let schema = schema.build()?;

        let config = CollectionConfig {
            name: name.to_string(),
            description: "AI Agent knowledges".to_string(),
        };

        let collection = db
            .open_or_create_collection(schema, config, async |collection| {
                collection.set_tokenizer(jieba_tokenizer());
                collection
                    .create_btree_index_nx("btree_user", "user")
                    .await?;
                collection
                    .create_search_index_nx(
                        "search_segments",
                        "segments",
                        HnswConfig {
                            dimension,
                            ..Default::default()
                        },
                    )
                    .await?;
                Ok::<(), DBError>(())
            })
            .await?;

        Ok(Self {
            name,
            dimension,
            collection,
            embedder: db.embedder(),
        })
    }
}

impl VectorSearchFeatures for KnowledgeStore {
    async fn top_n(&self, query: &str, n: usize) -> Result<Vec<String>, BoxError> {
        if n == 0 {
            return Ok(vec![]);
        }
        unimplemented!()
    }

    async fn top_n_ids(&self, query: &str, n: usize) -> Result<Vec<String>, BoxError> {
        unimplemented!()
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

        unimplemented!()
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

        unimplemented!()
    }

    async fn knowledge_add(&self, docs: Vec<KnowledgeInput>) -> Result<(), BoxError> {
        if docs.is_empty() {
            return Ok(());
        }

        unimplemented!()
    }
}
