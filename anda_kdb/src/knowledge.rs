use anda_core::{BoxError, Knowledge, KnowledgeFeatures, KnowledgeInput, VectorSearchFeatures};
use anda_db::{
    collection::{Collection, CollectionConfig},
    error::DBError,
    index::HnswConfig,
    query::{Filter, Query, RangeQuery, Search},
    schema::{
        AndaDBSchema, FieldEntry, FieldType, Fv, Json, Schema, SchemaError, Vector, vector_from_f32,
    },
};
use anda_db_tfs::jieba_tokenizer;
use anda_engine::{model::EmbeddingFeaturesDyn, unix_ms};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{sync::Arc, vec};

use crate::db::*;

#[derive(Debug, Clone, Deserialize, Serialize, AndaDBSchema)]
pub struct LocalKnowledge {
    pub _id: u64,
    pub user: String,
    pub meta: Json,
    // knowledge content
    pub text: String,
    // knowledge embedding for vector search
    pub embedding: Vector,
    // timestamp in milliseconds
    pub created_at: u64,
}

#[derive(Clone)]
pub struct KnowledgeStore {
    name: String,
    collection: Arc<Collection>,
    embedder: Option<Arc<dyn EmbeddingFeaturesDyn>>,
}

impl KnowledgeStore {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub async fn init(db: &mut AndaKDB, name: String, dimension: usize) -> Result<Self, DBError> {
        let schema = LocalKnowledge::schema()?;

        let config = CollectionConfig {
            name: name.to_string(),
            description: "AI Agent knowledges".to_string(),
        };

        let collection = db
            .open_or_create_collection(schema, config, async |collection| {
                collection.set_tokenizer(jieba_tokenizer());
                collection.create_btree_index_nx(&["user"]).await?;
                collection.create_btree_index_nx(&["created_at"]).await?;
                // create BM25 & HNSW indexes if not exists
                collection
                    .create_bm25_index_nx(&["user", "meta", "text"])
                    .await?;
                collection
                    .create_hnsw_index_nx(
                        "embedding",
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
            collection,
            embedder: db.embedder(),
        })
    }

    async fn try_embed_query(&self, query: String) -> Option<Vec<f32>> {
        if let Some(embedder) = self.embedder.as_ref() {
            match embedder.embed_query(query).await {
                Ok((embedding, _usage)) => Some(embedding.vec),
                Err(_) => None,
            }
        } else {
            None
        }
    }
}

impl VectorSearchFeatures for KnowledgeStore {
    async fn top_n(&self, query: &str, n: usize) -> Result<Vec<String>, BoxError> {
        if n == 0 {
            return Ok(vec![]);
        }

        let vector = self.try_embed_query(query.to_string()).await;
        let result: Vec<LocalKnowledge> = self
            .collection
            .search_as(Query {
                limit: Some(n),
                search: Some(Search {
                    text: Some(query.to_string()),
                    vector,
                    ..Default::default()
                }),
                ..Default::default()
            })
            .await?;

        Ok(result.into_iter().map(|doc| doc.text).collect())
    }

    async fn top_n_ids(&self, query: &str, n: usize) -> Result<Vec<String>, BoxError> {
        if n == 0 {
            return Ok(vec![]);
        }

        let vector = self.try_embed_query(query.to_string()).await;
        let result: Vec<u64> = self
            .collection
            .search_ids(Query {
                limit: Some(n),
                search: Some(Search {
                    text: Some(query.to_string()),
                    vector,
                    ..Default::default()
                }),
                ..Default::default()
            })
            .await?;

        Ok(result.into_iter().map(|id| id.to_string()).collect())
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

        let vector = self.try_embed_query(query.to_string()).await;
        let result: Vec<LocalKnowledge> = self
            .collection
            .search_as(Query {
                limit: Some(n),
                filter: user.map(|u| {
                    Filter::Field(("user".to_string(), RangeQuery::Eq(Fv::Text(u.to_string()))))
                }),
                search: Some(Search {
                    text: Some(query.to_string()),
                    vector,
                    ..Default::default()
                }),
            })
            .await?;

        Ok(result
            .into_iter()
            .map(|doc| Knowledge {
                id: doc._id.to_string(),
                user: doc.user,
                text: doc.text,
                meta: serde_json::from_value(doc.meta).unwrap_or_default(),
            })
            .collect())
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

        let mut filter = Filter::Field((
            "created_at".to_string(),
            RangeQuery::Ge(Fv::U64(unix_ms() - last_seconds as u64 * 1000)),
        ));

        if let Some(u) = user {
            filter = Filter::And(vec![
                Box::new(Filter::Field((
                    "user".to_string(),
                    RangeQuery::Eq(Fv::Text(u.to_string())),
                ))),
                Box::new(filter),
            ]);
        }
        let result: Vec<LocalKnowledge> = self
            .collection
            .search_as(Query {
                limit: Some(n),
                filter: Some(filter),
                ..Default::default()
            })
            .await?;

        Ok(result
            .into_iter()
            .map(|doc| Knowledge {
                id: doc._id.to_string(),
                user: doc.user,
                text: doc.text,
                meta: serde_json::from_value(doc.meta).unwrap_or_default(),
            })
            .collect())
    }

    async fn knowledge_add(&self, docs: Vec<KnowledgeInput>) -> Result<(), BoxError> {
        if docs.is_empty() {
            return Ok(());
        }
        let now = unix_ms();
        let docs = docs
            .into_iter()
            .map(|doc| LocalKnowledge {
                _id: 0,
                user: doc.user,
                meta: json!(doc.meta),
                text: doc.text,
                embedding: vector_from_f32(doc.vec),
                created_at: now,
            })
            .collect::<Vec<_>>();

        for doc in docs {
            let _ = self.collection.add_from(&doc).await?;
        }

        Ok(())
    }
}
