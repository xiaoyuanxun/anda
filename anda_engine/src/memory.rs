use anda_cognitive_nexus::CognitiveNexus;
use anda_core::{
    BoxError, FunctionDefinition, Message, Resource, StateFeatures, Tool, ToolOutput, Xid,
};
use anda_db::{
    collection::{Collection, CollectionConfig},
    database::AndaDB,
    error::DBError,
    schema,
};
use anda_db_schema::{AndaDBSchema, FieldEntry, FieldType, Schema, SchemaError};
use anda_db_tfs::jieba_tokenizer;
use anda_kip::{KIP_FUNCTION_DEFINITION, Request, Response, execute_kip};
use candid::Principal;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, LazyLock};

use crate::{context::BaseCtx, unix_ms};

pub static FUNCTION_DEFINITION: LazyLock<FunctionDefinition> =
    LazyLock::new(|| serde_json::from_value(KIP_FUNCTION_DEFINITION.clone()).unwrap());

#[derive(Debug, Clone, Deserialize, Serialize, AndaDBSchema)]
pub struct Chat {
    /// The unique identifier for this resource in the Anda DB collection "chat".
    pub _id: u64,

    #[field_type = "Bytes"]
    pub user: Principal,

    #[field_type = "Option<Bytes>"]
    pub thread: Option<Xid>,

    pub messages: Vec<Message>,

    /// The period when the chat was created, in hours (timestamp / 3600 / 1000).
    /// It is used to index the chat for faster retrieval by time.
    pub period: u64,

    /// The timestamp when the chat was created, in milliseconds.
    pub timestamp: u64,
}

#[derive(Debug, Deserialize, Serialize, AndaDBSchema)]
pub struct KIPLogs {
    /// The unique identifier for this resource in the Anda DB collection "kip_logs".
    pub _id: u64,

    #[field_type = "Bytes"]
    pub user: Principal,

    #[field_type = "Map<String, Json>"]
    pub request: anda_kip::Request,

    #[field_type = "Map<String, Json>"]
    pub response: anda_kip::Response,

    pub period: u64,

    pub timestamp: u64,
}

pub struct MemoryManagement {
    nexus: Arc<CognitiveNexus>,
    chats: Arc<Collection>,
    resources: Arc<Collection>,
    logs: Arc<Collection>,
}

impl MemoryManagement {
    pub async fn connect<F>(nexus: Arc<CognitiveNexus>, db: Arc<AndaDB>) -> Result<Self, BoxError> {
        let schema = Chat::schema()?;
        let chats = db
            .open_or_create_collection(
                schema,
                CollectionConfig {
                    name: "chats".to_string(),
                    description: "chats collection".to_string(),
                },
                async |collection| {
                    // set tokenizer
                    collection.set_tokenizer(jieba_tokenizer());
                    // create BTree indexes if not exists
                    collection.create_btree_index_nx(&["user"]).await?;
                    collection.create_btree_index_nx(&["thread"]).await?;
                    collection.create_btree_index_nx(&["period"]).await?;
                    collection.create_bm25_index_nx(&["messages"]).await?;

                    Ok::<(), DBError>(())
                },
            )
            .await?;

        let schema = Resource::schema()?;
        let resources = db
            .open_or_create_collection(
                schema,
                CollectionConfig {
                    name: "resources".to_string(),
                    description: "Resources collection".to_string(),
                },
                async |collection| {
                    // set tokenizer
                    collection.set_tokenizer(jieba_tokenizer());
                    // create BTree indexes if not exists
                    collection.create_btree_index_nx(&["tag"]).await?;
                    collection.create_btree_index_nx(&["mime_type"]).await?;

                    collection
                        .create_bm25_index_nx(&[
                            "tag",
                            "name",
                            "description",
                            "mime_type",
                            "metadata",
                        ])
                        .await?;

                    Ok::<(), DBError>(())
                },
            )
            .await?;

        let schema = KIPLogs::schema()?;
        let logs = db
            .open_or_create_collection(
                schema,
                CollectionConfig {
                    name: "kip_logs".to_string(),
                    description: "KIP logs collection".to_string(),
                },
                async |collection| {
                    // set tokenizer
                    collection.set_tokenizer(jieba_tokenizer());
                    // create BTree indexes if not exists
                    collection.create_btree_index_nx(&["period"]).await?;

                    Ok::<(), DBError>(())
                },
            )
            .await?;

        Ok(Self {
            nexus,
            chats,
            resources,
            logs,
        })
    }
}

impl Tool<BaseCtx> for MemoryManagement {
    type Args = Request;
    type Output = Response;

    fn name(&self) -> String {
        FUNCTION_DEFINITION.name.clone()
    }

    fn description(&self) -> String {
        FUNCTION_DEFINITION.description.clone()
    }

    fn definition(&self) -> FunctionDefinition {
        FUNCTION_DEFINITION.clone()
    }

    async fn call(
        &self,
        ctx: BaseCtx,
        request: Self::Args,
        _resources: Option<Vec<Resource>>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        let timestamp = unix_ms();
        let res = execute_kip(self.nexus.as_ref(), &request.to_command(), request.dry_run).await;
        let log = KIPLogs {
            _id: 0, // This will be set by the database
            user: ctx.caller().clone(),
            request: request,
            response: res.clone(),
            period: timestamp / 3600 / 1000,
            timestamp,
        };

        // ignore errors from adding logs
        let _ = self.logs.add_from(&log).await;
        Ok(ToolOutput::new(res))
    }
}
