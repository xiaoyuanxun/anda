use anda_core::{
    BaseContext, BoxError, CacheStoreFeatures, FunctionDefinition, MyThreads, RequestMeta,
    Resource, StateFeatures, ThreadId, ThreadMeta, Tool, ToolInput, ToolOutput, Value,
    gen_schema_for,
};
use candid::Principal;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{collections::BTreeSet, str::FromStr, sync::Arc};
use structured_logger::unix_ms;

use crate::context::BaseCtx;

pub static SYSTEM_PATH: &str = "_";

/// Represents system management tools for the Anda engine.
pub struct Management {
    ctx: BaseCtx,
    controller: Principal,
    managers: BTreeSet<Principal>,
}

impl Management {
    pub(crate) fn new(ctx: &BaseCtx, controller: Principal) -> Self {
        Self {
            ctx: ctx
                .child_with(
                    ctx.id,
                    SYSTEM_PATH.to_string(),
                    RequestMeta {
                        engine: None,
                        thread: None,
                        user: Some(ctx.name.clone()),
                    },
                )
                .expect("failed to create system context"),
            controller,
            managers: BTreeSet::new(),
        }
    }

    fn thread_meta_path(thread_id: &ThreadId) -> String {
        format!("TH_{}.meta.cbor", thread_id.xid())
    }

    fn my_threads_path(id: &Principal) -> String {
        format!("MYTH_{}.cbor", id.to_text())
    }

    /// Returns true if the caller is the controller of the engine.
    pub fn is_controller(&self, caller: &Principal) -> bool {
        caller == &self.controller
    }

    /// Returns true if the caller is the controller or a manager of the engine.
    pub fn is_manager(&self, caller: &Principal) -> bool {
        caller == &self.controller || self.managers.contains(caller)
    }

    /// Retrieves the thread metadata from the cache store.
    /// It does not check the permission of the caller for the thread.
    pub async fn get_thread_meta(&self, thread_id: &ThreadId) -> Result<ThreadMeta, BoxError> {
        let thread_key = Self::thread_meta_path(thread_id);
        self.ctx.cache_store_get::<ThreadMeta>(&thread_key).await
    }

    /// Loads the thread metadata from the cache store or remote engine.
    /// If the thread does not exist, a new thread will be created.
    pub async fn load_thread_meta(
        &self,
        caller: &Principal,
        thread_id: &Option<ThreadId>,
    ) -> Result<ThreadMeta, BoxError> {
        match thread_id {
            // Create a new thread if the thread_id is not provided.
            None => Ok(ThreadMeta::new(self.ctx.id, *caller, unix_ms())),
            Some(id) => {
                let thread_key = Self::thread_meta_path(id);
                match self.ctx.cache_store_get::<ThreadMeta>(&thread_key).await {
                    Ok(thread) => {
                        // Check if the caller has permission to access the thread.
                        if thread.has_permission(caller) {
                            Ok(thread)
                        } else {
                            Err(format!(
                                "caller {} does not have permission to access the thread {}",
                                caller.to_text(),
                                id
                            )
                            .into())
                        }
                    }
                    Err(_) => {
                        let threads = self.load_my_threads().await?;
                        if let Some(agent) = threads.get_agent_by(id) {
                            let endpoint =
                                self.ctx.remote.get_endpoint_by_id(agent).ok_or_else(|| {
                                    format!(
                                        "failed to get the engine endpoint: {}",
                                        agent.to_text()
                                    )
                                })?;
                            // Call the remote agent engine to get the thread metadata.
                            let output = self
                                .ctx
                                .remote_tool_call(
                                    &endpoint,
                                    ToolInput::new(
                                        ThreadMetaTool::NAME.to_string(),
                                        json!(ThreadMetaToolArgs {
                                            method: ThreadMetaToolMethod::GetThreadMeta,
                                            thread_id: id.to_string(),
                                            user_id: None,
                                        }),
                                    ),
                                )
                                .await?;
                            let thread: ThreadMeta = serde_json::from_value(output.output)?;
                            return Ok(thread);
                        }

                        // Create a new thread with parent if the thread does not exist.
                        let mut thread = ThreadMeta::new(self.ctx.id, *caller, unix_ms());
                        thread.parent = Some(id.to_owned());
                        Ok(thread)
                    }
                }
            }
        }
    }

    /// Saves the thread metadata to the cache store.
    pub async fn save_thread_meta(&self, mut thread: ThreadMeta) -> Result<(), BoxError> {
        let thread_key = Self::thread_meta_path(&thread.id);
        thread.updated_at = unix_ms();
        self.ctx.cache_store_set_and_wait(&thread_key, thread).await
    }

    /// Deletes the thread metadata from the cache store.
    pub async fn delete_thread_meta(
        &self,
        caller: &Principal,
        thread_id: &ThreadId,
    ) -> Result<(), BoxError> {
        let thread_key = Self::thread_meta_path(thread_id);
        match self.ctx.cache_store_get::<ThreadMeta>(&thread_key).await {
            Ok(thread) => {
                if thread.has_permission(caller) {
                    self.ctx.cache_store_delete(&thread_key).await
                } else {
                    Err(format!(
                        "caller {} does not have permission to delete the thread {}",
                        caller.to_text(),
                        thread_id
                    )
                    .into())
                }
            }
            Err(_) => Ok(()),
        }
    }

    /// Loads my threads index that participating in.
    pub async fn load_my_threads(&self) -> Result<MyThreads, BoxError> {
        let my_threads_key = Self::my_threads_path(&self.ctx.id);
        match self.ctx.cache_store_get::<MyThreads>(&my_threads_key).await {
            Ok(threads) => Ok(threads),
            Err(_) => Ok(MyThreads::new(self.ctx.id)),
        }
    }

    /// Saves my threads index that participating in.
    pub async fn save_my_threads(&self, threads: MyThreads) -> Result<(), BoxError> {
        let my_threads_key = Self::my_threads_path(&self.ctx.id);
        self.ctx
            .cache_store_set_and_wait(&my_threads_key, threads)
            .await
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
pub struct ThreadMetaToolArgs {
    /// The method to call.
    pub method: ThreadMetaToolMethod,

    /// The thread ID, e.g. "9z4e2mr0ui3e8a215n4g".
    pub thread_id: String,

    /// The user ID, e.g. "77ibd-jp5kr-moeco-kgoar-rro5v-5tng4-krif5-5h2i6-osf2f-2sjtv-kqe".
    pub user_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ThreadMetaToolMethod {
    GetThreadMeta,
    DeleteThreadMeta,
    AddParticipant,
    RemoveParticipant,
}

/// Represents a tool to manage thread metadata.
/// Thread is a conversation session between Agents and user. Threads store Messages and automatically handle truncation to fit content into a model’s context.
pub struct ThreadMetaTool {
    management: Arc<Management>,
    schema: Value,
}

impl ThreadMetaTool {
    pub const NAME: &'static str = "sys_my_threads";

    pub fn new(management: Arc<Management>) -> Self {
        let schema = gen_schema_for::<ThreadMetaToolArgs>();
        Self { management, schema }
    }
}

impl Tool<BaseCtx> for ThreadMetaTool {
    type Args = ThreadMetaToolArgs;
    type Output = Option<ThreadMeta>;

    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        "Manages thread metadata for user. Thread is a conversation session between AI agents and user.".to_string()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: self.schema.clone(),
            strict: Some(true),
        }
    }

    async fn call(
        &self,
        ctx: BaseCtx,
        args: Self::Args,
        resources: Option<Vec<Resource>>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        if resources.is_some() {
            return Err("resources are not supported".into());
        }
        let caller = ctx.caller();
        match args.method {
            ThreadMetaToolMethod::GetThreadMeta => {
                let thread_id = ThreadId::from_str(&args.thread_id)?;
                let thread = self.management.get_thread_meta(&thread_id).await?;
                if thread.has_permission(&caller) {
                    Ok(ToolOutput::new(Some(thread)))
                } else {
                    Err(format!(
                        "caller {} does not have permission to access the thread {}",
                        caller.to_text(),
                        thread_id
                    )
                    .into())
                }
            }
            ThreadMetaToolMethod::DeleteThreadMeta => {
                let thread_id = ThreadId::from_str(&args.thread_id)?;
                self.management
                    .delete_thread_meta(&caller, &thread_id)
                    .await?;
                Ok(ToolOutput::new(None))
            }
            ThreadMetaToolMethod::AddParticipant => {
                let thread_id = ThreadId::from_str(&args.thread_id)?;
                let user_id = args.user_id.as_ref().ok_or("user_id is required")?;
                let user = Principal::from_text(user_id)?;
                let mut thread = self.management.get_thread_meta(&thread_id).await?;
                if thread.has_permission(&caller) {
                    thread.participants.insert(user);
                    Ok(ToolOutput::new(Some(thread)))
                } else {
                    Err(format!(
                        "caller {} does not have permission to add participant to the thread {}",
                        caller.to_text(),
                        thread_id
                    )
                    .into())
                }
            }
            ThreadMetaToolMethod::RemoveParticipant => {
                let thread_id = ThreadId::from_str(&args.thread_id)?;
                let user_id = args.user_id.as_ref().ok_or("user_id is required")?;
                let user = Principal::from_text(user_id)?;
                let mut thread = self.management.get_thread_meta(&thread_id).await?;
                if thread.has_permission(&caller) {
                    thread.participants.remove(&user);
                    Ok(ToolOutput::new(Some(thread)))
                } else {
                    Err(format!(
                        "caller {} does not have permission to remove participant from the thread {}",
                        caller.to_text(),
                        thread_id
                    )
                    .into())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineBuilder;

    #[tokio::test]
    async fn test_thread_meta_tool() {
        let engine = EngineBuilder::new();
        let ctx = engine.mock_ctx();
        let management = Arc::new(Management::new(&ctx.base, ctx.id()));

        let tool = ThreadMetaTool::new(management);
        let definition = tool.definition();
        println!("{}", serde_json::to_string_pretty(&definition).unwrap());
        // {
        //     "name": "sys_my_threads",
        //     "description": "Manages thread metadata for user. Thread is a conversation session between AI agents and user.",
        //     "parameters": {
        //       "additionalProperties": false,
        //       "properties": {
        //         "method": {
        //           "description": "The method to call.",
        //           "enum": [
        //             "get_thread_meta",
        //             "delete_thread_meta",
        //             "add_participant",
        //             "remove_participant"
        //           ],
        //           "type": "string"
        //         },
        //         "thread_id": {
        //           "description": "The thread ID, e.g. \"9z4e2mr0ui3e8a215n4g\".",
        //           "type": "string"
        //         },
        //         "user_id": {
        //           "description": "The user ID, e.g. \"77ibd-jp5kr-moeco-kgoar-rro5v-5tng4-krif5-5h2i6-osf2f-2sjtv-kqe\".",
        //           "type": [
        //             "string",
        //             "null"
        //           ]
        //         }
        //       },
        //       "required": [
        //         "method",
        //         "thread_id",
        //         "user_id"
        //       ],
        //       "title": "ThreadMetaToolArgs",
        //       "type": "object"
        //     },
        //     "strict": true
        // }
    }
}
