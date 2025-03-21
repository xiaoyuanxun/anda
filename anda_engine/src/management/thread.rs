use anda_core::{
    ANONYMOUS, BoxError, FunctionDefinition, Resource, StateFeatures, ThreadMeta, Tool, ToolOutput,
    Value, Xid, gen_schema_for,
};
use candid::Principal;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{str::FromStr, sync::Arc};

use super::Management;
use crate::context::BaseCtx;

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
        if caller == ANONYMOUS {
            return Err("anonymous user is not allowed".into());
        }

        match args.method {
            ThreadMetaToolMethod::GetThreadMeta => {
                let thread_id = Xid::from_str(&args.thread_id)?;
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
                let thread_id = Xid::from_str(&args.thread_id)?;
                self.management
                    .delete_thread_meta(&caller, &thread_id)
                    .await?;
                Ok(ToolOutput::new(None))
            }

            ThreadMetaToolMethod::AddParticipant => {
                let thread_id = Xid::from_str(&args.thread_id)?;
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
                let thread_id = Xid::from_str(&args.thread_id)?;
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
    use crate::{
        engine::EngineBuilder,
        management::{ManagementBuilder, Visibility},
    };

    #[tokio::test]
    async fn test_thread_meta_tool() {
        let engine = EngineBuilder::new();
        let ctx = engine.mock_ctx();
        let management =
            Arc::new(ManagementBuilder::new(Visibility::Protected, ctx.id()).build(&ctx.base));

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
