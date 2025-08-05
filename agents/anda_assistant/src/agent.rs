use anda_cognitive_nexus::{CognitiveNexus, ConceptPK};
use anda_core::{
    Agent, AgentContext, AgentOutput, BoxError, CompletionFeatures, CompletionRequest, Json,
    Resource, StateFeatures, Tool, ToolSet,
};
use anda_db::database::AndaDB;
use anda_engine::{
    ANONYMOUS,
    context::{AgentCtx, BaseCtx, Web3SDK},
    memory::{ChatRef, MemoryManagement},
    unix_ms,
};
use anda_kip::{
    META_SELF_NAME, PERSON_SELF_KIP, PERSON_SYSTEM_KIP, PERSON_TYPE, SYSTEM_INSTRUCTIONS, parse_kml,
};
use chrono::prelude::*;
use std::sync::Arc;

/// An AI agent implementation for interacting with ICP blockchain ledgers.
/// This agent provides capabilities to check balances and transfer ICP tokens
/// using the provided tools.
#[derive(Clone)]
pub struct Assistant {
    max_input_tokens: usize,
    memory: Arc<MemoryManagement>,
    tools: Vec<String>,
}

static EMPTY_MESSAGES: &[Json] = &[];

impl Assistant {
    pub const NAME: &'static str = "assistant";
    pub async fn connect<F>(web3: Arc<Web3SDK>, db: Arc<AndaDB>) -> Result<Self, BoxError> {
        let my_id = web3.get_principal();
        let nexus = CognitiveNexus::connect(db.clone(), async |nexus| {
            if !nexus
                .has_concept(&ConceptPK::Object {
                    r#type: PERSON_TYPE.to_string(),
                    name: META_SELF_NAME.to_string(),
                })
                .await
            {
                let kml = &[
                    &PERSON_SELF_KIP
                        .replace("$self_reserved_principal_id", my_id.to_string().as_str()),
                    PERSON_SYSTEM_KIP,
                ]
                .join("\n");

                let result = nexus.execute_kml(parse_kml(kml)?, false).await?;
                log::info!(result:serde = result; "Init $self and $system");
            }

            Ok(())
        })
        .await?;
        let memory = Arc::new(MemoryManagement::connect(Arc::new(nexus), db).await?);
        let tool_name = memory.name();

        Ok(Self {
            max_input_tokens: 65535,
            memory,
            tools: vec![tool_name],
        })
    }

    pub fn with_max_input_tokens(&mut self, max_input_tokens: usize) -> &mut Self {
        self.max_input_tokens = max_input_tokens;
        self
    }

    pub fn tools(&self) -> Result<ToolSet<BaseCtx>, BoxError> {
        let mut tools = ToolSet::new();
        tools.add(self.memory.clone())?;
        Ok(tools)
    }
}

/// Implementation of the [`Agent`] trait for Assistant.
impl Agent<AgentCtx> for Assistant {
    /// Returns the agent's name identifier
    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    /// Returns a description of the agent's purpose and capabilities.
    fn description(&self) -> String {
        "AI assistant powered by the Knowledge Interaction Protocol (KIP)".to_string()
    }

    /// Returns a list of tool names that this agent depends on
    fn tool_dependencies(&self) -> Vec<String> {
        self.tools.clone()
    }

    /// Main execution method for the agent.
    ///
    /// # Arguments
    /// * `ctx` - The agent context containing execution environment.
    /// * `prompt` - The user's input prompt.
    /// * `resources`: Optional additional resources (not used).
    ///
    /// # Returns
    /// AgentOutput containing the response or an error if execution fails.
    async fn run(
        &self,
        ctx: AgentCtx,
        prompt: String,
        resources: Option<Vec<Resource>>,
    ) -> Result<AgentOutput, BoxError> {
        let caller = ctx.caller();
        if caller == &ANONYMOUS {
            return Err("anonymous caller not allowed".into());
        }

        let start_time = unix_ms();
        let utc: DateTime<Utc> = DateTime::from_timestamp_millis(start_time as i64).unwrap();
        let primer = self.memory.describe_primer().await?;
        let chats = self
            .memory
            .list_chats_by_user(caller, None, Some(42))
            .await?;
        let mut chat_history = chats
            .into_iter()
            .map(|chat| chat.messages)
            .collect::<Vec<_>>();
        let max_history_bytes = (self.max_input_tokens / 2) * 3; // Rough estimate of bytes per token
        let mut writer: Vec<u8> = Vec::with_capacity(128);
        let _ = serde_json::to_writer(&mut writer, &chat_history);
        let mut history_bytes = writer.len();
        while history_bytes > max_history_bytes && !chat_history.is_empty() {
            writer.clear();
            let _ = serde_json::to_writer(&mut writer, &chat_history.remove(0));
            history_bytes = history_bytes.saturating_sub(writer.len());
        }

        let chat_history: Vec<Json> = chat_history.into_iter().flatten().collect();
        let chat_history_len = chat_history.len();
        let req = CompletionRequest {
            system: Some(format!(
                "{}\n{}\n---\nCurrent Time: {}",
                SYSTEM_INSTRUCTIONS,
                primer,
                utc.to_rfc3339()
            )),
            prompt,
            prompter_name: Some(format!("{}", caller)),
            chat_history,
            tools: ctx.tool_definitions(Some(
                &self.tools.iter().map(|v| v.as_str()).collect::<Vec<_>>(),
            )),
            tool_choice_required: false,
            ..Default::default()
        };

        let res = ctx.completion(req, None).await?;
        let end_time = unix_ms();
        let messages = res
            .full_history
            .as_ref()
            .map(|msgs| {
                if msgs.len() > chat_history_len {
                    &msgs[chat_history_len..]
                } else {
                    msgs
                }
            })
            .unwrap_or(EMPTY_MESSAGES);
        let chat = ChatRef {
            _id: 0, // This will be set by the database
            user: caller,
            thread: None,
            messages,
            resources: resources.as_ref().map(|v| v.as_ref()),
            artifacts: res.artifacts.as_ref().map(|v| v.as_ref()),
            period: end_time / 3600 / 1000,
            start_time,
            end_time,
        };
        let _ = self.memory.add_chat(&chat).await;
        Ok(res)
    }
}
