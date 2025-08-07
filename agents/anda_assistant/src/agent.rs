use anda_cognitive_nexus::{CognitiveNexus, ConceptPK};
use anda_core::{
    Agent, AgentContext, AgentOutput, BoxError, CompletionFeatures, CompletionRequest, Document,
    Json, Resource, StateFeatures, Tool, ToolSet, evaluate_tokens, update_resources,
};
use anda_db::database::AndaDB;
use anda_db_schema::{Ft, Fv};
use anda_engine::{
    ANONYMOUS,
    context::{AgentCtx, BaseCtx, Web3SDK},
    extension::fetch::FetchWebResourcesTool,
    memory::{
        ConversationRef, ConversationStatus, GetResourceContentTool, ListConversationsTool,
        MemoryManagement, SearchConversationsTool,
    },
    unix_ms,
};
use anda_kip::{
    EVENT_KIP, META_SELF_NAME, PERSON_SELF_KIP, PERSON_SYSTEM_KIP, PERSON_TYPE,
    SYSTEM_INSTRUCTIONS, parse_kml,
};
use chrono::prelude::*;
use ciborium::cbor;
use std::{collections::BTreeMap, sync::Arc};

/// An AI agent implementation for interacting with ICP blockchain ledgers.
/// This agent provides capabilities to check balances and transfer ICP tokens
/// using the provided tools.
#[derive(Clone)]
pub struct Assistant {
    max_input_tokens: usize,
    memory: Arc<MemoryManagement>,
    tools: Vec<String>,
}

impl Assistant {
    pub const NAME: &'static str = "assistant";
    pub async fn connect<F>(db: Arc<AndaDB>, web3: Arc<Web3SDK>) -> Result<Self, BoxError> {
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
                    EVENT_KIP,
                ]
                .join("\n");

                let result = nexus.execute_kml(parse_kml(kml)?, false).await?;
                log::info!(result:serde = result; "Init $self and $system");
            }

            Ok(())
        })
        .await?;

        let memory = Arc::new(MemoryManagement::connect(db, Arc::new(nexus)).await?);
        let memory_name = memory.name();

        Ok(Self {
            max_input_tokens: 65535,
            memory,
            tools: vec![
                memory_name,
                SearchConversationsTool::NAME.to_string(),
                ListConversationsTool::NAME.to_string(),
                GetResourceContentTool::NAME.to_string(),
                FetchWebResourcesTool::NAME.to_string(),
            ],
        })
    }

    pub fn with_max_input_tokens(&mut self, max_input_tokens: usize) -> &mut Self {
        self.max_input_tokens = max_input_tokens;
        self
    }

    pub fn tools(&self) -> Result<ToolSet<BaseCtx>, BoxError> {
        let mut tools = ToolSet::new();
        tools.add(self.memory.clone())?;
        tools.add(SearchConversationsTool::new(self.memory.clone()))?;
        tools.add(ListConversationsTool::new(self.memory.clone()))?;
        tools.add(GetResourceContentTool::new(self.memory.clone()))?;
        tools.add(FetchWebResourcesTool::new())?;
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

    fn supported_resource_tags(&self) -> Vec<String> {
        vec!["text".to_string()]
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
        resources: Vec<Resource>,
    ) -> Result<AgentOutput, BoxError> {
        let caller = ctx.caller();
        if caller == &ANONYMOUS {
            return Err("anonymous caller not allowed".into());
        }

        let created_at = unix_ms();
        let utc: DateTime<Utc> = DateTime::from_timestamp_millis(created_at as i64).unwrap();
        let utc = utc.to_rfc3339();
        let primer = self.memory.describe_primer().await?;
        let system = format!(
            "{}\n---\n# Your Identity & Knowledge Domain Map\n{}\n---\n# Current Time: {}",
            SYSTEM_INSTRUCTIONS, primer, utc
        );

        let (conversations, cursor) = self
            .memory
            .list_conversations_by_user(caller, None, Some(3))
            .await?;
        let mut chat_history = conversations
            .into_iter()
            .map(|chat| chat.messages)
            .collect::<Vec<_>>();
        let max_history_bytes = self
            .max_input_tokens
            .saturating_sub((evaluate_tokens(&system) + evaluate_tokens(&prompt)) * 3)
            * 3; // Rough estimate of bytes per token
        let mut writer: Vec<u8> = Vec::with_capacity(256);
        let _ = serde_json::to_writer(&mut writer, &chat_history);
        let mut history_bytes = writer.len();
        while history_bytes > max_history_bytes && !chat_history.is_empty() {
            writer.clear();
            let _ = serde_json::to_writer(&mut writer, &chat_history.remove(0));
            history_bytes = history_bytes.saturating_sub(writer.len());
        }

        let chat_history: Vec<Json> = chat_history.into_iter().flatten().collect();
        let chat_history_len = chat_history.len();
        let resources = update_resources(caller, resources);
        let rs = self.memory.try_add_resources(&resources).await?;

        let mut docs: Vec<Document> = Vec::with_capacity(resources.len());
        for r in rs.iter() {
            docs.push(r.into());
        }

        if let Some(cursor) = cursor {
            docs.push(Document {
                content: cursor.into(),
                metadata: BTreeMap::from([
                    ("_type".to_string(), "Cursor".into()),
                    (
                        "description".to_string(),
                        "List previous conversations with this cursor".into(),
                    ),
                ]),
            })
        }

        let mut conversation = ConversationRef {
            _id: 0,
            user: caller,
            thread: None,
            messages: &[],
            resources: &rs,
            artifacts: &[],
            status: ConversationStatus::Working,
            period: created_at / 3600 / 1000,
            created_at,
            updated_at: created_at,
        };

        let id = self.memory.add_conversation(&conversation).await?;
        conversation._id = id;

        let req = CompletionRequest {
            system,
            prompt,
            prompter_name: Some(format!("{}", caller)),
            chat_history,
            documents: docs.into(),
            tools: ctx.tool_definitions(Some(
                &self.tools.iter().map(|v| v.as_str()).collect::<Vec<_>>(),
            )),
            tool_choice_required: false,
            ..Default::default()
        };

        let mut res = ctx.completion(req, resources).await?;
        res.conversation = Some(id);

        let artifacts = self.memory.try_add_resources(&res.artifacts).await?;

        conversation.messages = if res.full_history.len() > chat_history_len {
            &res.full_history[chat_history_len..]
        } else {
            &res.full_history
        };
        conversation.artifacts = &artifacts;
        conversation.status = ConversationStatus::Completed;
        conversation.updated_at = unix_ms();

        let _ = self
            .memory
            .update_conversation(
                id,
                BTreeMap::from([
                    (
                        "messages".to_string(),
                        Fv::array_from(cbor!(conversation.messages).unwrap(), &[Ft::Json])?,
                    ),
                    (
                        "artifacts".to_string(),
                        Fv::array_from(
                            cbor!(conversation.artifacts).unwrap(),
                            &[Resource::field_type()],
                        )?,
                    ),
                    (
                        "status".to_string(),
                        Fv::Text(conversation.status.to_string()),
                    ),
                    ("updated_at".to_string(), Fv::U64(conversation.updated_at)),
                ]),
            )
            .await;
        Ok(res)
    }
}
