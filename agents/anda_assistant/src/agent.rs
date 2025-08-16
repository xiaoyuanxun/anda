use anda_cognitive_nexus::{CognitiveNexus, ConceptPK};
use anda_core::{
    Agent, AgentContext, AgentOutput, BoxError, CompletionRequest, Document, Documents, Json,
    Message, Principal, Resource, StateFeatures, Tool, ToolSet, evaluate_tokens, update_resources,
};
use anda_db::{database::AndaDB, index::BTree};
use anda_engine::{
    ANONYMOUS,
    context::{AgentCtx, BaseCtx},
    extension::fetch::FetchWebResourcesTool,
    memory::{
        Conversation, ConversationRef, ConversationState, ConversationStatus,
        GetResourceContentTool, ListConversationsTool, MemoryManagement, SearchConversationsTool,
    },
    rfc3339_datetime_now, unix_ms,
};
use anda_kip::{
    META_SELF_NAME, PERSON_SELF_KIP, PERSON_SYSTEM_KIP, PERSON_TYPE, SYSTEM_INSTRUCTIONS, parse_kml,
};
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone)]
pub struct Assistant {
    max_input_tokens: usize,
    memory: Arc<MemoryManagement>,
    tools: Vec<String>,
    system_instructions: String,
}

impl Assistant {
    pub const NAME: &'static str = "assistant";
    pub async fn connect(db: Arc<AndaDB>, id: Option<Principal>) -> Result<Self, BoxError> {
        let id = id
            .map(|v| v.to_string())
            .unwrap_or_else(|| "uuc56-gyb".to_string()); // Principal::from_slice(&[1])
        let nexus = CognitiveNexus::connect(db.clone(), async |nexus| {
            if !nexus
                .has_concept(&ConceptPK::Object {
                    r#type: PERSON_TYPE.to_string(),
                    name: META_SELF_NAME.to_string(),
                })
                .await
            {
                let kml = &[
                    &PERSON_SELF_KIP.replace("$self_reserved_principal_id", &id),
                    PERSON_SYSTEM_KIP,
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
            system_instructions: SYSTEM_INSTRUCTIONS.to_string(),
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

    pub fn with_system_instructions(&mut self, instructions: &str) -> &mut Self {
        self.system_instructions = instructions.to_string();
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

    pub fn memory(&self) -> Arc<MemoryManagement> {
        self.memory.clone()
    }

    pub async fn to_kip_system_role_instructions(&self) -> Result<String, BoxError> {
        let system = self.memory.describe_system().await?;

        Ok(format!(
            "{}\n---\n# Your Identity & Knowledge Domain\n{}",
            SYSTEM_INSTRUCTIONS, system
        ))
    }

    pub async fn self_name(&self) -> Option<String> {
        if let Ok(concept) = self
            .memory
            .nexus()
            .get_concept(&ConceptPK::Object {
                r#type: PERSON_TYPE.to_string(),
                name: META_SELF_NAME.to_string(),
            })
            .await
        {
            concept
                .attributes
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        } else {
            None
        }
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
        let primer = self.memory.describe_primer().await?;
        let system = format!(
            "{}\n---\n# Your Identity & Knowledge Domain Map\n{}\n",
            SYSTEM_INSTRUCTIONS, primer
        );

        let (mut conversations, mut cursor) = self
            .memory
            .list_conversations_by_user(caller, None, Some(7))
            .await?;
        let max_history_bytes = self
            .max_input_tokens
            .saturating_sub((evaluate_tokens(&system) + evaluate_tokens(&prompt)) * 2)
            * 3; // Rough estimate of bytes per token
        let mut writer: Vec<u8> = Vec::with_capacity(256);
        let _ = serde_json::to_writer(&mut writer, &conversations);
        let mut history_bytes = writer.len();
        while history_bytes > max_history_bytes && !conversations.is_empty() {
            writer.clear();
            let conv = conversations.remove(0);
            cursor = BTree::to_cursor(&conv._id);
            let _ = serde_json::to_writer(&mut writer, &conv);
            history_bytes = history_bytes.saturating_sub(writer.len());
        }

        let mut chat_history: Vec<Json> = vec![];
        if !conversations.is_empty() {
            let docs: Vec<Document> = conversations.iter().map(Document::from).collect();
            let content = format!(
                "Current Datetime: {}\nPrevious Conversations: \n---\n{}",
                rfc3339_datetime_now(),
                Documents::from(docs)
            );
            chat_history.push(serde_json::json!(Message {
                role: "tool".into(),
                content: content.into(),
                ..Default::default()
            }));
        }

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

        let mut conversation = Conversation {
            _id: 0,
            user: *caller,
            thread: None,
            messages: vec![],
            resources: resources.clone(),
            artifacts: vec![],
            status: ConversationStatus::Working,
            period: created_at / 3600 / 1000,
            created_at,
            updated_at: created_at,
        };

        let id = self
            .memory
            .add_conversation(ConversationRef::from(&conversation))
            .await?;
        conversation._id = id;
        ctx.base.set_state(ConversationState::from(&conversation));

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

        let mut runner = ctx.completion_iter(req, resources);

        let mut res = runner.next().await?.unwrap(); // 理论上一定有输出
        res.conversation = Some(id);

        let artifacts = self.memory.try_add_resources(&res.artifacts).await?;

        conversation.messages = res.full_history.clone();
        conversation.artifacts = artifacts;
        conversation.status = if runner.is_done() {
            ConversationStatus::Completed
        } else if res.failed_reason.is_some() {
            ConversationStatus::Failed
        } else {
            ConversationStatus::Working
        };
        conversation.updated_at = unix_ms();

        let _ = self
            .memory
            .update_conversation(id, conversation.to_changes()?)
            .await;

        ctx.base.set_state(ConversationState::from(&conversation));

        let assistant = self.clone();
        tokio::spawn(async move {
            while let Some(res) = runner.next().await.map_err(|err| {
                log::error!("Conversation {id} in CompletionRunner error: {:?}", err);
                err
            })? {
                let artifacts = assistant.memory.try_add_resources(&res.artifacts).await?;

                conversation.messages = res.full_history.clone();
                conversation.artifacts = artifacts;
                conversation.status = if runner.is_done() {
                    ConversationStatus::Completed
                } else if res.failed_reason.is_some() {
                    ConversationStatus::Failed
                } else {
                    ConversationStatus::Working
                };
                conversation.updated_at = unix_ms();

                let _ = assistant
                    .memory
                    .update_conversation(id, conversation.to_changes()?)
                    .await;

                ctx.base.set_state(ConversationState::from(&conversation));

                if res.failed_reason.is_some() {
                    break;
                }
            }

            Ok::<(), BoxError>(())
        });

        Ok(res)
    }
}
