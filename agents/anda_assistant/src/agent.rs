use anda_cognitive_nexus::{CognitiveNexus, ConceptPK};
use anda_core::{
    Agent, AgentContext, AgentOutput, BoxError, CompletionRequest, Document, Documents, Json,
    Message, Principal, Resource, StateFeatures, Tool, ToolSet, Usage, evaluate_tokens,
    update_resources,
};
use anda_db::{database::AndaDB, index::BTree};
use anda_engine::{
    ANONYMOUS,
    context::{AgentCtx, BaseCtx},
    extension::fetch::FetchWebResourcesTool,
    json_set_unix_ms_timestamp,
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

    pub fn with_max_input_tokens(mut self, max_input_tokens: usize) -> Self {
        self.max_input_tokens = max_input_tokens;
        self
    }

    pub fn with_system_instructions(mut self, instructions: &str) -> Self {
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

    pub async fn caller_name(&self, caller: &Principal) -> Option<String> {
        if let Ok(concept) = self
            .memory
            .nexus()
            .get_concept(&ConceptPK::Object {
                r#type: PERSON_TYPE.to_string(),
                name: caller.to_string(),
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

        let user_name = ctx
            .meta()
            .user
            .clone()
            .unwrap_or_else(|| caller.to_string());
        let caller_info = self
            .memory
            .describe_caller(caller)
            .await
            .unwrap_or_else(|_| {
                serde_json::json!({
                    "id": caller.to_string(),
                    "name": user_name
                })
            });

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
        let max_history_bytes = self.max_input_tokens.saturating_sub(
            ((evaluate_tokens(&system) + evaluate_tokens(&prompt)) as f64 * 1.2) as usize,
        ) * 3; // Rough estimate of bytes per token
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

        let mut history_docs: Vec<Document> = Vec::with_capacity(conversations.len() + 2);
        history_docs.push(Document {
            content: caller_info,
            metadata: BTreeMap::from([
                ("_type".to_string(), "User".into()),
                ("description".to_string(), "User identity".into()),
            ]),
        });
        history_docs.extend(conversations.into_iter().map(Document::from));
        if let Some(cursor) = cursor {
            history_docs.push(Document {
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

        let mut chat_history: Vec<Json> = vec![];
        chat_history.push(serde_json::json!(Message {
            role: "user".into(),
            content: format!(
                "Current Datetime: {}\n---\n{}",
                rfc3339_datetime_now(),
                Documents::from(history_docs)
            )
            .into(),
            name: Some("$system".into()),
            ..Default::default()
        }));

        let resources = update_resources(caller, resources);
        let rs = self.memory.try_add_resources(&resources).await?;
        let resource_docs: Vec<Document> = rs.iter().map(Document::from).collect();

        let mut conversation = Conversation {
            _id: 0,
            user: *caller,
            thread: None,
            messages: json_set_unix_ms_timestamp(
                vec![serde_json::json!(Message {
                    role: "user".into(),
                    content: prompt.clone().into(),
                    ..Default::default()
                })],
                created_at,
            ),
            resources: rs,
            artifacts: vec![],
            status: ConversationStatus::Submitted,
            period: created_at / 3600 / 1000,
            created_at,
            updated_at: created_at,
            usage: Usage::default(),
        };

        let id = self
            .memory
            .add_conversation(ConversationRef::from(&conversation))
            .await?;
        conversation._id = id;
        ctx.base.set_state(ConversationState::from(&conversation));
        let res = AgentOutput {
            conversation: Some(id),
            ..Default::default()
        };

        let assistant = self.clone();
        let mut runner = ctx.completion_iter(
            CompletionRequest {
                system,
                prompt,
                chat_history,
                documents: resource_docs.into(),
                tools: ctx.tool_definitions(Some(
                    &self.tools.iter().map(|v| v.as_str()).collect::<Vec<_>>(),
                )),
                tool_choice_required: false,
                ..Default::default()
            },
            resources,
        );

        tokio::spawn(async move {
            let mut rt = async || {
                let mut first_round = true;
                loop {
                    match runner.next().await {
                        Ok(None) => break,
                        Ok(Some(mut res)) => {
                            let now_ms = unix_ms();
                            let artifacts =
                                assistant.memory.try_add_resources(&res.artifacts).await?;

                            if first_round {
                                first_round = false;
                                let response = res.full_history.pop().ok_or(
                                    "No response message in the first round of completion",
                                )?;
                                conversation.messages.clear(); // clear the first pending message.
                                conversation
                                    .append_messages(clear_messages(res.full_history), created_at);
                                conversation
                                    .append_messages(clear_messages(vec![response]), now_ms);
                            } else {
                                res.full_history.drain(0..conversation.messages.len());
                                conversation
                                    .append_messages(clear_messages(res.full_history), now_ms);
                            }

                            conversation.artifacts = artifacts;
                            conversation.status = if runner.is_done() {
                                ConversationStatus::Completed
                            } else if res.failed_reason.is_some() {
                                ConversationStatus::Failed
                            } else {
                                ConversationStatus::Working
                            };
                            conversation.usage = res.usage;
                            conversation.updated_at = now_ms;

                            if let Some(failed_reason) = res.failed_reason {
                                conversation.append_messages(
                                    vec![serde_json::json!(Message {
                                        role: "assistant".into(),
                                        content: failed_reason.into(),
                                        name: Some("$system".into()),
                                        ..Default::default()
                                    })],
                                    now_ms,
                                );
                            }

                            let old = assistant.memory.get_conversation(conversation._id).await?;
                            if old.status == ConversationStatus::Canceled
                                && (conversation.status == ConversationStatus::Submitted
                                    || conversation.status == ConversationStatus::Working)
                            {
                                conversation.status = ConversationStatus::Canceled;
                            }

                            let _ = assistant
                                .memory
                                .update_conversation(id, conversation.to_changes()?)
                                .await;

                            ctx.base.set_state(ConversationState::from(&conversation));

                            if conversation.status == ConversationStatus::Canceled
                                || conversation.status == ConversationStatus::Failed
                            {
                                break;
                            }
                        }
                        Err(err) => {
                            log::error!("Conversation {id} in CompletionRunner error: {:?}", err);
                            let now_ms = unix_ms();
                            conversation.append_messages(
                                vec![serde_json::json!(Message {
                                    role: "assistant".into(),
                                    content: err.to_string().into(),
                                    name: Some("$system_warning".into()),
                                    ..Default::default()
                                })],
                                now_ms,
                            );

                            conversation.status = ConversationStatus::Failed;
                            conversation.updated_at = now_ms;
                            let _ = assistant
                                .memory
                                .update_conversation(id, conversation.to_changes()?)
                                .await;

                            ctx.base.set_state(ConversationState::from(&conversation));
                            break;
                        }
                    }
                }

                Ok::<(), BoxError>(())
            };

            match rt().await {
                Ok(_) => {}
                Err(err) => {
                    log::error!("Error occurred in conversation {id}: {:?}", err);
                }
            }
        });

        Ok(res)
    }
}

// clear Gemini's messages
fn clear_messages(mut messages: Vec<Json>) -> Vec<Json> {
    for msg in messages.iter_mut() {
        if let Some(content) = msg.get_mut("content")
            && let Some(parts) = content.as_array_mut()
        {
            for part in parts.iter_mut() {
                if let Some(part) = part.as_object_mut() {
                    part.remove("thoughtSignature"); // thoughtSignature is verbose
                }
            }
        }
    }

    messages
}
