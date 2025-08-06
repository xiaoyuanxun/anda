use anda_cognitive_nexus::{CognitiveNexus, ConceptPK};
use anda_core::{
    Agent, AgentContext, AgentOutput, BoxError, CompletionFeatures, CompletionRequest, Document,
    Json, Resource, ResourceRef, StateFeatures, Tool, ToolSet, evaluate_tokens,
};
use anda_db::{database::AndaDB, error::DBError};
use anda_engine::{
    ANONYMOUS,
    context::{AgentCtx, BaseCtx, Web3SDK},
    extension::fetch::FetchWebResourcesTool,
    memory::{
        ChatRef, GetResourceTool, ListConversationsTool, MemoryManagement, SearchConversationsTool,
    },
    unix_ms,
};
use anda_kip::{
    EVENT_KIP, META_SELF_NAME, PERSON_SELF_KIP, PERSON_SYSTEM_KIP, PERSON_TYPE,
    SYSTEM_INSTRUCTIONS, parse_kml,
};
use chrono::prelude::*;
use ic_cose_types::cose::sha3_256;
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
                    EVENT_KIP,
                ]
                .join("\n");

                let result = nexus.execute_kml(parse_kml(kml)?, false).await?;
                log::info!(result:serde = result; "Init $self and $system");
            }

            Ok(())
        })
        .await?;

        let memory = Arc::new(MemoryManagement::connect(Arc::new(nexus), db).await?);
        let memory_name = memory.name();

        Ok(Self {
            max_input_tokens: 65535,
            memory,
            tools: vec![
                memory_name,
                SearchConversationsTool::NAME.to_string(),
                ListConversationsTool::NAME.to_string(),
                GetResourceTool::NAME.to_string(),
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
        tools.add(GetResourceTool::new(self.memory.clone()))?;
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
        mut resources: Vec<Resource>,
    ) -> Result<AgentOutput, BoxError> {
        let caller = ctx.caller();
        if caller == &ANONYMOUS {
            return Err("anonymous caller not allowed".into());
        }

        let start_time = unix_ms();
        let utc: DateTime<Utc> = DateTime::from_timestamp_millis(start_time as i64).unwrap();
        let utc = utc.to_rfc3339();
        let primer = self.memory.describe_primer().await?;
        let system = format!(
            "{}\n---\n# Your Identity & Knowledge Domain Map\n{}\n---\n# Current Time: {}",
            SYSTEM_INSTRUCTIONS, primer, utc
        );

        let (chats, cursor) = self
            .memory
            .list_chats_by_user(caller, None, Some(3))
            .await?;
        let mut chat_history = chats
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
        let mut rs: Vec<Resource> = Vec::with_capacity(resources.len());
        let mut docs: Vec<Document> = Vec::with_capacity(resources.len());
        for r in resources.iter_mut() {
            if let Some(blob) = &r.blob {
                r.hash = Some(sha3_256(blob).into());
            }
            // TODO: check when no blob

            if r.metadata.is_none() {
                r.metadata = Some(serde_json::Map::new());
            }

            let meta = r.metadata.as_mut().unwrap();
            meta.insert("user".to_string(), caller.to_string().into());
            meta.insert("created_at".to_string(), utc.clone().into());

            let rf: ResourceRef = (r as &Resource).into();
            let id = if r._id > 0 {
                r._id // TODO: check if the resource exists and has permission
            } else {
                match self.memory.add_resource(&rf).await {
                    Ok(id) => id,
                    Err(DBError::AlreadyExists { _id, .. }) => _id,
                    Err(err) => Err(err)?,
                }
            };

            let r2 = Resource {
                _id: id,
                blob: None,
                ..r.clone()
            };
            docs.push((&r2).into());
            rs.push(r2)
        }

        if !rs.is_empty() {
            self.memory.flush_resources().await?;
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

        let res = ctx.completion(req, resources).await?;
        let end_time = unix_ms();
        let messages = if res.full_history.len() > chat_history_len {
            &res.full_history[chat_history_len..]
        } else {
            &res.full_history
        };
        let chat = ChatRef {
            _id: 0, // This will be set by the database
            user: caller,
            thread: None,
            messages,
            resources: &rs,
            artifacts: &res.artifacts,
            period: end_time / 3600 / 1000,
            start_time,
            end_time,
        };
        let _ = self.memory.add_chat(&chat).await;
        Ok(res)
    }
}
