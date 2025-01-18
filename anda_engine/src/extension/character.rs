use anda_core::{
    evaluate_tokens, Agent, AgentContext, AgentOutput, BoxError, CacheExpiry, CacheFeatures,
    CompletionFeatures, CompletionRequest, Documents, Embedding, EmbeddingFeatures, Knowledge,
    KnowledgeFeatures, KnowledgeInput, Message, StateFeatures, VectorSearchFeatures,
};
use chrono::prelude::*;
use ic_cose_types::to_cbor_bytes;
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};

use super::{
    attention::{Attention, AttentionCommand, ContentQuality},
    segmenter::DocumentSegmenter,
};

use crate::{context::AgentCtx, store::MAX_STORE_OBJECT_SIZE};

const MAX_CHAT_HISTORY: usize = 42;
const CHAT_HISTORY_TTI: Duration = Duration::from_secs(3600 * 24 * 7);

/// Character definition structure containing all attributes and behavioral traits
/// For a complete, production-level character definition example, see:
/// https://github.com/ldclabs/anda/blob/main/agents/anda_bot/Character.toml
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Character {
    /// Character's display name, used for identification, e.g., "Anda ICP"
    pub name: String,

    /// Character's account or username, used for system identification and message routing
    pub username: String,

    /// Character's professional identity or role description, e.g., "Scientist and Prophet"
    pub identity: String,

    /// Character's backstory and historical background
    pub description: String,

    /// List of personality traits that define the character's behavior, e.g., brave, cunning, kind
    pub traits: Vec<String>,

    /// List of motivations and objectives that drive the character's actions
    pub goals: Vec<String>,

    /// List of expertise areas the character specializes in, e.g., "quantum physics", "time travel"
    pub topics: Vec<String>,

    /// Communication style and expression characteristics
    pub style: Style,

    /// Learning capabilities and adaptability configurations
    pub learning: Learning,
}

/// Defines the character's communication style and expression patterns
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Style {
    /// List of speech tones, e.g., formal, casual, humorous
    pub tone: Vec<String>,

    /// Communication style descriptions for chat interactions
    pub chat: Vec<String>,

    /// Communication style descriptions for post content
    pub post: Vec<String>,

    /// List of commonly used adjectives in character's speech
    pub adjectives: Vec<String>,

    /// List of key interests that the character focuses on
    pub interests: Vec<String>,

    /// List of meme phrases or internet slang the character uses
    pub meme_phrases: Vec<String>,
}

/// Defines the character's learning capabilities and adaptability
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Learning {
    /// List of active inquiry behaviors, describing questions or exploration directions
    pub active_inquiry: Vec<String>,

    /// Memory capacity description, defining the character's ability to retain context
    pub memory: String,

    /// Persona flexibility description, defining how the character adapts to user interaction styles
    pub persona_flexibility: String,

    /// List of required tools for the character to perform tasks (checked during agent registration)
    pub tools: Vec<String>,

    /// List of optional tools that the character can use but aren't required
    pub optional_tools: Vec<String>,
}

impl Character {
    pub fn from_toml(content: &str) -> Result<Self, BoxError> {
        let character: Self = toml::from_str(content)?;
        Ok(character)
    }

    pub fn to_toml(&self) -> Result<String, BoxError> {
        let content = toml::to_string(&self)?;
        Ok(content)
    }

    pub fn to_request(&self, prompt: String, prompter_name: Option<String>) -> CompletionRequest {
        let utc: DateTime<Utc> = Utc::now();
        let system = format!(
            "Character Definition:\n\
            Your name: {}\n\
            Your username: {}\n\
            Your identity: {}\n\
            Background: {}\n\
            Personality traits: {}\n\
            Motivations and goals: {}\n\
            Topics of expertise: {}\n\
            The current time is {}.\
            ",
            self.name,
            self.username,
            self.identity,
            self.description,
            self.traits.join(", "),
            self.goals.join(", "),
            self.topics.join(", "),
            utc.to_rfc3339_opts(SecondsFormat::Secs, true)
        );

        let style_context = format!(
            "Your personality and communication style:\n\
            - Tone of speech: {}\n\
            Communication style:\n\
            - In chat:\n{}\n\n\
            - In posts:\n{}\n\n\
            Expression elements:\n\
            - Common adjectives: {}\n\n\
            Personal elements:\n\
            - Key interests: {}\n\
            - Meme-related phrases: {}\
            ",
            self.style.tone.join(", "),
            self.style.chat.join("\n"),
            self.style.post.join("\n"),
            self.style.adjectives.join(", "),
            self.style.interests.join(", "),
            self.style.meme_phrases.join(", "),
        );

        let learning_context = format!(
            "Curiosity-driven behavior:\n{}\
            ",
            self.learning.active_inquiry.join("\n"),
        );

        CompletionRequest {
            system: Some(system),
            system_name: Some(self.name.clone()),
            prompt,
            prompter_name,
            ..Default::default()
        }
        .context("style_context".to_string(), style_context)
        .context("self_learning_context".to_string(), learning_context)
    }

    pub fn build<K: KnowledgeFeatures + VectorSearchFeatures>(
        self,
        attention: Attention,
        segmenter: DocumentSegmenter,
        knowledge: K,
    ) -> CharacterAgent<K> {
        CharacterAgent::new(self, attention, segmenter, knowledge)
    }
}

#[derive(Debug, Clone)]
pub struct CharacterAgent<K: KnowledgeFeatures + VectorSearchFeatures> {
    pub character: Arc<Character>,
    pub attention: Arc<Attention>,
    pub segmenter: Arc<DocumentSegmenter>,
    pub knowledge: Arc<K>,
}

impl<K: KnowledgeFeatures + VectorSearchFeatures> CharacterAgent<K> {
    pub fn new(
        character: Character,
        attention: Attention,
        segmenter: DocumentSegmenter,
        knowledge: K,
    ) -> Self {
        Self {
            character: Arc::new(character),
            attention: Arc::new(attention),
            segmenter: Arc::new(segmenter),
            knowledge: Arc::new(knowledge),
        }
    }

    pub async fn latest_knowledge(
        &self,
        last_seconds: u32,
        n: usize,
        user: Option<String>,
    ) -> Result<Vec<Knowledge>, BoxError> {
        self.knowledge
            .knowledge_latest_n(last_seconds, n, user)
            .await
    }
}

impl<K> Agent<AgentCtx> for CharacterAgent<K>
where
    K: KnowledgeFeatures + VectorSearchFeatures + Clone + Send + Sync + 'static,
{
    fn name(&self) -> String {
        self.character.username.clone()
    }

    fn description(&self) -> String {
        self.character.description.clone()
    }

    fn tool_dependencies(&self) -> Vec<String> {
        self.character.learning.tools.clone()
    }

    async fn run(
        &self,
        ctx: AgentCtx,
        prompt: String,
        _attachment: Option<Vec<u8>>,
    ) -> Result<AgentOutput, BoxError> {
        // read chat history from store
        let mut chat_history = if let Some(user) = ctx.user() {
            let chat: Vec<Message> = ctx
                .cache_get_with(&user, async {
                    Ok((Vec::new(), Some(CacheExpiry::TTI(CHAT_HISTORY_TTI))))
                })
                .await?;
            Some((user, chat))
        } else {
            None
        };

        let mut content_quality = ContentQuality::Ignore;
        if evaluate_tokens(&prompt) <= self.attention.min_content_tokens {
            let recent_messages: Vec<Message> = vec![];
            match self
                .attention
                .should_reply(
                    &ctx,
                    &self.character.username,
                    &self.character.topics,
                    chat_history
                        .as_ref()
                        .map(|(_, c)| c)
                        .unwrap_or(&recent_messages),
                    &Message {
                        role: "user".to_string(),
                        content: prompt.clone().into(),
                        name: ctx.user(),
                        ..Default::default()
                    },
                )
                .await
            {
                AttentionCommand::Stop | AttentionCommand::Ignore => {
                    return Ok(AgentOutput {
                        content: "I'm sorry, I will stop responding.".to_string(),
                        failed_reason: Some("STOP_COMMAND".to_string()),
                        ..Default::default()
                    });
                }
                _ => {}
            }
        } else {
            content_quality = self.attention.evaluate_content(&ctx, &prompt).await;
        }

        let knowledges: Documents = if content_quality == ContentQuality::Ignore {
            let knowledges = self.knowledge.top_n(&prompt, 5).await.unwrap_or_default();
            knowledges.into()
        } else {
            // do not append knowledges if content quality is high
            Documents::default()
        };

        if content_quality > ContentQuality::Ignore {
            let content = prompt.clone();
            let ctx = ctx.clone();
            let user = ctx.user().unwrap_or("anonymous".to_string());
            let segmenter = self.segmenter.clone();
            let knowledge = self.knowledge.clone();

            // save high quality content to knowledge store in background
            tokio::spawn(async move {
                let (docs, _) = segmenter.segment(&ctx, &content).await?;
                let mut vecs: Vec<Embedding> = Vec::with_capacity(docs.segments.len());
                for texts in docs.segments.chunks(16) {
                    match ctx.embed(texts.to_owned()).await {
                        Ok(embeddings) => vecs.extend(embeddings),
                        Err(err) => {
                            log::error!("Failed to embed segments: {}", err);
                        }
                    }
                }

                let docs: Vec<KnowledgeInput> = vecs
                    .into_iter()
                    .map(|embedding| KnowledgeInput {
                        user: user.clone(),
                        text: embedding.text,
                        meta: serde_json::Map::new().into(),
                        vec: embedding.vec,
                    })
                    .collect();
                let total = docs.len();
                if let Err(err) = knowledge.knowledge_add(docs).await {
                    log::error!("failed to add {} knowledges: {}", total, err);
                }

                Ok::<(), BoxError>(())
            });
        }

        let tools: Vec<&str> = self
            .character
            .learning
            .tools
            .iter()
            .chain(self.character.learning.optional_tools.iter())
            .map(|s| s.as_str())
            .collect();
        let tools = ctx.tool_definitions(Some(&tools));

        let mut req = self
            .character
            .to_request(prompt, ctx.user())
            .append_documents(knowledges)
            .append_tools(tools);

        if let Some((user, chat)) = &mut chat_history {
            req.chat_history = chat.clone();
            chat.push(Message {
                role: "user".to_string(),
                content: req.prompt.clone().into(),
                name: Some(user.clone()),
                ..Default::default()
            });

            // tools will be auto called in completion
            let res = ctx.completion(req).await?;
            if res.failed_reason.is_none() {
                if !res.content.is_empty() {
                    chat.push(Message {
                        role: "assistant".to_string(),
                        content: res.content.clone().into(),
                        ..Default::default()
                    });
                }
                if let Some(tool_calls) = &res.tool_calls {
                    for tool_res in tool_calls {
                        chat.push(Message {
                            role: "tool".to_string(),
                            content: "".to_string().into(),
                            name: Some(tool_res.name.clone()),
                            tool_call_id: Some(tool_res.id.clone()),
                        });
                    }
                }

                if chat.len() > MAX_CHAT_HISTORY {
                    chat.drain(0..(chat.len() - MAX_CHAT_HISTORY));
                }

                // save chat history to cache
                let data = to_cbor_bytes(&chat);
                let data = if data.len() < MAX_STORE_OBJECT_SIZE {
                    data
                } else {
                    chat.drain(0..(chat.len() / 2));
                    to_cbor_bytes(&chat)
                };
                if data.len() < MAX_STORE_OBJECT_SIZE {
                    let _ = ctx
                        .cache_set(user, (chat, Some(CacheExpiry::TTI(CHAT_HISTORY_TTI))))
                        .await;
                } else {
                    let _ = ctx.cache_delete(user).await;
                }
            }

            Ok(res)
        } else {
            ctx.completion(req).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_character_agent() {
        let character = Character {
            name: "Anda".to_string(),
            identity: "Scientist and Prophet".to_string(),
            description: "Anda is a scientist and prophet who has the ability to see the future."
                .to_string(),
            traits: vec![
                "brave".to_string(),
                "cunning".to_string(),
                "kind".to_string(),
            ],
            goals: vec![
                "save the world".to_string(),
                "prevent the apocalypse".to_string(),
            ],
            topics: vec!["quantum physics".to_string(), "time travel".to_string()],
            style: Style {
                tone: vec!["formal".to_string(), "casual".to_string()],
                chat: vec!["friendly".to_string(), "helpful".to_string()],
                post: vec!["insightful".to_string(), "thought-provoking".to_string()],
                adjectives: vec!["brave".to_string(), "cunning".to_string()],
                interests: vec!["quantum physics".to_string(), "time travel".to_string()],
                meme_phrases: vec![
                    "I have seen the future".to_string(),
                    "The end is near".to_string(),
                ],
            },
            learning: Learning {
                active_inquiry: vec!["What is the future?".to_string()],
                memory: "Unlimited".to_string(),
                persona_flexibility: "Dynamic".to_string(),
                tools: vec!["submit_character".to_string()],
                optional_tools: vec!["submit_character".to_string()],
            },
            ..Default::default()
        };
        let req = character.to_request("Who are you?".to_string(), None);
        println!("{}\n", req.system.as_ref().unwrap());
        println!("{}\n", req.prompt_with_context());
        println!("{:?}", req.tools);
    }
}
