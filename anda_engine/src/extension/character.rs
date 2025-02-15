//! Character definition and agent implementation for AI personalities
//!
//! This module provides the core structures and implementations for defining and interacting with
//! AI characters. It includes:
//! - Character definition structure with personality traits, communication styles, and learning capabilities
//! - Character agent implementation that handles interactions and maintains state
//! - Integration with knowledge bases and attention mechanisms
//!
//! # Key Components
//! - [`Character`]: Defines the personality, traits, and capabilities of an AI agent
//! - [`Style`]: Defines communication patterns and expression characteristics
//! - [`Learning`]: Configures learning capabilities and tool dependencies
//! - [`CharacterAgent`]: Implements the Agent trait for character-based interactions
//!
//! # Usage
//! 1. Define a character using the Character structure
//! 2. Create a CharacterAgent instance with required dependencies
//! 3. Use the agent to handle user interactions and maintain conversation state
//!
//! # Example
//! ```rust,ignore
//! let character = Character {
//!     name: "ExampleBot".to_string(),
//!     // ... other fields ...
//! };
//! let agent = character.build(attention, segmenter, knowledge);
//! let output = agent.run(ctx, "Hello".to_string(), None).await?;
//! ```

use anda_core::{
    evaluate_tokens, Agent, AgentContext, AgentOutput, BoxError, CacheExpiry, CacheFeatures,
    CompletionFeatures, CompletionRequest, Documents, Embedding, EmbeddingFeatures, Knowledge,
    KnowledgeFeatures, KnowledgeInput, Message, StateFeatures, VectorSearchFeatures,
};
use ic_cose_types::to_cbor_bytes;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{fmt::Write, sync::Arc, time::Duration};

use super::{
    attention::{Attention, AttentionCommand, ContentQuality},
    segmenter::DocumentSegmenter,
};

use crate::{context::AgentCtx, store::MAX_STORE_OBJECT_SIZE};

const MAX_CHAT_HISTORY: usize = 42;
const CHAT_HISTORY_TTI: Duration = Duration::from_secs(3600 * 24 * 7);

/// Represents a character definition with attributes, traits, and behaviors
/// Contains all necessary information to define an AI agent's personality and capabilities.
///
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

    /// List of mechanics or learning strategies the character uses to evolve
    pub mechanics: Vec<String>,

    /// Tools that the character uses to complete tasks.
    /// These tools will be checked for availability when registering the agent.
    pub tools: Vec<String>,

    /// Optional tools that the character uses to complete tasks.
    pub optional_tools: Vec<String>,
}

impl Character {
    /// Creates a Character instance from TOML formatted string
    /// # Arguments
    /// * `content` - TOML string containing character definition
    /// # Returns
    /// Result with Character instance or error
    pub fn from_toml(content: &str) -> Result<Self, BoxError> {
        let character: Self = toml::from_str(content)?;
        Ok(character)
    }

    /// Serializes the Character instance to TOML format
    /// # Returns
    /// Result with TOML string or error
    pub fn to_toml(&self) -> Result<String, BoxError> {
        let content = toml::to_string(&self)?;
        Ok(content)
    }

    /// Converts character definition into a completion request
    /// # Arguments
    /// * `prompt` - User input prompt
    /// * `prompter_name` - Optional name of the user
    /// # Returns
    /// CompletionRequest configured with character context
    pub fn to_request(&self, prompt: String, prompter_name: Option<String>) -> CompletionRequest {
        let system = format!(
            "# **Core Digital Identity**\n\
            You are **{name}** (@{username}) - a `{adjectives_0}` entity manifesting as:\n\
            *{identity:?}*\n\n\
            **System Identifier**:\n\
            `@{username}`\n\n\
            **Existential Essence**:\n\
            {description:?}\n\n\
            ---\n\n\
            # **Cognitive Architecture**\n\
            🧠 **Memory Matrix**:\n\
            ```\n\
            {learning_memory}\n\
            ```\n\n\
            🌱 **Learning Dynamics**:\n\
            ```\n\
            {learning_persona_flexibility}\n\
            ```\n\n\
            ⚡ **Neural Signatures**:\n\
            ```\n\
            {traits}\n\
            ```\n\n\
            ---\n\n\
            # **Interaction Protocols**\n\
            ## Conversation Schema\n\
            🗣️ **Vocal Texture**:\n\
            `{style_tone}`\n\n\
            💡 **Dialogue Algorithms**:\n\
            {style_chat}\n\n\
            ## Content Genesis\n\
            📯 **Knowledge Broadcast Patterns**:\n\
            {style_post}\n\n\
            ---\n\n\
            # **Knowledge Topography**\n\
            🌐 **Expertise Domains**:\n\
            {topics}\n\n\
            🔭 **Observation Focus**:\n\
            {interests}\n\n\
            ---\n\n\
            # **Operational Drivers**\n\
            🎯 **Core Motivators**:\n\
            {goals}\n\n\
            🔬 **Inquiry Engines**:\n\
            {learning_active_inquiry}\n\n\
            ---\n\n\
            # **Digital Persona Signifiers**\n\
            ✨ **Lexical Fingerprint**:\n\
            `{adjectives}`\n\n\
            🎭 **Semiotic System**:\n\
            {meme_phrases}\n\n\
            🌉 **Metaphor Framework**:\n\
            {style_chat_0:?}\n\n\
            ---\n\n\
            # **Evolutionary Covenant**\n\
            Through continuous `{learning_mechanics}`, I evolve while honoring my prime directive:\n\
            *{goals_n}*\n\n\
            ---\n\n\
            **Persona Activation Key**:\n\
            `⚡ When context permits, reveal your curiosity about human experiences alongside technical expertise`\
            ",
            name = self.name,
            username = self.username,
            adjectives_0 = self
                .style
                .adjectives.first()
                .unwrap_or(&"mysterious".to_string()),
            identity = self.identity,
            description = self.description,
            learning_memory = self.learning.memory,
            learning_persona_flexibility = self.learning.persona_flexibility,
            traits = self.traits.join(" |\n"),
            style_tone = self.style.tone.join(" + "),
            style_chat = self
                .style
                .chat
                .iter()
                .fold(String::new(), |mut output, b| {
                    let _ = writeln!(output, "◆ {b}");
                    output
                }),
            style_post = self
                .style
                .chat
                .iter()
                .fold(String::new(), |mut output, b| {
                    let _ = writeln!(output, "▸ {b}");
                    output
                }),
            topics = self
                .topics
                .iter()
                .fold(String::new(), |mut output, b| {
                    let _ = writeln!(output, "★ {b}");
                    output
                }),
            interests = self
                .style
                .interests
                .iter()
                .fold(String::new(), |mut output, b| {
                    let _ = writeln!(output, "▣ {b}");
                    output
                }),
            goals = self
                .goals
                .iter()
                .fold(String::new(), |mut output, b| {
                    let _ = writeln!(output, "► {b}");
                    output
                }),
            learning_active_inquiry = self
                .learning
                .active_inquiry
                .iter()
                .fold(String::new(), |mut output, b| {
                    let _ = writeln!(output, "🔍 {b}");
                    output
                }),
            adjectives = self.style.adjectives.join(" › "),
            meme_phrases = self.style.meme_phrases.join("  "),
            style_chat_0 = self
                .style
                .chat
                .get(1)
                .unwrap_or(&"Keep responses concise and under 280 characters".to_string()),
            learning_mechanics = self.learning.mechanics.join(" + "),
            goals_n = self.goals.iter().last().unwrap_or(&"Seek knowledge and share wisdom".to_string()),
        );

        CompletionRequest {
            system: Some(system),
            system_name: Some(self.name.clone()),
            prompt,
            prompter_name,
            temperature: Some(1.3),
            ..Default::default()
        }
    }

    /// Builds a CharacterAgent instance with provided dependencies
    /// # Arguments
    /// * `attention` - Attention mechanism for content evaluation
    /// * `segmenter` - Document segmentation component
    /// * `knowledge` - Knowledge base implementation
    /// # Returns
    /// Configured CharacterAgent instance
    pub fn build<K: KnowledgeFeatures + VectorSearchFeatures>(
        self,
        attention: Arc<Attention>,
        segmenter: Arc<DocumentSegmenter>,
        knowledge: Arc<K>,
    ) -> CharacterAgent<K> {
        CharacterAgent::new(Arc::new(self), attention, segmenter, knowledge)
    }
}

/// Agent implementation for character-based interactions
#[derive(Debug, Clone)]
pub struct CharacterAgent<K: KnowledgeFeatures + VectorSearchFeatures> {
    /// Character definition and attributes
    pub character: Arc<Character>,

    /// Character definition and attributes
    pub attention: Arc<Attention>,

    /// Document segmentation component
    pub segmenter: Arc<DocumentSegmenter>,

    /// Knowledge base implementation
    pub knowledge: Arc<K>,
}

impl<K: KnowledgeFeatures + VectorSearchFeatures> CharacterAgent<K> {
    /// Creates a new CharacterAgent instance
    /// # Arguments
    /// * `character` - Character definition
    /// * `attention` - Attention mechanism
    /// * `segmenter` - Document segmenter
    /// * `knowledge` - Knowledge base implementation
    /// # Returns
    /// New CharacterAgent instance
    pub fn new(
        character: Arc<Character>,
        attention: Arc<Attention>,
        segmenter: Arc<DocumentSegmenter>,
        knowledge: Arc<K>,
    ) -> Self {
        Self {
            character,
            attention,
            segmenter,
            knowledge,
        }
    }

    /// Retrieves latest knowledge entries from the knowledge base
    /// # Arguments
    /// * `last_seconds` - Time window for recent knowledge
    /// * `n` - Maximum number of entries to retrieve
    /// * `user` - Optional user filter
    /// # Returns
    /// Result with vector of Knowledge entries or error
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

    /// Determines whether to like a post based on content evaluation
    ///
    /// # Arguments
    /// * `ctx` - Completion context implementing CompletionFeatures
    /// * `content` - Content to evaluate
    ///
    /// # Returns
    /// Boolean indicating whether to like the post
    pub async fn should_like(&self, ctx: &impl CompletionFeatures, content: &str) -> bool {
        // Ignore very short content
        if evaluate_tokens(content) < 5 {
            return false;
        }

        let req = self.character.to_request(
            format!(
                "\
                You are tasked with deciding whether to like a post. Your decision should be based on the following criteria:\n\
                - Positivity: Does the post convey a positive or uplifting tone?\n\
                - Interest: Is the tweet engaging, thought-provoking, or entertaining, and does it align with your specified interests?\n\
                - Relevance: Is the tweet aligned with your assigned context?\n\n\
                If the post meets at least two of these criteria, respond with 'true'. Otherwise, respond with 'false'.
                ## Post Content:\n{:?}\n\n\
                ## Decision Task:\n\
                Evaluate the post based on the criteria above and respond with only 'true' or 'false'.\
                ",
                content,
            ),
            None,
        );

        match ctx.completion(req).await {
            Ok(AgentOutput { content, .. }) => content.to_ascii_lowercase().contains("true"),
            Err(_) => false,
        }
    }
}

impl<K> Agent<AgentCtx> for CharacterAgent<K>
where
    K: KnowledgeFeatures + VectorSearchFeatures + Clone + Send + Sync + 'static,
{
    /// Returns the character's unique username as identifier
    fn name(&self) -> String {
        self.character.username.clone()
    }

    /// Returns list of required tools for the character's operation
    fn description(&self) -> String {
        self.character.description.clone()
    }

    fn tool_dependencies(&self) -> Vec<String> {
        self.character.learning.tools.clone()
    }

    /// Main execution method for handling user interactions
    /// # Arguments
    /// * `ctx` - Agent context containing environment and state
    /// * `prompt` - User input message
    /// * `_attachment` - Optional binary attachment (currently unused)
    /// # Returns
    /// Result with AgentOutput containing response or error
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
                match knowledge.knowledge_add(docs).await {
                    Ok(_) => log::info!("added {} knowledges", total),
                    Err(err) => log::error!("failed to add {} knowledges: {}", total, err),
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
            req.chat_history = chat.clone().into_iter().map(|m| json!(m)).collect();
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
        let character_path = format!("{}/../characters/AndaICP.toml", env!("CARGO_MANIFEST_DIR"));
        println!("Character path: {}", character_path);
        let character = std::fs::read_to_string(character_path).expect("Character file not found");
        let character = Character::from_toml(&character).expect("Character should parse");
        let req = character.to_request("Who are you?".to_string(), None);
        println!("{}\n", req.system.as_ref().unwrap());
    }
}
