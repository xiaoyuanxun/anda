use std::time::Duration;

use anda_core::{
    Agent, AgentContext, AgentOutput, BoxError, CacheExpiry, CacheFeatures, CompletionFeatures,
    CompletionRequest, Documents, MessageInput, StateFeatures, VectorSearchFeatures,
};
use ic_cose_types::to_cbor_bytes;
use serde::{Deserialize, Serialize};

use crate::{
    context::AgentCtx,
    plugin::attention::{Attention, AttentionCommand},
    store::MAX_STORE_OBJECT_SIZE,
};

const MAX_CHAT_HISTORY: usize = 42;
const CHAT_HISTORY_TTI: Duration = Duration::from_secs(3600 * 24 * 7);

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Character {
    /// Name of the character, e.g. "Anda"
    pub name: String,

    /// Character’s profession, status, or role, e.g. "Scientist and Prophet"
    pub identity: String,

    /// Character’s backstory, upbringing, or history.
    pub description: String,

    /// Character’s personality traits, e.g., brave, cunning, kind, etc.
    pub traits: Vec<String>,

    /// Character’s motivations, desires, or objectives.
    pub goals: Vec<String>,

    /// Character’s areas of expertise, e.g., "quantum physics", "time travel", etc.
    pub topics: Vec<String>,

    /// Character’s communication style, interests, and meme-related phrases.
    pub style: Style,

    /// Tools that the character uses to complete tasks.
    /// These tools will be checked for availability when registering the agent.
    pub tools: Vec<String>,

    /// Optional tools that the character uses to complete tasks.
    pub optional_tools: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Style {
    /// Tone of speech, e.g., formal, casual, humorous
    pub tone: Vec<String>,
    /// Communication style in chat
    pub chat: Vec<String>,
    /// Communication style in posts
    pub post: Vec<String>,
    /// Common adjectives used by the character
    pub adjectives: Vec<String>,
    /// Key interests of the character
    pub interests: Vec<String>,
    /// Meme-related phrases used by the character
    pub meme_phrases: Vec<String>,
    /// Example messages for reference
    pub example_messages: Vec<String>,
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

    pub fn to_request(&self, prompt: String) -> CompletionRequest {
        let system = format!(
            "Character Definition:\n\
            Your name: {}\n\
            Your identity: {}\n\
            Background: {}\n\
            Personality traits: {}\n\
            Motivations and goals: {}\n\
            Topics of expertise: {}",
            self.name,
            self.identity,
            self.description,
            self.traits.join(", "),
            self.goals.join(", "),
            self.topics.join(", "),
        );

        let style_context = format!(
            "Your personality and communication style:\n\
            - Tone of speech: {}\n\
            Communication style:\n\
            - In chat: {}\n\
            - In posts: {}\n\n\
            Expression elements:\n\
            - Common adjectives: {}\n\n\
            Personal elements:\n\
            - Key interests: {}\n\
            - Meme-related phrases: {}\n\n\
            Example messages for reference:\n\
            {}",
            self.style.tone.join(", "),
            self.style.chat.join(", "),
            self.style.post.join(", "),
            self.style.adjectives.join(", "),
            self.style.interests.join(", "),
            self.style.meme_phrases.join(", "),
            self.style.example_messages.join("\n")
        );

        CompletionRequest {
            system: Some(system),
            prompt,
            ..Default::default()
        }
        .context("style_context".to_string(), style_context)
    }
}

#[derive(Debug, Clone)]
pub struct CharacterAgent {
    character: Character,
    attention: Attention,
}

impl CharacterAgent {
    pub fn new(character: Character, attention: Attention) -> Self {
        Self {
            character,
            attention,
        }
    }
}

impl Agent<AgentCtx> for CharacterAgent {
    fn name(&self) -> String {
        self.character.name.clone()
    }

    fn description(&self) -> String {
        self.character.description.clone()
    }

    fn tool_dependencies(&self) -> Vec<String> {
        self.character.tools.clone()
    }

    async fn run(
        &self,
        ctx: AgentCtx,
        prompt: String,
        _attachment: Option<Vec<u8>>,
    ) -> Result<AgentOutput, BoxError> {
        // read chat history from store
        let mut chat_history = if let Some(user) = ctx.user() {
            let chat: Vec<MessageInput> = ctx
                .cache_get_with(&user, async {
                    Ok((Vec::new(), Some(CacheExpiry::TTI(CHAT_HISTORY_TTI))))
                })
                .await?;
            Some((user, chat))
        } else {
            None
        };

        let recent_messages: Vec<String> = chat_history
            .as_ref()
            .map(|(_, c)| c.iter().map(|m| m.content.clone()).collect())
            .unwrap_or_default();
        match self
            .attention
            .should_reply(&ctx, &prompt, recent_messages)
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

        let knowledges = ctx.top_n(&prompt, 5).await.unwrap_or_default();
        let knowledges: Documents = knowledges.into();

        let tools: Vec<&str> = self
            .character
            .tools
            .iter()
            .chain(self.character.optional_tools.iter())
            .map(|s| s.as_str())
            .collect();
        let tools = ctx.tool_definitions(Some(&tools));

        let mut req = self
            .character
            .to_request(prompt)
            .append_documents(knowledges)
            .append_tools(tools);

        if let Some((user, chat)) = &mut chat_history {
            req.chat_history = chat.clone();
            chat.push(MessageInput {
                role: "user".to_string(),
                content: req.prompt.clone(),
                ..Default::default()
            });

            // tools will be auto called in completion
            let res = ctx.completion(req).await?;
            if res.failed_reason.is_none() && !res.content.is_empty() {
                // TODO: tool_calls?
                chat.push(MessageInput {
                    role: "assistant".to_string(),
                    content: res.content.clone(),
                    ..Default::default()
                });

                if chat.len() > MAX_CHAT_HISTORY {
                    chat.drain(0..(chat.len() - MAX_CHAT_HISTORY));
                }

                // save chat history to cache
                let data = to_cbor_bytes(&chat);
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
                example_messages: vec![
                    "Hello, I am Anda".to_string(),
                    "The future is uncertain".to_string(),
                ],
            },
            tools: vec!["submit_character".to_string()],
            ..Default::default()
        };
        let req = character.to_request("Who are you?".to_string());
        println!("{}\n", req.system.as_ref().unwrap());
        println!("{}\n", req.prompt_with_context());
        println!("{:?}", req.tools);
    }
}
