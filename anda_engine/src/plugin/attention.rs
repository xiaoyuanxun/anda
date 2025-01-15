use anda_core::{AgentOutput, CompletionFeatures, CompletionRequest};

use crate::context::AgentCtx;

static RESPOND_COMMAND: &str = "RESPOND";
static IGNORE_COMMAND: &str = "IGNORE";
static STOP_COMMAND: &str = "STOP";
static STOP_PHRASES: [&str; 13] = [
    "shut up",
    "dont talk",
    "silence",
    "stop talking",
    "be quiet",
    "hush",
    "wtf",
    "stfu",
    "stupid bot",
    "dumb bot",
    "stop responding",
    "can you not",
    "can you stop",
];

#[derive(Debug, PartialEq)]
pub enum AttentionCommand {
    Respond,
    Ignore,
    Stop,
}

#[derive(Debug, Clone)]
pub struct Attention {
    phrases: Vec<String>,
    min_prompt_len: usize,
}

impl Default for Attention {
    fn default() -> Self {
        Self {
            phrases: STOP_PHRASES.iter().map(|s| s.to_string()).collect(),
            min_prompt_len: 10,
        }
    }
}

impl Attention {
    pub fn new(phrases: Vec<String>, min_prompt_len: usize) -> Self {
        Self {
            phrases,
            min_prompt_len,
        }
    }

    pub async fn should_reply(
        &self,
        ctx: &AgentCtx,
        message: &str,
        recent_messages: Vec<String>,
    ) -> AttentionCommand {
        if self.phrases.iter().any(|phrase| message.contains(phrase)) {
            return AttentionCommand::Stop;
        }

        // Ignore very short messages
        if message.len() < self.min_prompt_len {
            return AttentionCommand::Ignore;
        }

        let msgs = recent_messages
            .into_iter()
            .map(|msg| format!("- {}", msg))
            .collect::<Vec<_>>()
            .join("\n");

        let req = CompletionRequest {
            system: Some(format!("\
                You are in a room with other users. You should only respond when addressed or when the conversation is relevant to you.\n\n\
                Response options:\n\
                {RESPOND_COMMAND} - Message is directed at you or conversation is relevant\n\
                {IGNORE_COMMAND} - Message is not interesting or not directed at you\n\
                {STOP_COMMAND} - User wants you to stop or conversation has concluded")),
            prompt: format!(
                "Recent messages:\n{}\n\nLatest message: {}\n\n\
                Choose one response option:",
                msgs, message
            ),
            ..Default::default()
        };

        match ctx.completion(req).await {
            Ok(AgentOutput { content, .. }) => {
                if content.contains(RESPOND_COMMAND) {
                    AttentionCommand::Respond
                } else if content.contains(STOP_COMMAND) {
                    AttentionCommand::Stop
                } else {
                    AttentionCommand::Ignore
                }
            }
            Err(_) => AttentionCommand::Ignore,
        }
    }

    pub async fn should_like(&self, ctx: &AgentCtx, tweet_content: &str) -> bool {
        let req = CompletionRequest {
            system: Some("You are deciding whether to like a tweet. Consider if the content is positive, interesting, or relevant.".to_string()),
            prompt: format!(
                "Tweet: {}\n\n\
                Respond with only 'true' or 'false':",
                tweet_content
            ),
            ..Default::default()
        };

        match ctx.completion(req).await {
            Ok(AgentOutput { content, .. }) => content.to_ascii_lowercase().contains("true"),
            Err(_) => false,
        }
    }

    pub async fn should_retweet(&self, ctx: &AgentCtx, tweet_content: &str) -> bool {
        let req = CompletionRequest {
            system: Some("You are deciding whether to retweet. Only retweet if the content is highly valuable, interesting, or aligns with your values.".to_string()),
            prompt: format!(
                "Tweet: {}\n\n\
                Respond with only 'true' or 'false':",
                tweet_content
            ),
            ..Default::default()
        };

        match ctx.completion(req).await {
            Ok(AgentOutput { content, .. }) => content.to_ascii_lowercase().contains("true"),
            Err(_) => false,
        }
    }

    pub async fn should_quote(&self, ctx: &AgentCtx, tweet_content: &str) -> bool {
        let req = CompletionRequest {
            system: Some("You are deciding whether to quote tweet. Quote tweet if the content deserves commentary, \
            could benefit from additional context, or warrants a thoughtful response.".to_string()),
            prompt: format!(
                "Tweet: {}\n\n\
                Respond with only 'true' or 'false':",
                tweet_content
            ),
            ..Default::default()
        };

        match ctx.completion(req).await {
            Ok(AgentOutput { content, .. }) => content.to_ascii_lowercase().contains("true"),
            Err(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_agent() {
        // let agent = Attention::default();
    }
}
