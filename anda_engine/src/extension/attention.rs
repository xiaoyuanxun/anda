use anda_core::{evaluate_tokens, AgentOutput, CompletionFeatures, CompletionRequest, Message};

use crate::context::AgentCtx;

static HIGH_REWARD_COMMAND: &str = "HIGH_REWARD";
static MEDIUM_REWARD_COMMAND: &str = "MEDIUM_REWARD";
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

#[derive(Debug, Default, PartialEq, PartialOrd)]
pub enum AttentionCommand {
    Stop,
    #[default]
    Ignore,
    Respond,
}

#[derive(Debug, Default, PartialEq, PartialOrd)]
pub enum ContentQuality {
    #[default]
    Ignore,
    Good,
    Exceptional,
}

#[derive(Debug, Clone)]
pub struct Attention {
    phrases: Vec<String>,
    pub min_prompt_tokens: usize,
    pub min_content_tokens: usize,
}

impl Default for Attention {
    fn default() -> Self {
        Self {
            phrases: STOP_PHRASES.iter().map(|s| s.to_string()).collect(),
            min_prompt_tokens: 4,
            min_content_tokens: 60,
        }
    }
}

impl Attention {
    pub fn new(phrases: Vec<String>, min_prompt_tokens: usize, min_content_tokens: usize) -> Self {
        Self {
            phrases,
            min_prompt_tokens,
            min_content_tokens,
        }
    }

    pub async fn evaluate_content(&self, ctx: &AgentCtx, content: &str) -> ContentQuality {
        // Ignore very short content
        if evaluate_tokens(content) < self.min_content_tokens {
            return ContentQuality::Ignore;
        }

        let req = CompletionRequest {
            system: Some(format!("\
                You are an expert evaluator for article content quality, specializing in assessing knowledge value. Your task is to analyze the provided article, classify its quality into three levels, and determine the appropriate storage and reward action.\n\n\
                Evaluation criteria:\n\
                1. Knowledge Depth: Does the article provide detailed, well-researched, or expert-level insights?\n\
                2. Originality: Is the content unique, creative, or innovative?\n\
                3. Relevance: Is the content actionable, practical, or useful for the intended audience?\n\n\
                Classification Levels:\n\
                - {HIGH_REWARD_COMMAND}: The article has exceptional knowledge value, with deep insights, originality, and significant relevance.\n\
                - {MEDIUM_REWARD_COMMAND}: The article has good knowledge value, meeting most criteria but with some areas for improvement.\n\
                - {IGNORE_COMMAND}: The article does not meet the criteria for high or medium knowledge value and requires no action.")),
            prompt: format!("\
                Article Content:\n{}\n\n\
                Evaluation Task:\n\
                1. Analyze the article based on Knowledge Depth, Originality, and Relevance.\n\
                2. Classify the article into one of the three levels:\n\
                - {HIGH_REWARD_COMMAND}: Exceptional quality.\n\
                - {MEDIUM_REWARD_COMMAND}: Good quality.\n\
                - {IGNORE_COMMAND}: Low quality or no significant knowledge value.\n\
                3. Provide a brief explanation for your classification, citing specific strengths or weaknesses of the article.\
                ",
                content
            ),
            ..Default::default()
        };

        match ctx.completion(req).await {
            Ok(AgentOutput { content, .. }) => {
                if content.contains(HIGH_REWARD_COMMAND) {
                    ContentQuality::Exceptional
                } else if content.contains(MEDIUM_REWARD_COMMAND) {
                    ContentQuality::Good
                } else {
                    ContentQuality::Ignore
                }
            }
            Err(_) => ContentQuality::Ignore,
        }
    }

    pub async fn should_reply(
        &self,
        ctx: &AgentCtx,
        my_name: &str,
        topics: &[String],
        recent_messages: &[Message],
        message: &Message,
    ) -> AttentionCommand {
        let content = message.content.to_string().to_lowercase();
        if self.phrases.iter().any(|phrase| content.contains(phrase)) {
            return AttentionCommand::Stop;
        }

        // Ignore very short messages
        if evaluate_tokens(&content) < self.min_prompt_tokens {
            return AttentionCommand::Ignore;
        }

        let recent_messages: Vec<String> = recent_messages
            .iter()
            .map(|msg| {
                format!(
                    "{}: {:?}",
                    msg.name.as_ref().unwrap_or(&msg.role),
                    msg.content
                )
            })
            .collect();
        let user_message = format!(
            "{}: {:?}",
            message.name.as_ref().unwrap_or(&message.role),
            message.content
        );

        let req = CompletionRequest {
            system: Some(format!("\
                You are {my_name}.\n\
                You are part of a multi-user discussion environment. Your primary task is to evaluate the relevance of each message to your assigned conversation topics and decide whether to respond. Always prioritize messages that directly mention you or are closely related to the conversation topic.\n\n\
                Response options:\n\
                - {RESPOND_COMMAND}: The message is directly addressed to you or is highly relevant to the conversation topic.\n\
                - {IGNORE_COMMAND}: The message is not addressed to you and is unrelated to the conversation topic.\n\
                - {STOP_COMMAND}: The user has explicitly requested you to stop or the conversation has ended.")),
            prompt: format!("\
                Assigned Conversation Topics: {}\n\
                Recent Messages:\n{}\n\
                Latest message:\n{}\n\n\
                Decision Task:\n\
                Evaluate whether the latest message requires your response. Choose one response option from the list above and provide a brief explanation for your choice.\
                ",
                topics.join(", "), recent_messages.join("\n"), user_message
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

    pub async fn should_like(&self, ctx: &AgentCtx, content: &str) -> bool {
        // Ignore very short content
        if evaluate_tokens(content) < self.min_prompt_tokens {
            return false;
        }

        let req = CompletionRequest {
            system: Some("\
            You are tasked with deciding whether to like a post. Your decision should be based on the following criteria:\n\
            - Positivity: Does the post convey a positive or uplifting tone?\n\
            - Interest: Is the post engaging, thought-provoking, or entertaining?\n\
            - Relevance: Is the post aligned with your assigned interests or context?\n\n\
            If the post meets at least one of these criteria, respond with 'true'. Otherwise, respond with 'false'.
            ".to_string()),
            prompt: format!("\
                Post Content: {}\n\n\
                Decision Task:\n\
                Evaluate the post based on the criteria above and respond with only 'true' or 'false'.\
                ",
                content
            ),
            ..Default::default()
        };

        match ctx.completion(req).await {
            Ok(AgentOutput { content, .. }) => content.to_ascii_lowercase().contains("true"),
            Err(_) => false,
        }
    }

    pub async fn should_retweet(&self, ctx: &AgentCtx, content: &str) -> bool {
        // Ignore very short content
        if evaluate_tokens(content) < self.min_prompt_tokens {
            return false;
        }

        let req = CompletionRequest {
            system: Some("\
            You are tasked with deciding whether to retweet a post. Your decision should be based on the following criteria:\n\
            - High Value: Does the post provide significant knowledge, insights, or meaningful contributions?\n\
            - Interest: Is the post highly engaging, thought-provoking, or likely to resonate with a broader audience?\n\
            - Alignment: Does the post reflect your values, beliefs, or the message you want to amplify?\n\n\
            Retweet only if the post strongly satisfies at least one of these criteria.\
            ".to_string()),
            prompt: format!("\
                Post Content: {}\n\n\
                Decision Task:\n\
                Evaluate the post based on the criteria above and respond with only 'true' or 'false'.\
                ",
                content
            ),
            ..Default::default()
        };

        match ctx.completion(req).await {
            Ok(AgentOutput { content, .. }) => content.to_ascii_lowercase().contains("true"),
            Err(_) => false,
        }
    }

    pub async fn should_quote(&self, ctx: &AgentCtx, content: &str) -> bool {
        // Ignore very short content
        if evaluate_tokens(content) < self.min_prompt_tokens {
            return false;
        }

        let req = CompletionRequest {
            system: Some("\
            You are tasked with deciding whether to quote a post. Base your decision on the following criteria:\n\
            - Deserves Commentary: Does the post raise a point, idea, or question that merits your unique perspective or opinion?\n\
            - Needs Additional Context: Could the post's content benefit from clarification, expansion, or supplementary information to enhance its value?\n\
            - Warrants Thoughtful Response: Does the post address a topic or issue that requires a nuanced, constructive, or meaningful reply?\n\n\
            Quote the post only if it satisfies at least one of these criteria significantly.\
            ".to_string()),
            prompt: format!("\
                Post Content: {}\n\n\
                Decision Task:\n\
                Evaluate the post based on the criteria above and respond with only 'true' or 'false'.\
                ",
                content
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
    use super::*;

    #[test]
    fn test_content_quality() {
        assert!(ContentQuality::Good > ContentQuality::Ignore);
    }
}
