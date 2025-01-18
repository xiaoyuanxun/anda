use anda_core::{
    evaluate_tokens, Agent, AgentOutput, BoxError, CompletionFeatures, Tool, ToolCall,
};

use super::extractor::{Deserialize, Extractor, JsonSchema, Serialize, SubmitTool};
use crate::context::AgentCtx;

#[derive(Debug, Clone, Default, Deserialize, Serialize, JsonSchema)]
pub struct SegmentOutput {
    pub segments: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DocumentSegmenter {
    extractor: Extractor<SegmentOutput>,
    tool_name: String,
    segment_tokens: usize,
    max_tokens: usize,
}

impl Default for DocumentSegmenter {
    fn default() -> Self {
        Self::new(500, 8000)
    }
}

impl DocumentSegmenter {
    pub fn new(segment_tokens: usize, max_tokens: usize) -> Self {
        let tool = SubmitTool::<SegmentOutput>::new();
        let tool_name = tool.name();
        let max_tokens_guard = max_tokens - 100;
        let system = format!("\
            You are an expert in summarizing and segmenting long documents. Your task is to take a lengthy knowledge document and break it into multiple concise segments. Each segment should meet the following requirements:\n\n\
            1. Token Limit per Segment: Each segment must not exceed {segment_tokens} tokens.\n\
            2. Total Token Limit: The combined tokens of all segments must not exceed {max_tokens_guard} tokens.\n\
            3. Semantic Integrity: Each segment should be semantically complete, meaning it should convey a clear idea or topic without being cut off mid-thought.\n\
            4. Key Information: Ensure that all critical information from the original document is preserved in the segments.\n\
            5. Clarity: Each segment should be easy to understand and free from unnecessary repetition.\n\n\
            Output Format:\n\
            Use the `{tool_name}` tool to return the segments as a JSON array of strings. Each string represents a segment, and the total tokens of all segments must not exceed {max_tokens_guard}.\
        ");

        let extractor = Extractor::new_with_tool(tool, Some(max_tokens), Some(system));
        Self {
            extractor,
            tool_name,
            segment_tokens,
            max_tokens,
        }
    }

    pub async fn segment(
        &self,
        ctx: &impl CompletionFeatures,
        content: &str,
    ) -> Result<(SegmentOutput, AgentOutput), BoxError> {
        if evaluate_tokens(content) <= self.segment_tokens {
            let res = SegmentOutput {
                segments: vec![content.to_string()],
            };
            let res_str = serde_json::to_string(&res)?;
            return Ok((
                res,
                AgentOutput {
                    content: "".to_string(),
                    failed_reason: None,
                    tool_calls: Some(vec![ToolCall {
                        id: self.tool_name.clone(),
                        name: self.tool_name.clone(),
                        args: res_str.clone(),
                        result: Some(res_str),
                    }]),
                },
            ));
        }

        let tool_name = &self.tool_name;
        let segment_tokens = self.segment_tokens;
        let max_tokens_guard = self.max_tokens - 100;
        let prompt = format!("\
            Document Content:\n\
            {content}\n\n\
            Token Limit per Segment: {segment_tokens}\n\
            Total Token Limit: {max_tokens_guard}\n\
            Break the document into segments, ensuring each segment does not exceed {segment_tokens} tokens and the total tokens do not exceed {max_tokens_guard}.  Use the `{tool_name}` tool to return the results.\
            ",
        );

        self.extractor.extract(ctx, prompt).await
    }
}

impl Agent<AgentCtx> for DocumentSegmenter {
    fn name(&self) -> String {
        "document_segmenter".to_string()
    }

    fn description(&self) -> String {
        "Take a lengthy knowledge document and break it into multiple concise segments using LLMs."
            .to_string()
    }

    async fn run(
        &self,
        ctx: AgentCtx,
        prompt: String,
        _attachment: Option<Vec<u8>>,
    ) -> Result<AgentOutput, BoxError> {
        let (_, res) = self.segment(&ctx, &prompt).await?;
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_definition() {
        let tool = SubmitTool::<SegmentOutput>::new();
        let definition = tool.definition();
        println!("{}", tool.name());
        println!("{}", serde_json::to_string_pretty(&definition).unwrap());
    }
}
