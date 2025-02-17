//! Document Segmentation Module
//!
//! This module provides intelligent document segmentation capabilities using LLMs.
//! It breaks down long documents into smaller, semantically complete chunks while
//! respecting token limits and preserving the original meaning.
//!
//! # Key Features
//! - Semantic-aware segmentation preserving document integrity
//! - Configurable token limits for both individual segments and total output
//! - LLM-powered extraction for intelligent boundary detection
//! - Integration with the Extractor framework for structured output
//!
//! # Main Components
//! - [`DocumentSegmenter`]: The core segmentation tool implementing the Agent trait
//! - [`SegmentOutput`]: The structured output format containing segmented text
//!
//! # Usage
//! The module is typically used through the DocumentSegmenter struct which provides:
//! - Initialization with custom token limits
//! - Direct segmentation via the `segment()` method
//! - Agent interface implementation for integration with the broader system
//!
//! # Example
//! ```rust,ignore
//! let segmenter = DocumentSegmenter::new(500, 8000);
//! let segments = segmenter.segment(&ctx, long_document).await?;
//! ```

use anda_core::{
    evaluate_tokens, Agent, AgentOutput, BoxError, CompletionFeatures, Tool, ToolCall,
};
use schemars::JsonSchema;

use super::extractor::{Deserialize, Extractor, Serialize, SubmitTool};
use crate::context::AgentCtx;

/// Represents the output of document segmentation containing multiple text segments
#[derive(Debug, Clone, Default, Deserialize, Serialize, JsonSchema)]
pub struct SegmentOutput {
    pub segments: Vec<String>,
}

/// A document segmentation tool that breaks long documents into smaller, semantically complete chunks
/// using LLMs while respecting token limits.
///
/// Implementation Details:
/// Built on top of the [`Extractor`] for structured output generation.
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
    const NAME: &'static str = "document_segmenter";
    /// Creates a new DocumentSegmenter with specified token limits
    ///
    /// # Arguments
    /// * `segment_tokens` - Maximum tokens allowed per individual segment
    /// * `max_tokens` - Maximum total tokens allowed for all segments combined
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

    /// Segments a document into smaller chunks while preserving semantic meaning
    ///
    /// # Arguments
    /// * `ctx` - Context implementing CompletionFeatures
    /// * `content` - The document content to be segmented
    ///
    /// # Returns
    /// Result containing the segmented output and agent output
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
                    full_history: None,
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
    /// Returns the name "document_segmenter" of the segmenter tool
    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    /// Returns the description of the segmenter tool
    fn description(&self) -> String {
        "Take a lengthy knowledge document and break it into multiple concise segments using LLMs."
            .to_string()
    }

    /// Executes the document segmentation process
    ///
    /// # Arguments
    /// * `ctx` - Agent context
    /// * `prompt` - Input document content
    /// * `_attachment` - Optional binary attachment (not used in this implementation)
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
