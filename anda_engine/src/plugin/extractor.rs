//! This module provides high-level abstractions for extracting structured data from text using LLMs.
//!
//! Note: The target structure must implement the `serde::Deserialize`, `serde::Serialize`,
//! and `schemars::JsonSchema` traits. Those can be easily derived using the `derive` macro.

use anda_core::{
    Agent, AgentOutput, BoxError, CompletionFeatures, CompletionRequest, FunctionDefinition, Tool,
    Value,
};
use serde_json::json;
use std::marker::PhantomData;

pub use schemars::{schema_for, JsonSchema};
pub use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::context::{AgentCtx, BaseCtx};

pub struct SubmitTool<T: JsonSchema + DeserializeOwned + Send + Sync> {
    _t: PhantomData<T>,
}

impl<T> SubmitTool<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + Send + Sync,
{
    pub fn new() -> SubmitTool<T> {
        SubmitTool { _t: PhantomData }
    }
}

impl<T> Tool<BaseCtx> for SubmitTool<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + Send + Sync,
{
    const NAME: &'static str = "submit";

    type Args = T;
    type Output = T;

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: Self::NAME.to_string(),
            description: "Submit the structured data you extracted from the provided text."
                .to_string(),
            parameters: json!(schema_for!(T)),
            strict: Some(true),
        }
    }

    async fn call(&self, _ctx: BaseCtx, data: Self::Args) -> Result<Self::Output, BoxError> {
        Ok(data)
    }
}

/// Extractor for structured data from text
pub struct Extractor<T: JsonSchema + DeserializeOwned + Serialize + Send + Sync>(pub SubmitTool<T>);

impl<T: JsonSchema + DeserializeOwned + Serialize + Send + Sync> Extractor<T> {
    pub fn new() -> Self {
        Self(SubmitTool { _t: PhantomData })
    }

    pub async fn extract(&self, ctx: AgentCtx, prompt: String) -> Result<T, BoxError> {
        let req = CompletionRequest {
            preamble: Some(format!("\
                You are an AI assistant whose purpose is to\
                extract structured data from the provided text.\n\
                You will have access to a `submit` function that defines the structure of the data to extract from the provided text.\n\
                Use the `submit` function to submit the structured data.\n\
                Be sure to fill out every field and ALWAYS CALL THE `submit` function, event with default values!!!.")),
            prompt,
            tools: vec![self.0.definition()],
            ..Default::default()
        };

        let mut res = ctx.completion(req).await?;
        if let Some(tool_calls) = res.tool_calls.as_mut() {
            for tool in tool_calls.iter_mut() {
                if let Ok(val) = serde_json::from_str::<T>(&tool.args) {
                    return Ok(val);
                }
            }
        }

        Err("extract failed".into())
    }
}

impl<T> Agent<AgentCtx> for Extractor<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + Send + Sync,
{
    const NAME: &'static str = "extractor";

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: Self::NAME.to_string(),
            description: "Extract structured data from text using LLMs.".to_string(),
            parameters: Value::Null,
            strict: None,
        }
    }

    async fn run(
        &self,
        ctx: AgentCtx,
        prompt: String,
        _attachment: Option<Vec<u8>>,
    ) -> Result<AgentOutput, BoxError> {
        let req = CompletionRequest {
            preamble: Some(format!("\
                You are an AI assistant whose purpose is to\
                extract structured data from the provided text.\n\
                You will have access to a `submit` function that defines the structure of the data to extract from the provided text.\n\
                Use the `submit` function to submit the structured data.\n\
                Be sure to fill out every field and ALWAYS CALL THE `submit` function, event with default values!!!.")),
            prompt,
            tools: vec![self.0.definition()],
            ..Default::default()
        };

        let mut res = ctx.completion(req).await?;
        if let Some(tool_calls) = res.tool_calls.as_mut() {
            for tool in tool_calls.iter_mut() {
                if serde_json::from_str::<T>(&tool.args).is_ok() {
                    tool.result = Some(tool.args.clone());
                    return Ok(res);
                }
            }
        }

        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
    struct TestStruct {
        name: String,
        age: Option<u8>,
    }

    #[test]
    fn test_submit_tool() {
        let tool = SubmitTool::<TestStruct>::new();
        let definition = tool.definition();
        assert_eq!(definition.name, "submit");
        let s = serde_json::to_string(&definition).unwrap();
        println!("{}", s);
        // {"name":"submit","description":"Submit the structured data you extracted from the provided text.","parameters":{"$schema":"http://json-schema.org/draft-07/schema#","properties":{"age":{"format":"uint8","minimum":0.0,"type":["integer","null"]},"name":{"type":"string"}},"required":["name"],"title":"TestStruct","type":"object"},"strict":true}
    }
}
