//! Module providing core tooling functionality for AI Agents
//!
//! This module defines the core traits and structures for creating and managing tools
//! that can be used by AI Agents. It provides:
//! - The [`Tool`] trait for defining custom tools with typed arguments and outputs
//! - Dynamic dispatch capabilities through [`ToolDyn`] trait
//! - A [`ToolSet`] collection for managing multiple tools
//!
//! # Key Features
//! - Type-safe tool definitions with schema validation
//! - Asynchronous execution model
//! - Dynamic dispatch support for runtime tool selection
//! - Tool registration and management system
//!
//! # Usage
//!
//! ## Reference Implementations
//! 1. [`GoogleSearchTool`](https://github.com/ldclabs/anda/blob/main/anda_engine/src/extension/google.rs) -
//!    A tool for performing web searches and retrieve results
//! 2. [`SubmitTool`](https://github.com/ldclabs/anda/blob/main/anda_engine/src/extension/extractor.rs) -
//!    A tool for extracting structured data using LLMs
//! 3. [`TransferTool`](https://github.com/ldclabs/anda/blob/main/tools/anda_icp/src/ledger/transfer.rs) -
//!    A tool for handling ICP blockchain transfers
//! 4. [`BalanceOfTool`](https://github.com/ldclabs/anda/blob/main/tools/anda_icp/src/ledger/balance.rs) -
//!    A tool for querying ICP blockchain balances
//!
//! These reference implementations share a common feature: they automatically generate the JSON Schema
//! required for LLMs Function Calling.

use serde::{Serialize, de::DeserializeOwned};
use std::{collections::BTreeMap, future::Future, marker::PhantomData, sync::Arc};

use crate::{
    BoxError, BoxPinFut, Function, Resource, ToolOutput, Value, context::BaseContext,
    model::FunctionDefinition, validate_function_name,
};

/// Core trait for implementing tools that can be used by the AI Agent system
///
/// # Type Parameters
/// - `C`: The context type that implements `BaseContext`, must be thread-safe and have a static lifetime
pub trait Tool<C>: Send + Sync
where
    C: BaseContext + Send + Sync,
{
    /// The arguments type of the tool.
    type Args: DeserializeOwned + Send;
    /// The output type of the tool.
    type Output: Serialize;

    /// Returns the tool's name
    ///
    /// # Rules
    /// - Must not be empty
    /// - Must not exceed 64 characters
    /// - Must start with a lowercase letter
    /// - Can only contain: lowercase letters (a-z), digits (0-9), and underscores (_)
    /// - Unique within the engine
    fn name(&self) -> String;

    /// Returns the tool's capabilities description in a short string
    fn description(&self) -> String;

    /// Provides the tool's definition including its parameters schema.
    ///
    /// # Returns
    /// - `FunctionDefinition`: The schema definition of the tool's parameters and metadata
    fn definition(&self) -> FunctionDefinition;

    /// It is used to select resources based on the provided tags.
    /// If the tool requires specific resources, it can filter them based on the tags.
    /// By default, it returns an empty list.
    ///
    /// # Arguments
    /// - `tags`: List of tags to filter resources
    ///
    /// # Returns
    /// - A list of resource tags from the tags provided that supported by the tool
    fn supported_resource_tags(&self) -> Vec<String> {
        Vec::new()
    }

    /// Initializes the tool with the given context.
    /// It will be called once when building the engine.
    fn init(&self, _ctx: C) -> impl Future<Output = Result<(), BoxError>> + Send {
        futures::future::ready(Ok(()))
    }

    /// Executes the tool with given context and arguments
    ///
    /// # Arguments
    /// - `ctx`: The execution context implementing `BaseContext`
    /// - `args`: JSON value containing the input arguments for the tool
    /// - `resources`: Optional additional resources, If resources don’t meet the tool’s expectations, return an error.
    ///
    /// # Returns
    /// - A future resolving to a JSON value containing the tool's output
    /// - Returns `BoxError` if execution fails
    fn call(
        &self,
        ctx: C,
        args: Self::Args,
        resources: Option<Vec<Resource>>,
    ) -> impl Future<Output = Result<ToolOutput<Self::Output>, BoxError>> + Send;

    /// Executes the tool with given context and arguments using raw JSON string
    /// Returns the output as a string in JSON format.
    fn call_raw(
        &self,
        ctx: C,
        args: String,
        resources: Option<Vec<Resource>>,
    ) -> impl Future<Output = Result<ToolOutput<Value>, BoxError>> + Send {
        async move {
            let args: Self::Args = serde_json::from_str(&args)
                .map_err(|err| format!("tool {}, invalid args: {}", self.name(), err))?;
            let result = self
                .call(ctx, args, resources)
                .await
                .map_err(|err| format!("tool {}, call failed: {}", self.name(), err))?;
            let output = serde_json::to_value(&result.output)?;
            Ok(ToolOutput {
                output,
                resources: result.resources,
                usage: result.usage,
            })
        }
    }
}

/// Dynamic dispatch version of the Tool trait
///
/// This trait allows for runtime polymorphism of tools, enabling different tool implementations
/// to be stored and called through a common interface.
pub trait ToolDyn<C>: Send + Sync
where
    C: BaseContext + Send + Sync,
{
    /// Returns the tool's name as a String
    fn name(&self) -> String;

    /// Provides the tool's definition including its parameters schema
    fn definition(&self) -> FunctionDefinition;

    fn supported_resource_tags(&self) -> Vec<String>;

    /// Initializes the tool with the given context
    fn init(&self, ctx: C) -> BoxPinFut<Result<(), BoxError>>;

    /// Executes the tool with given context and arguments using dynamic dispatch
    fn call(
        &self,
        ctx: C,
        args: String,
        resources: Option<Vec<Resource>>,
    ) -> BoxPinFut<Result<ToolOutput<Value>, BoxError>>;
}

/// Wrapper to convert static Tool implementation to dynamic dispatch
struct ToolWrapper<T, C>(Arc<T>, PhantomData<C>)
where
    T: Tool<C> + 'static,
    C: BaseContext + Send + Sync + 'static;

impl<T, C> ToolDyn<C> for ToolWrapper<T, C>
where
    T: Tool<C> + 'static,
    C: BaseContext + Send + Sync + 'static,
{
    fn name(&self) -> String {
        self.0.name()
    }

    fn definition(&self) -> FunctionDefinition {
        self.0.definition()
    }

    fn supported_resource_tags(&self) -> Vec<String> {
        self.0.supported_resource_tags()
    }

    fn init(&self, ctx: C) -> BoxPinFut<Result<(), BoxError>> {
        let tool = self.0.clone();
        Box::pin(async move { tool.init(ctx).await })
    }

    fn call(
        &self,
        ctx: C,
        args: String,
        resources: Option<Vec<Resource>>,
    ) -> BoxPinFut<Result<ToolOutput<Value>, BoxError>> {
        let tool = self.0.clone();
        Box::pin(async move { tool.call_raw(ctx, args, resources).await })
    }
}

/// Collection of tools that can be used by the AI Agent
///
/// # Type Parameters
/// - `C`: The context type that implements `BaseContext`, must have a static lifetime
#[derive(Default)]
pub struct ToolSet<C: BaseContext> {
    pub set: BTreeMap<String, Box<dyn ToolDyn<C>>>,
}

impl<C> ToolSet<C>
where
    C: BaseContext + Send + Sync + 'static,
{
    /// Creates a new empty ToolSet
    pub fn new() -> Self {
        Self {
            set: BTreeMap::new(),
        }
    }

    /// Checks if a tool with the given name exists in the set
    pub fn contains(&self, name: &str) -> bool {
        self.set.contains_key(name)
    }

    /// Gets the definition of a specific tool by name
    ///
    /// # Returns
    /// - `Some(FunctionDefinition)` if the tool exists
    /// - `None` if the tool is not found
    pub fn definition(&self, name: &str) -> Option<FunctionDefinition> {
        self.set.get(name).map(|tool| tool.definition())
    }

    /// Gets definitions for multiple tools, optionally filtered by names
    ///
    /// # Arguments
    /// - `names`: Optional slice of tool names to filter by. If None, returns all definitions
    ///
    /// # Returns
    /// - Vector of `FunctionDefinition` for the requested tools
    pub fn definitions(&self, names: Option<&[&str]>) -> Vec<FunctionDefinition> {
        self.set
            .iter()
            .filter_map(|(name, tool)| match names {
                Some(names) => {
                    if names.contains(&name.as_str()) {
                        Some(tool.definition())
                    } else {
                        None
                    }
                }
                None => Some(tool.definition()),
            })
            .collect()
    }

    pub fn functions(&self, names: Option<&[&str]>) -> Vec<Function> {
        self.set
            .iter()
            .filter_map(|(name, tool)| match names {
                Some(names) => {
                    if names.contains(&name.as_str()) {
                        Some(Function {
                            definition: tool.definition(),
                            supported_resource_tags: tool.supported_resource_tags(),
                        })
                    } else {
                        None
                    }
                }
                None => Some(Function {
                    definition: tool.definition(),
                    supported_resource_tags: tool.supported_resource_tags(),
                }),
            })
            .collect()
    }

    /// Adds a new tool to the set
    ///
    /// # Arguments
    /// - `tool`: The tool to add, must implement the `Tool` trait
    pub fn add<T>(&mut self, tool: T) -> Result<(), BoxError>
    where
        T: Tool<C> + Send + Sync + 'static,
    {
        let name = tool.name();
        validate_function_name(&name)?;
        if self.set.contains_key(&name) {
            return Err(format!("tool {} already exists", name).into());
        }

        let tool_dyn = ToolWrapper(Arc::new(tool), PhantomData);
        self.set.insert(name, Box::new(tool_dyn));
        Ok(())
    }

    /// Retrieves a tool by name
    pub fn get(&self, name: &str) -> Option<&Box<dyn ToolDyn<C>>> {
        self.set.get(name)
    }
}
