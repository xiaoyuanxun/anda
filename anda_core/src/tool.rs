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

use ic_cose_types::validate_str;
use serde::{de::DeserializeOwned, Serialize};
use std::{collections::BTreeMap, future::Future, marker::PhantomData, sync::Arc};

use crate::{context::BaseContext, model::FunctionDefinition, BoxError, BoxPinFut};

/// Core trait for implementing tools that can be used by the AI Agent system
///
/// # Type Parameters
/// - `C`: The context type that implements `BaseContext`, must be thread-safe and have a static lifetime
pub trait Tool<C>: Send + Sync
where
    C: BaseContext + Send + Sync,
{
    /// A constant flag indicating whether the agent should continue processing the tool result
    /// with completion model after execution
    const CONTINUE: bool;
    /// The arguments type of the tool.
    type Args: DeserializeOwned + Send;
    /// The output type of the tool.
    type Output: Serialize;

    /// Returns the tool's name
    /// This name should be unique within the engine.
    ///
    /// # Rules
    /// - Must not be empty
    /// - Length must be ≤ 64 characters
    /// - Can only contain: lowercase letters (a-z), digits (0-9), and underscores (_)
    fn name(&self) -> String;

    /// Returns the tool's capabilities description in a short string
    fn description(&self) -> String;

    /// Provides the tool's definition including its parameters schema.
    ///
    /// # Returns
    /// - `FunctionDefinition`: The schema definition of the tool's parameters and metadata
    fn definition(&self) -> FunctionDefinition;

    /// Executes the tool with given context and arguments
    ///
    /// # Arguments
    /// - `ctx`: The execution context implementing `BaseContext`
    /// - `args`: JSON value containing the input arguments for the tool
    ///
    /// # Returns
    /// - A future resolving to a JSON value containing the tool's output
    /// - Returns `BoxError` if execution fails
    fn call(
        &self,
        ctx: C,
        args: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, BoxError>> + Send;

    /// Executes the tool with given context and arguments using raw JSON string
    /// Returns the output as struct.
    fn call_string(
        &self,
        ctx: C,
        args: String,
    ) -> impl Future<Output = Result<Self::Output, BoxError>> + Send {
        async move {
            let args: Self::Args = serde_json::from_str(&args)
                .map_err(|err| format!("tool {}, invalid args: {}", self.name(), err))?;
            let result = self
                .call(ctx, args)
                .await
                .map_err(|err| format!("tool {}, call failed: {}", self.name(), err))?;
            Ok(result)
        }
    }

    /// Executes the tool with given context and arguments using raw JSON string
    /// Returns the output as a string in JSON format.
    fn call_raw(
        &self,
        ctx: C,
        args: String,
    ) -> impl Future<Output = Result<(String, bool), BoxError>> + Send {
        async move {
            let result = self.call_string(ctx, args).await?;
            Ok((serde_json::to_string(&result)?, Self::CONTINUE))
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

    /// Executes the tool with given context and arguments using dynamic dispatch
    fn call(&self, ctx: C, args: String) -> BoxPinFut<Result<(String, bool), BoxError>>;
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

    fn call(&self, ctx: C, args: String) -> BoxPinFut<Result<(String, bool), BoxError>> {
        let tool = self.0.clone();
        Box::pin(async move { tool.call_raw(ctx, args).await })
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

    /// Adds a new tool to the set
    ///
    /// # Arguments
    /// - `tool`: The tool to add, must implement the `Tool` trait
    pub fn add<T>(&mut self, tool: T) -> Result<(), BoxError>
    where
        T: Tool<C> + Send + Sync + 'static,
    {
        let name = tool.name();
        validate_str(&name)?;
        if self.set.contains_key(&name) {
            return Err(format!("tool {} already exists", name).into());
        }

        let tool_dyn = ToolWrapper(Arc::new(tool), PhantomData);
        self.set.insert(name, Box::new(tool_dyn));
        Ok(())
    }

    /// Calls a tool by name with the given context and arguments
    ///
    /// # Arguments
    /// - `name`: The name of the tool to call
    /// - `ctx`: The execution context
    /// - `args`: JSON value containing the input arguments
    ///
    /// # Returns
    /// - A future resolving to the tool's output as a JSON value
    /// - Returns an error if the tool is not found
    pub fn call(
        &self,
        name: &str,
        ctx: C,
        args: String,
    ) -> BoxPinFut<Result<(String, bool), BoxError>> {
        if let Some(tool) = self.set.get(name) {
            tool.call(ctx, args)
        } else {
            Box::pin(futures::future::ready(Err(format!(
                "tool {name} not found"
            )
            .into())))
        }
    }
}
