//! Module providing core agent functionality for AI systems
//!
//! This module defines the core traits and structures for creating and managing AI agents. It provides:
//! - The [`Agent`] trait for defining custom agents with specific capabilities
//! - Dynamic dispatch capabilities through [`AgentDyn`] trait
//! - An [`AgentSet`] collection for managing multiple agents
//!
//! # Key Features
//! - Type-safe agent definitions with clear interfaces
//! - Asynchronous execution model
//! - Dynamic dispatch support for runtime agent selection
//! - Agent registration and management system
//! - Tool dependency management
//!
//! # Architecture Overview
//! The module follows a dual-trait pattern:
//! 1. [`Agent`] - Static trait for defining concrete agent implementations
//! 2. [`AgentDyn`] - Dynamic trait for runtime polymorphism
//!
//! The [`AgentSet`] acts as a registry and execution manager for agents, providing:
//! - Agent registration and lookup
//! - Bulk definition retrieval
//! - Execution routing
//!
//! # Usage
//!
//! ## Reference Implementations
//! 1. [`Extractor`](https://github.com/ldclabs/anda/blob/main/anda_engine/src/extension/extractor.rs) -
//!    An agent for structured data extraction using LLMs
//! 2. [`DocumentSegmenter`](https://github.com/ldclabs/anda/blob/main/anda_engine/src/extension/segmenter.rs) -
//!    A document segmentation tool using LLMs
//! 3. [`CharacterAgent`](https://github.com/ldclabs/anda/blob/main/anda_engine/src/extension/character.rs) -
//!    A role-playing AI agent, also serving as the core agent for [`anda_bot`](https://github.com/ldclabs/anda/blob/main/agents/anda_bot/README.md)

use serde_json::json;
use std::{collections::BTreeMap, future::Future, marker::PhantomData, sync::Arc};

use crate::{
    context::AgentContext,
    model::{AgentOutput, FunctionDefinition},
    validate_path_part, BoxError, BoxPinFut,
};

/// Core trait defining an AI agent's behavior
///
/// # Type Parameters
/// - `C`: The context type that implements `AgentContext`, must be thread-safe and have a static lifetime
pub trait Agent<C>: Send + Sync
where
    C: AgentContext + Send + Sync,
{
    /// Returns the agent's name as a String
    /// The unique name of the agent. This name should be valid Path string and unique within the engine in lowercase.
    fn name(&self) -> String;

    /// Returns the agent's capabilities description in a short string
    fn description(&self) -> String;

    /// Returns the agent's function definition for API integration
    ///
    /// # Returns
    /// - `FunctionDefinition`: The structured definition of the agent's capabilities
    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: json!({"type":"string"}),
            strict: None,
        }
    }

    fn tool_dependencies(&self) -> Vec<String> {
        Vec::new()
    }

    /// Executes the agent's main logic with given context and inputs
    ///
    /// # Arguments
    /// - `ctx`: The execution context implementing `AgentContext`
    /// - `prompt`: The input prompt or message for the agent
    /// - `attachment`: Optional additional data in JSON format
    ///
    /// # Returns
    /// - A future resolving to `Result<AgentOutput, BoxError>`
    fn run(
        &self,
        ctx: C,
        prompt: String,
        attachment: Option<Vec<u8>>,
    ) -> impl Future<Output = Result<AgentOutput, BoxError>> + Send;
}

/// Dynamic dispatch version of Agent trait for runtime flexibility
///
/// This trait allows for runtime polymorphism of agents, enabling dynamic agent selection
/// and execution without knowing the concrete type at compile time.
pub trait AgentDyn<C>: Send + Sync
where
    C: AgentContext + Send + Sync,
{
    fn name(&self) -> String;

    fn definition(&self) -> FunctionDefinition;

    fn tool_dependencies(&self) -> Vec<String>;

    fn run(
        &self,
        ctx: C,
        prompt: String,
        attachment: Option<Vec<u8>>,
    ) -> BoxPinFut<Result<AgentOutput, BoxError>>;
}

/// Adapter for converting static Agent to dynamic dispatch
struct AgentWrapper<T, C>(Arc<T>, PhantomData<C>)
where
    T: Agent<C> + 'static,
    C: AgentContext + Send + Sync + 'static;

impl<T, C> AgentDyn<C> for AgentWrapper<T, C>
where
    T: Agent<C> + 'static,
    C: AgentContext + Send + Sync + 'static,
{
    fn name(&self) -> String {
        self.0.name()
    }

    fn definition(&self) -> FunctionDefinition {
        self.0.definition()
    }

    fn tool_dependencies(&self) -> Vec<String> {
        self.0.tool_dependencies()
    }

    fn run(
        &self,
        ctx: C,
        prompt: String,
        attachment: Option<Vec<u8>>,
    ) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        let agent = self.0.clone();
        Box::pin(async move { agent.run(ctx, prompt, attachment).await })
    }
}

/// Collection of registered agents with lookup and execution capabilities
///
/// # Type Parameters
/// - `C`: The context type that implements `AgentContext`
#[derive(Default)]
pub struct AgentSet<C: AgentContext> {
    pub set: BTreeMap<String, Box<dyn AgentDyn<C>>>,
}

impl<C> AgentSet<C>
where
    C: AgentContext + Send + Sync + 'static,
{
    /// Creates a new empty AgentSet
    pub fn new() -> Self {
        Self {
            set: BTreeMap::new(),
        }
    }

    /// Checks if an agent with given name exists
    pub fn contains(&self, name: &str) -> bool {
        self.set.contains_key(&name.to_ascii_lowercase())
    }

    /// Retrieves definition for a specific agent
    pub fn definition(&self, name: &str) -> Option<FunctionDefinition> {
        self.set
            .get(&name.to_ascii_lowercase())
            .map(|tool| tool.definition())
    }

    /// Returns definitions for all or specified agents
    ///
    /// # Arguments
    /// - `names`: Optional slice of agent names to filter by
    ///
    /// # Returns
    /// - `Vec<FunctionDefinition>`: Vector of agent definitions
    pub fn definitions(&self, names: Option<&[&str]>) -> Vec<FunctionDefinition> {
        let names: Option<Vec<String>> =
            names.map(|names| names.iter().map(|n| n.to_ascii_lowercase()).collect());
        self.set
            .iter()
            .filter_map(|(name, tool)| match &names {
                Some(names) => {
                    if names.contains(name) {
                        Some(tool.definition())
                    } else {
                        None
                    }
                }
                None => Some(tool.definition()),
            })
            .collect()
    }

    /// Registers a new agent in the set
    ///
    /// # Arguments
    /// - `agent`: The agent to register, must implement `Agent<C>`
    pub fn add<T>(&mut self, agent: T) -> Result<(), BoxError>
    where
        T: Agent<C> + Send + Sync + 'static,
    {
        let name = agent.name().to_ascii_lowercase();
        if self.set.contains_key(&name) {
            return Err(format!("agent {} already exists", name).into());
        }
        validate_path_part(&name)?;
        let agent_dyn = AgentWrapper(Arc::new(agent), PhantomData);
        self.set.insert(name, Box::new(agent_dyn));
        Ok(())
    }

    /// Executes a specific agent with given parameters
    ///
    /// # Arguments
    /// - `name`: The name of the agent to execute
    /// - `ctx`: The execution context
    /// - `prompt`: The input prompt
    /// - `attachment`: Optional additional data
    ///
    /// # Returns
    /// - A boxed future resolving to `Result<AgentOutput, BoxError>`
    ///
    /// # Errors
    /// - Returns an error if the agent is not found
    pub fn run(
        &self,
        name: &str,
        ctx: C,
        prompt: String,
        attachment: Option<Vec<u8>>,
    ) -> BoxPinFut<Result<AgentOutput, BoxError>> {
        if let Some(agent) = self.set.get(&name.to_ascii_lowercase()) {
            agent.run(ctx, prompt, attachment)
        } else {
            Box::pin(futures::future::ready(Err(format!(
                "agent {name} not found"
            )
            .into())))
        }
    }
}
