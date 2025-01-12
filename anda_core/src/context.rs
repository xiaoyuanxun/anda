use candid::{utils::ArgumentEncoder, CandidType};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{future::Future, time::Duration};

pub use candid::Principal;
pub use object_store::{path::Path, ObjectMeta, PutMode, PutResult};
pub use serde_json::Value;
pub use tokio_util::sync::CancellationToken;

use crate::BoxError;

/// AgentContext provides the execution environment for Agents.
/// It combines core functionality with AI-specific features:
/// - BaseContext: Fundamental operations
/// - CompletionFeatures: LLM completions and function calling
/// - EmbeddingFeatures: Text embeddings
/// - VectorSearchFeatures: Semantic search
pub trait AgentContext:
    BaseContext
    + CompletionFeatures<Self::Error>
    + EmbeddingFeatures<Self::Error>
    + VectorSearchFeatures<Self::Error>
{
    /// Executes a local tool with provided arguments
    fn tool_call(
        &self,
        tool_name: &str,
        args: &Value,
    ) -> impl Future<Output = Result<Value, Self::Error>> + Send;

    /// Executes a remote tool on another agent
    fn remote_tool_call(
        &self,
        endpoint: &str,
        tool_name: &str,
        args: &Value,
    ) -> impl Future<Output = Result<Value, Self::Error>> + Send;

    /// Runs a local agent with optional attachment
    fn agent_run(
        &self,
        agent_name: &str,
        prompt: &str,
        attachment: Option<Value>,
    ) -> impl Future<Output = Result<AgentOutput, Self::Error>> + Send;

    /// Runs a remote agent on another endpoint
    fn remote_agent_run(
        &self,
        endpoint: &str,
        agent_name: &str,
        prompt: &str,
        attachment: Option<Value>,
    ) -> impl Future<Output = Result<AgentOutput, Self::Error>> + Send;
}

/// BaseContext is the core context interface available when calling Agent or Tool.
/// It provides access to various feature sets including:
/// - StateFeatures: User, caller, time, and cancellation token
/// - KeysFeatures: Cryptographic key operations
/// - StoreFeatures: Persistent storage
/// - CacheFeatures: In-memory caching
/// - CanisterFeatures: ICP blockchain interactions
/// - HttpFeatures: HTTP request capabilities
pub trait BaseContext:
    Sized
    + StateFeatures<Self::Error>
    + KeysFeatures<Self::Error>
    + StoreFeatures<Self::Error>
    + CacheFeatures<Self::Error>
    + CanisterFeatures<Self::Error>
    + HttpFeatures<Self::Error>
{
    /// Error type for all context operations
    type Error: Into<BoxError>;
}

/// StateFeatures is one of the context feature sets available when calling Agent or Tool.
pub trait StateFeatures<Err>: Sized {
    /// Gets the username from request context.
    /// Note: This is not verified and should not be used as a trusted identifier.
    /// For example, if triggered by a bot of X platform, this might be the username
    /// of the user interacting with the bot.
    fn user(&self) -> String;

    /// Gets the verified caller principal if available.
    /// A non-None value indicates the request has been verified
    /// using ICP blockchain's signature verification algorithm.
    fn caller(&self) -> Option<Principal>;

    /// Gets the cancellation token for the current execution context.
    /// Each call level has its own token scope.
    /// For example, when an agent calls a tool, the tool receives
    /// a child token of the agent's token.
    /// Cancelling the agent's token will cancel all its child calls,
    /// but cancelling a tool's token won't affect its parent agent.
    fn cancellation_token(&self) -> CancellationToken;

    /// Gets the time elapsed since the original context was created
    fn time_elapsed(&self) -> Duration;

    /// Gets current unix timestamp in milliseconds
    fn unix_ms() -> u64;

    /// Generates N random bytes
    fn rand_bytes<const N: usize>() -> [u8; N];
}

/// Represents the output of an agent execution
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct AgentOutput {
    /// The output content from the agent, may be empty
    pub content: String,

    /// Indicates failure reason if present, None means successful execution
    /// Should be None when finish_reason is "stop" or "tool_calls"
    pub failed_reason: Option<String>,

    /// Tool call that this message is responding to. If this message is a response to a tool call, this field should be set to the tool call ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Extracted valid JSON when using JSON response_format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extracted_json: Option<Value>,
}

/// Represents a tool call response with it's ID, function name, and arguments
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: String,

    /// The result of the tool call, processed by agents engine, if available
    pub result: Option<Value>,
}

/// Represents a message in the agent's conversation history
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct MessageInput {
    /// Message role: "developer", "system", "user", "assistant", "tool"
    pub role: String,

    /// The content of the message
    pub content: String,

    /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Tool call that this message is responding to. If this message is a response to a tool call, this field should be set to the tool call ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Defines a callable function with its metadata and schema
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct FunctionDefinition {
    /// Name of the function
    pub name: String,

    /// Description of what the function does
    pub description: String,

    /// JSON schema defining the function's parameters
    pub parameters: serde_json::Value,

    /// Whether to enable strict schema adherence when generating the function call. If set to true, the model will follow the exact schema defined in the parameters field. Only a subset of JSON Schema is supported when strict is true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Struct representing a general completion request that can be sent to a completion model provider.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct CompletionRequest {
    /// The prompt to be sent to the completion model provider as "developer" or "system" role
    pub prompt: String,

    /// The preamble to be sent to the completion model provider
    pub preamble: Option<String>,

    /// The chat history to be sent to the completion model provider
    pub chat_history: Vec<MessageInput>,

    /// The tools to be sent to the completion model provider
    pub tools: Vec<FunctionDefinition>,

    /// The temperature to be sent to the completion model provider
    pub temperature: Option<f64>,

    /// The max tokens to be sent to the completion model provider
    pub max_tokens: Option<u64>,

    /// An object specifying the JSON format that the model must output.
    /// https://platform.openai.com/docs/guides/structured-outputs
    /// The format can be one of the following:
    /// `{ "type": "json_object" }`
    /// `{ "type": "json_schema", "json_schema": {...} }`
    pub response_format: Option<Value>,

    /// The stop sequence to be sent to the completion model provider
    pub stop: Option<Vec<String>>,
}

/// Provides LLM completion capabilities for agents
pub trait CompletionFeatures<Err>: Sized {
    /// Generates a completion based on the given prompt and context
    ///
    /// # Arguments
    /// * `prompt` - The input prompt for the completion
    /// * `json_output` - Whether to force JSON output format
    /// * `chat_history` - Conversation history as context
    /// * `tools` - Available functions the model can call
    fn completion(
        &self,
        req: CompletionRequest,
    ) -> impl Future<Output = Result<AgentOutput, Err>> + Send;
}

/// Represents a text embedding with its original text and vector representation
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Embedding {
    /// The original text that was embedded
    pub text: String,

    /// The embedding vector (typically high-dimensional float array)
    pub vec: Vec<f32>,
}

/// Provides text embedding capabilities for agents
pub trait EmbeddingFeatures<Err>: Sized {
    /// The number of dimensions in the embedding vector.
    fn ndims(&self) -> usize;

    /// Generates embeddings for multiple texts in a batch
    /// Returns a vector of Embedding structs in the same order as input texts
    fn embed(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> impl Future<Output = Result<Vec<Embedding>, Err>> + Send;

    /// Generates a single embedding for a query text
    /// Optimized for single text embedding generation
    fn embed_query(&self, text: &str) -> impl Future<Output = Result<Embedding, Err>> + Send;
}

/// Provides vector search capabilities for semantic similarity search
pub trait VectorSearchFeatures<Err>: Sized {
    /// Performs a semantic search to find top n most similar documents
    /// Returns a list of deserialized json document
    fn top_n<T>(&self, query: &str, n: usize) -> impl Future<Output = Result<Vec<T>, Err>> + Send
    where
        T: DeserializeOwned;

    /// Performs a semantic search but returns only document IDs
    /// More efficient when only document identifiers are needed
    fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<String>, Err>> + Send;
}

/// KeysFeatures is one of the context feature sets available when calling Agent or Tool.
///
/// The Agent engine running in TEE has a permanent fixed 48-bit root key,
/// from which AES, Ed25519, Secp256k1 keys are derived.
/// The Agent/Tool name is included in key derivation, ensuring isolation
/// even with the same derivation path.
pub trait KeysFeatures<Err>: Sized {
    /// Derives a 256-bit AES-GCM key from the given derivation path
    fn a256gcm_key(
        &self,
        derivation_path: &[&[u8]],
    ) -> impl Future<Output = Result<[u8; 32], Err>> + Send;

    /// Signs a message using Ed25519 signature scheme from the given derivation path
    fn ed25519_sign_message(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
    ) -> impl Future<Output = Result<[u8; 64], Err>> + Send;

    /// Verifies an Ed25519 signature from the given derivation path
    fn ed25519_verify(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
        signature: &[u8],
    ) -> impl Future<Output = Result<(), Err>> + Send;

    /// Gets the public key for Ed25519 from the given derivation path
    fn ed25519_public_key(
        &self,
        derivation_path: &[&[u8]],
    ) -> impl Future<Output = Result<[u8; 32], Err>> + Send;

    /// Signs a message using Secp256k1 BIP340 Schnorr signature from the given derivation path
    fn secp256k1_sign_message_bip340(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
    ) -> impl Future<Output = Result<[u8; 64], Err>> + Send;

    /// Verifies a Secp256k1 BIP340 Schnorr signature from the given derivation path
    fn secp256k1_verify_bip340(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
        signature: &[u8],
    ) -> impl Future<Output = Result<(), Err>> + Send;

    /// Signs a message using Secp256k1 ECDSA signature from the given derivation path
    fn secp256k1_sign_message_ecdsa(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
    ) -> impl Future<Output = Result<[u8; 64], Err>> + Send;

    /// Verifies a Secp256k1 ECDSA signature from the given derivation path
    fn secp256k1_verify_ecdsa(
        &self,
        derivation_path: &[&[u8]],
        message: &[u8],
        signature: &[u8],
    ) -> impl Future<Output = Result<(), Err>> + Send;

    /// Gets the compressed SEC1-encoded public key for Secp256k1 from the given derivation path
    fn secp256k1_public_key(
        &self,
        derivation_path: &[&[u8]],
    ) -> impl Future<Output = Result<[u8; 33], Err>> + Send;
}

/// StoreFeatures is one of the context feature sets available when calling Agent or Tool.
///
/// Provides persistent storage capabilities for Agents and Tools to store and manage data.
/// All operations are asynchronous and return Result types with custom error handling.
pub trait StoreFeatures<Err>: Sized {
    /// Retrieves data from storage at the specified path
    fn store_get(
        &self,
        path: &Path,
    ) -> impl Future<Output = Result<(bytes::Bytes, ObjectMeta), Err>> + Send;

    /// Lists objects in storage with optional prefix and offset filters
    ///
    /// # Arguments
    /// * `prefix` - Optional path prefix to filter results
    /// * `offset` - Optional path to start listing from (exclude)
    fn store_list(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> impl Future<Output = Result<Vec<ObjectMeta>, Err>> + Send;

    /// Stores data at the specified path with a given write mode
    ///
    /// # Arguments
    /// * `path` - Target storage path
    /// * `mode` - Write mode (Create, Overwrite, etc.)
    /// * `val` - Data to store as bytes
    fn store_put(
        &self,
        path: &Path,
        mode: PutMode,
        val: bytes::Bytes,
    ) -> impl Future<Output = Result<PutResult, Err>> + Send;

    /// Renames a storage object if the target path doesn't exist
    ///
    /// # Arguments
    /// * `from` - Source path
    /// * `to` - Destination path
    fn store_rename_if_not_exists(
        &self,
        from: &Path,
        to: &Path,
    ) -> impl Future<Output = Result<(), Err>> + Send;

    /// Deletes data at the specified path
    ///
    /// # Arguments
    /// * `path` - Path of the object to delete
    fn store_delete(&self, path: &Path) -> impl Future<Output = Result<(), Err>> + Send;
}

/// Cache expiration policy for cached items
#[derive(Debug, Clone)]
pub enum CacheExpiry {
    /// Time-to-Live: Entry expires after duration from when it was set
    TTL(Duration),
    /// Time-to-Idle: Entry expires after duration from last access
    TTI(Duration),
}

/// CacheFeatures is one of the context feature sets available when calling Agent or Tool.
///
/// Provides isolated in-memory cache storage with TTL/TTI expiration.
/// Cache data is ephemeral and will be lost on engine restart.
pub trait CacheFeatures<Err>: Sized {
    /// Checks if a key exists in the cache
    fn cache_contains(&self, key: &str) -> bool;

    /// Gets a cached value by key, returns error if not found or deserialization fails
    fn cache_get<T>(&self, key: &str) -> impl Future<Output = Result<T, Err>> + Send
    where
        T: DeserializeOwned;

    /// Gets a cached value or initializes it if missing
    ///
    /// If key doesn't exist, calls init function to create value and cache it
    fn cache_get_with<T, F>(
        &self,
        key: &str,
        init: F,
    ) -> impl Future<Output = Result<T, Err>> + Send
    where
        T: Sized + DeserializeOwned + Serialize + Send,
        F: Future<Output = Result<(T, Option<CacheExpiry>), Err>> + Send + 'static;

    /// Sets a value in cache with optional expiration policy
    fn cache_set<T>(
        &self,
        key: &str,
        val: (T, Option<CacheExpiry>),
    ) -> impl Future<Output = ()> + Send
    where
        T: Sized + Serialize + Send;

    /// Deletes a cached value by key, returns true if key existed
    fn cache_delete(&self, key: &str) -> impl Future<Output = bool> + Send;
}

/// CanisterFeatures is one of the context feature sets available when calling Agent or Tool.
///
/// Allows Agents/Tools to interact with any canister contract on the ICP blockchain.
/// The Agent engine will sign canister requests, and they share the same identity ID.
/// A single TEE instance runs only one Agent engine and has only one ICP identity.
pub trait CanisterFeatures<Err>: Sized {
    /// Performs a query call to a canister (read-only, no state changes)
    ///
    /// # Arguments
    /// * `canister` - Target canister principal
    /// * `method` - Method name to call
    /// * `args` - Input arguments encoded in Candid format
    fn canister_query<
        In: ArgumentEncoder + Send,
        Out: CandidType + for<'a> candid::Deserialize<'a>,
    >(
        &self,
        canister: &Principal,
        method: &str,
        args: In,
    ) -> impl Future<Output = Result<Out, Err>> + Send;

    /// Performs an update call to a canister (may modify state)
    ///
    /// # Arguments
    /// * `canister` - Target canister principal
    /// * `method` - Method name to call
    /// * `args` - Input arguments encoded in Candid format
    fn canister_update<
        In: ArgumentEncoder + Send,
        Out: CandidType + for<'a> candid::Deserialize<'a>,
    >(
        &self,
        canister: &Principal,
        method: &str,
        args: In,
    ) -> impl Future<Output = Result<Out, Err>> + Send;
}

/// HttpFeatures provides HTTP request capabilities for Agents and Tools
///
/// All HTTP requests are managed and scheduled by the Agent engine.
/// Since Agents may run in WASM containers, implementations should not
/// implement HTTP requests directly.
pub trait HttpFeatures<Err>: Sized {
    /// Makes an HTTPs request
    ///
    /// # Arguments
    /// * `url` - Target URL, should start with `https://`
    /// * `method` - HTTP method (GET, POST, etc.)
    /// * `headers` - Optional HTTP headers
    /// * `body` - Optional request body (default empty)
    fn https_call(
        &self,
        url: &str,
        method: http::Method,
        headers: Option<http::HeaderMap>,
        body: Option<Vec<u8>>, // default is empty
    ) -> impl Future<Output = Result<reqwest::Response, Err>> + Send;

    /// Makes a signed HTTPs request with message authentication
    ///
    /// # Arguments
    /// * `url` - Target URL
    /// * `method` - HTTP method (GET, POST, etc.)
    /// * `message_digest` - 32-byte message digest for signing
    /// * `headers` - Optional HTTP headers
    /// * `body` - Optional request body (default empty)
    fn https_signed_call(
        &self,
        url: &str,
        method: http::Method,
        message_digest: &[u8; 32],
        headers: Option<http::HeaderMap>,
        body: Option<Vec<u8>>, // default is empty
    ) -> impl Future<Output = Result<reqwest::Response, Err>> + Send;

    /// Makes a signed CBOR-encoded RPC call
    ///
    /// # Arguments
    /// * `endpoint` - URL endpoint to send the request to
    /// * `method` - RPC method name to call
    /// * `params` - Parameters to serialize as CBOR and send with the request
    fn https_signed_rpc<T>(
        &self,
        endpoint: &str,
        method: &str,
        params: impl Serialize + Send,
    ) -> impl Future<Output = Result<T, Err>> + Send
    where
        T: DeserializeOwned;
}
