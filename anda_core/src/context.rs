use candid::{utils::ArgumentEncoder, CandidType, Principal};
use ic_cose_types::types::object_store::{GetResult, ObjectMeta, PutMode};
use object_store::path::Path;
use serde::{de::DeserializeOwned, Serialize};
use std::{
    future::Future,
    ops::{Deref, DerefMut},
    time::Duration,
};
use tokio_util::sync::CancellationToken;

/// A global state manager for Agent or Tool
/// 
/// Wraps any type `S` to provide shared state management with 
/// automatic dereferencing capabilities
#[derive(Debug, Default, Clone, Copy)]
pub struct State<S>(pub S);

impl<S> Deref for State<S> {
    type Target = S;

    /// Provides immutable access to the inner state
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S> DerefMut for State<S> {
    /// Provides mutable access to the inner state
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// BaseContext is the core context interface available when calling Agent or Tool.
/// It provides access to various feature sets including:
/// - KeysFeatures: Cryptographic key operations
/// - StoreFeatures: Persistent storage
/// - CacheFeatures: In-memory caching
/// - CanisterFeatures: ICP blockchain interactions
/// - HttpFeatures: HTTP request capabilities
pub trait BaseContext:
    Sized
    + KeysFeatures<Self::Error>
    + StoreFeatures<Self::Error>
    + CacheFeatures<Self::Error>
    + CanisterFeatures<Self::Error>
    + HttpFeatures<Self::Error>
{
    /// Error type for all context operations
    type Error: std::error::Error;

    /// Gets the username from request context.
    /// Note: This is not verified and should not be used as a trusted identifier.
    /// For example, if triggered by a bot of X platform, this might be the username
    /// of the user interacting with the bot.
    fn user() -> String;

    /// Gets current unix timestamp in milliseconds
    fn unix_ms() -> u64;

    /// Gets the verified caller principal if available.
    /// A non-None value indicates the request has been verified
    /// using ICP blockchain's signature verification algorithm.
    fn caller() -> Option<Principal>;

    /// Gets the cancellation token for the current execution context.
    /// Each call level has its own token scope.
    /// For example, when an agent calls a tool, the tool receives
    /// a child token of the agent's token.
    /// Cancelling the agent's token will cancel all its child calls,
    /// but cancelling a tool's token won't affect its parent agent.
    fn cancellation_token() -> CancellationToken;
}

/// KeysFeatures is one of the context feature sets available when calling Agent or Tool.
/// 
/// The Agent engine running in TEE has a permanent fixed 48-bit root key,
/// from which AES, Ed25519, Secp256k1 keys are derived.
/// The Agent/Tool name is included in key derivation, ensuring isolation
/// even with the same derivation path.
pub trait KeysFeatures<Err>: Sized {
    /// Generates N random bytes
    fn rand_bytes<const N: usize>() -> [u8; N];

    /// Derives a 256-bit AES-GCM key from the given derivation path
    fn a256gcm_key(
        &self,
        derivation_path: Vec<Vec<u8>>,
    ) -> impl Future<Output = Result<[u8; 32], Err>> + Send;

    /// Signs a message using Ed25519 signature scheme from the given derivation path
    fn ed25519_sign_message(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
    ) -> impl Future<Output = Result<[u8; 64], Err>> + Send;

    /// Verifies an Ed25519 signature from the given derivation path
    fn ed25519_verify(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
        signature: &[u8],
    ) -> impl Future<Output = Result<(), Err>> + Send;

    /// Gets the public key for Ed25519 from the given derivation path
    fn ed25519_public_key(
        &self,
        derivation_path: Vec<Vec<u8>>,
    ) -> impl Future<Output = Result<[u8; 32], Err>> + Send;

    /// Signs a message using Secp256k1 BIP340 Schnorr signature from the given derivation path
    fn secp256k1_sign_message_bip340(
        &self,
        derivation_path: Vec<Vec<u8>>,
        msg: &[u8],
    ) -> impl Future<Output = Result<[u8; 64], Err>> + Send;

    /// Verifies a Secp256k1 BIP340 Schnorr signature from the given derivation path
    fn secp256k1_verify_bip340(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
        signature: &[u8],
    ) -> impl Future<Output = Result<(), Err>> + Send;

    /// Signs a message using Secp256k1 ECDSA signature from the given derivation path
    fn secp256k1_sign_message_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        msg: &[u8],
    ) -> impl Future<Output = Result<[u8; 64], Err>> + Send;

    /// Verifies a Secp256k1 ECDSA signature from the given derivation path
    fn secp256k1_verify_ecdsa(
        &self,
        derivation_path: Vec<Vec<u8>>,
        message: &[u8],
        signature: &[u8],
    ) -> impl Future<Output = Result<(), Err>> + Send;

    /// Gets the compressed SEC1-encoded public key for Secp256k1 from the given derivation path
    fn secp256k1_public_key(
        &self,
        derivation_path: Vec<Vec<u8>>,
    ) -> impl Future<Output = Result<[u8; 33], Err>> + Send;
}

/// StoreFeatures is one of the context feature sets available when calling Agent or Tool.
/// 
/// Provides persistent storage capabilities for Agents and Tools to store and manage data.
/// All operations are asynchronous and return Result types with custom error handling.
pub trait StoreFeatures<Err>: Sized {
    /// Retrieves data from storage at the specified path
    fn store_get(&self, path: &Path) -> impl Future<Output = Result<GetResult, Err>> + Send;

    /// Lists objects in storage with optional prefix and offset filters
    /// 
    /// # Arguments
    /// * `prefix` - Optional path prefix to filter results
    /// * `offset` - Optional path to start listing from (exclude)
    fn store_list(
        &self,
        prefix: Option<&Path>,
        offset: Option<&Path>,
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
    ) -> impl Future<Output = Result<(), Err>> + Send;

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
    fn cache_get<T>(&self, key: &str) -> Result<T, Err>
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
        T: DeserializeOwned + Serialize,
        F: Future<Output = Result<(T, Option<CacheExpiry>), Err>>;

    /// Sets a value in cache with optional expiration policy
    fn cache_set<V>(&self, key: &str, val: (V, Option<CacheExpiry>))
    where
        V: Sized + Serialize;

    /// Deletes a cached value by key, returns true if key existed
    fn cache_delete(&self, key: &str) -> bool;
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
        method: &http::Method,
        headers: Option<http::HeaderMap>,
        body: Option<Vec<u8>>, // default is empty
    ) -> impl Future<Output = Result<crate::HttpResponse, Err>> + Send;

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
        method: &http::Method,
        message_digest: &[u8; 32],
        headers: Option<http::HeaderMap>,
        body: Option<Vec<u8>>, // default is empty
    ) -> impl Future<Output = Result<crate::HttpResponse, Err>> + Send;
}
