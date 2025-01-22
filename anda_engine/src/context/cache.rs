//! In-memory caching system for AI Agent components
//!
//! This module provides a thread-safe, in-memory LRU cache implementation with expiration policies
//! for storing serialized data. The cache is primarily used by AI Agents and Tools to store
//! frequently accessed data with configurable expiration policies.
//!
//! # Key Features
//! - LRU (Least Recently Used) eviction policy
//! - Configurable maximum capacity
//! - Time-to-Idle (TTI) and Time-to-Live (TTL) expiration policies
//! - Thread-safe operations
//! - Automatic serialization/deserialization using CBOR format
//!
//! # Usage
//! The cache is isolated per agent/tool using path-based namespacing. Each agent/tool has its own
//! isolated cache storage within the shared cache instance.
//!
//! # Performance Characteristics
//! - O(1) time complexity for get/set operations
//! - Memory usage scales with cache capacity and item sizes
//! - Automatic eviction of expired items
//!
//! # Limitations
//! - Data is not persisted across system restarts
//! - Maximum cache size is limited by available memory
//! - Serialization/deserialization overhead for large objects

use anda_core::BoxError;
use anda_core::{context::CacheExpiry, path_lowercase};
use bytes::Bytes;
use ciborium::from_reader;
use ic_cose_types::to_cbor_bytes;
use moka::{future::Cache, policy::Expiry};
use object_store::path::Path;
use serde::{de::DeserializeOwned, Serialize};
use std::{
    future::Future,
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Debug)]
pub struct CacheService {
    cache: Cache<String, Arc<(Bytes, Option<CacheExpiry>)>>,
}

/// CacheService provides an in-memory LRU cache with expiration for AI Agent system's agents and tools.
/// 
/// In the Anda Engine implementation, the `path` parameter is derived from agents' or tools' `name`,
/// ensuring that each agent or tool has isolated cache storage.
/// 
/// Note: Data is cached only in memory and will be lost upon system restart. 
/// For persistent storage, use `StoreFeatures`.
impl CacheService {
    /// Creates a new CacheService instance with specified maximum capacity
    ///
    /// # Arguments
    /// * `max_capacity` - Maximum number of items the cache can hold (u64)
    ///
    /// # Default Behavior
    /// - Maximum time-to-idle (TTI): 7 days
    /// - Uses custom expiration policy based on CacheExpiry
    pub fn new(max_capacity: u64) -> Self {
        let expire = CacheServiceExpiry;
        Self {
            cache: Cache::builder()
                .max_capacity(max_capacity)
                // max TTI is 7 days
                .time_to_idle(Duration::from_secs(3600 * 24 * 7))
                .expire_after(expire)
                .build(),
        }
    }
}

impl CacheService {
    /// Checks if a key exists in the cache
    ///
    /// # Arguments
    /// * `path` - The base path for the key
    /// * `key` - The key to check
    ///
    /// # Returns
    /// `true` if key exists, `false` otherwise
    pub fn contains(&self, path: &Path, key: &str) -> bool {
        self.cache
            .contains_key(path_lowercase(&path.child(key)).as_ref())
    }

    /// Retrieves a cached value by key
    ///
    /// # Arguments
    /// * `path` - The base path for the key
    /// * `key` - The key to retrieve
    ///
    /// # Returns
    /// Result containing deserialized value if successful, error otherwise
    pub async fn get<T>(&self, path: &Path, key: &str) -> Result<T, BoxError>
    where
        T: DeserializeOwned,
    {
        if let Some(val) = self
            .cache
            .get(path_lowercase(&path.child(key)).as_ref())
            .await
        {
            from_reader(&val.0[..]).map_err(|err| err.into())
        } else {
            Err(format!("key {} not found", key).into())
        }
    }

    /// Gets a cached value or initializes it if missing
    ///
    /// If key doesn't exist, calls init function to create value and cache it
    ///
    /// # Arguments
    /// * `path` - The base path for the key
    /// * `key` - The key to retrieve or initialize
    /// * `init` - Async function that returns the value and optional expiry
    ///
    /// # Returns
    /// Result containing deserialized value if successful, error otherwise
    pub async fn get_with<T, F>(&self, path: &Path, key: &str, init: F) -> Result<T, BoxError>
    where
        T: Sized + DeserializeOwned + Serialize + Send,
        F: Future<Output = Result<(T, Option<CacheExpiry>), BoxError>> + Send + 'static,
    {
        futures_util::pin_mut!(init);
        match self
            .cache
            .try_get_with(path_lowercase(&path.child(key)).into(), async move {
                match init.await {
                    Ok((val, expiry)) => {
                        let data = to_cbor_bytes(&val);
                        Ok(Arc::new((data.into(), expiry)))
                    }
                    Err(e) => Err(e),
                }
            })
            .await
        {
            Ok(val) => from_reader(&val.0[..]).map_err(|e| e.into()),
            Err(err) => Err(format!("key {} init failed: {}", key, err).into()),
        }
    }

    /// Sets a value in cache with optional expiration policy
    ///
    /// # Arguments
    /// * `path` - The base path for the key
    /// * `key` - The key to set
    /// * `val` - Tuple containing value and optional expiry policy
    pub async fn set<T>(&self, path: &Path, key: &str, val: (T, Option<CacheExpiry>))
    where
        T: Sized + Serialize + Send,
    {
        let data = to_cbor_bytes(&val.0);
        self.cache
            .insert(
                path_lowercase(&path.child(key)).into(),
                Arc::new((data.into(), val.1)),
            )
            .await;
    }

    /// Deletes a cached value by key
    ///
    /// # Arguments
    /// * `path` - The base path for the key
    /// * `key` - The key to delete
    ///
    /// # Returns
    /// `true` if key existed and was deleted, `false` otherwise
    pub async fn delete(&self, path: &Path, key: &str) -> bool {
        self.cache
            .remove(path_lowercase(&path.child(key)).as_ref())
            .await
            .is_some()
    }
}

struct CacheServiceExpiry;

impl Expiry<String, Arc<(Bytes, Option<CacheExpiry>)>> for CacheServiceExpiry {
    fn expire_after_create(
        &self,
        _key: &String,
        value: &Arc<(Bytes, Option<CacheExpiry>)>,
        _created_at: Instant,
    ) -> Option<Duration> {
        match value.1 {
            Some(CacheExpiry::TTL(du)) => Some(du),
            Some(CacheExpiry::TTI(du)) => Some(du),
            None => None,
        }
    }

    fn expire_after_read(
        &self,
        _key: &String,
        value: &Arc<(Bytes, Option<CacheExpiry>)>,
        _read_at: Instant,
        duration_until_expiry: Option<Duration>,
        _last_modified_at: Instant,
    ) -> Option<Duration> {
        match value.1 {
            Some(CacheExpiry::TTL(_)) => duration_until_expiry,
            Some(CacheExpiry::TTI(du)) => Some(du),
            None => None,
        }
    }

    fn expire_after_update(
        &self,
        _key: &String,
        value: &Arc<(Bytes, Option<CacheExpiry>)>,
        _updated_at: Instant,
        _duration_until_expiry: Option<Duration>,
    ) -> Option<Duration> {
        match value.1 {
            Some(CacheExpiry::TTL(du)) => Some(du),
            Some(CacheExpiry::TTI(du)) => Some(du),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
    struct Profile {
        name: String,
        age: Option<u8>,
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_cache_service() {
        let path1 = Path::from("path1");
        let path2 = Path::from("path2");
        let cache = CacheService::new(100);
        assert!(!cache.contains(&path1, "key"));
        assert!(cache.get::<Profile>(&path2, "key").await.is_err());

        let profile = Profile {
            name: "Anda".to_string(),
            age: Some(18),
        };
        let p1 = profile.clone();
        let res = cache
            .get_with(&path1, "key", async move {
                Ok((p1, Some(CacheExpiry::TTI(Duration::from_secs(10)))))
            })
            .await
            .unwrap();
        assert_eq!(res, profile);

        let res = cache.get::<Profile>(&path1, "key").await.unwrap();
        assert_eq!(res, profile);
        assert!(cache.get::<Profile>(&path2, "key").await.is_err());

        cache
            .set(
                &path1,
                "key",
                (
                    Profile {
                        name: "Anda".to_string(),
                        age: Some(19),
                    },
                    Some(CacheExpiry::TTI(Duration::from_secs(10))),
                ),
            )
            .await;
        let res = cache.get::<Profile>(&path1, "key").await.unwrap();
        assert_ne!(res, profile);
        assert_eq!(res.age, Some(19));

        cache.delete(&path1, "key").await;
        assert!(cache.get::<Profile>(&path1, "key").await.is_err());
    }
}
