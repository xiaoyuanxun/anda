use anda_core::context::CacheExpiry;
use anda_core::BoxError;
use bytes::Bytes;
use ciborium::from_reader;
use ic_cose_types::to_cbor_bytes;
use moka::{future::Cache, policy::Expiry};
use object_store::path::Path;
use serde::{de::DeserializeOwned, Serialize};
use std::{
    future::Future,
    time::{Duration, Instant},
};

#[derive(Debug)]
pub struct CacheService {
    cache: Cache<String, (Bytes, Option<CacheExpiry>)>,
}

impl CacheService {
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
    pub fn cache_contains(&self, path: &Path, key: &str) -> bool {
        self.cache.contains_key(path.child(key).as_ref())
    }

    /// Gets a cached value by key, returns error if not found or deserialization fails
    pub async fn cache_get<T>(&self, path: &Path, key: &str) -> Result<T, BoxError>
    where
        T: DeserializeOwned,
    {
        if let Some((bytes, _)) = self.cache.get(path.child(key).as_ref()).await {
            from_reader(&bytes[..]).map_err(|e| e.into())
        } else {
            Err(format!("Key {} not found", key).into())
        }
    }

    /// Gets a cached value or initializes it if missing
    ///
    /// If key doesn't exist, calls init function to create value and cache it
    pub async fn cache_get_with<T, F>(&self, path: &Path, key: &str, init: F) -> Result<T, BoxError>
    where
        T: Sized + DeserializeOwned + Serialize + Send,
        F: Future<Output = Result<(T, Option<CacheExpiry>), BoxError>> + Send + 'static,
    {
        futures_util::pin_mut!(init);
        match self
            .cache
            .try_get_with(path.child(key).into(), async move {
                match init.await {
                    Ok((val, expiry)) => {
                        let data = to_cbor_bytes(&val);
                        Ok((data.into(), expiry))
                    }
                    Err(e) => Err(e),
                }
            })
            .await
        {
            Ok((bytes, _)) => from_reader(&bytes[..]).map_err(|e| e.into()),
            Err(e) => Err(format!("Key {} init failed: {}", key, e).into()),
        }
    }

    /// Sets a value in cache with optional expiration policy
    pub async fn cache_set<T>(&self, path: &Path, key: &str, val: (T, Option<CacheExpiry>))
    where
        T: Sized + Serialize + Send,
    {
        let data = to_cbor_bytes(&val.0);
        self.cache
            .insert(path.child(key).into(), (data.into(), val.1))
            .await;
    }

    /// Deletes a cached value by key, returns true if key existed
    pub async fn cache_delete(&self, path: &Path, key: &str) -> bool {
        self.cache.remove(path.child(key).as_ref()).await.is_some()
    }
}

struct CacheServiceExpiry;

impl Expiry<String, (Bytes, Option<CacheExpiry>)> for CacheServiceExpiry {
    fn expire_after_create(
        &self,
        _key: &String,
        value: &(Bytes, Option<CacheExpiry>),
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
        value: &(Bytes, Option<CacheExpiry>),
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
        value: &(Bytes, Option<CacheExpiry>),
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
