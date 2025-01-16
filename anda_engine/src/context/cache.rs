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
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Debug)]
pub struct CacheService {
    cache: Cache<String, Arc<(Bytes, Option<CacheExpiry>)>>,
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
    pub fn contains(&self, path: &Path, key: &str) -> bool {
        self.cache.contains_key(path.child(key).as_ref())
    }

    /// Gets a cached value by key, returns error if not found or deserialization fails
    pub async fn get<T>(&self, path: &Path, key: &str) -> Result<T, BoxError>
    where
        T: DeserializeOwned,
    {
        if let Some(val) = self.cache.get(path.child(key).as_ref()).await {
            from_reader(&val.0[..]).map_err(|err| err.into())
        } else {
            Err(format!("key {} not found", key).into())
        }
    }

    /// Gets a cached value or initializes it if missing
    ///
    /// If key doesn't exist, calls init function to create value and cache it
    pub async fn get_with<T, F>(&self, path: &Path, key: &str, init: F) -> Result<T, BoxError>
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
    pub async fn set<T>(&self, path: &Path, key: &str, val: (T, Option<CacheExpiry>))
    where
        T: Sized + Serialize + Send,
    {
        let data = to_cbor_bytes(&val.0);
        self.cache
            .insert(path.child(key).into(), Arc::new((data.into(), val.1)))
            .await;
    }

    /// Deletes a cached value by key, returns true if key existed
    pub async fn delete(&self, path: &Path, key: &str) -> bool {
        self.cache.remove(path.child(key).as_ref()).await.is_some()
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
