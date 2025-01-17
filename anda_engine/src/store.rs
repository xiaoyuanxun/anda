use anda_core::{path_lowercase, BoxError, BoxPinFut, ObjectMeta, Path, PutMode, PutResult};
use futures::TryStreamExt;
use object_store::{ObjectStore, PutOptions};
use std::sync::Arc;

pub const MAX_STORE_OBJECT_SIZE: usize = 1024 * 1024 * 2; // 2 MB

pub trait VectorSearchFeaturesDyn: Send + Sync + 'static {
    fn top_n(
        &self,
        namespace: Path,
        query: String,
        n: usize,
    ) -> BoxPinFut<Result<Vec<String>, BoxError>>;

    fn top_n_ids(
        &self,
        namespace: Path,
        query: String,
        n: usize,
    ) -> BoxPinFut<Result<Vec<String>, BoxError>>;
}

#[derive(Clone)]
pub struct VectorStore {
    inner: Arc<dyn VectorSearchFeaturesDyn>,
}

impl VectorStore {
    pub fn new(inner: Arc<dyn VectorSearchFeaturesDyn>) -> Self {
        Self { inner }
    }

    pub fn not_implemented() -> Self {
        Self {
            inner: Arc::new(NotImplemented),
        }
    }
}

impl VectorSearchFeaturesDyn for VectorStore {
    fn top_n(
        &self,
        namespace: Path,
        query: String,
        n: usize,
    ) -> BoxPinFut<Result<Vec<String>, BoxError>> {
        self.inner.top_n(namespace, query, n)
    }

    fn top_n_ids(
        &self,
        namespace: Path,
        query: String,
        n: usize,
    ) -> BoxPinFut<Result<Vec<String>, BoxError>> {
        self.inner.top_n_ids(namespace, query, n)
    }
}

/// A placeholder for not implemented features.
#[derive(Clone, Debug)]
pub struct NotImplemented;

impl VectorSearchFeaturesDyn for NotImplemented {
    fn top_n(
        &self,
        _namespace: Path,
        _query: String,
        _n: usize,
    ) -> BoxPinFut<Result<Vec<String>, BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }

    fn top_n_ids(
        &self,
        _namespace: Path,
        _query: String,
        _n: usize,
    ) -> BoxPinFut<Result<Vec<String>, BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }
}

#[derive(Clone, Debug)]
pub struct MockImplemented;

impl VectorSearchFeaturesDyn for MockImplemented {
    fn top_n(
        &self,
        _namespace: Path,
        _query: String,
        _n: usize,
    ) -> BoxPinFut<Result<Vec<String>, BoxError>> {
        Box::pin(futures::future::ready(Ok(vec![])))
    }

    fn top_n_ids(
        &self,
        _namespace: Path,
        _query: String,
        _n: usize,
    ) -> BoxPinFut<Result<Vec<String>, BoxError>> {
        Box::pin(futures::future::ready(Ok(vec![])))
    }
}

#[derive(Clone)]
pub struct Store {
    store: Arc<dyn ObjectStore>,
}

impl Store {
    pub fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self { store }
    }

    /// Retrieves data from storage at the specified path
    pub async fn store_get(
        &self,
        namespace: &Path,
        path: &Path,
    ) -> Result<(bytes::Bytes, ObjectMeta), BoxError> {
        let path = path_lowercase(&namespace.child(path.as_ref()));
        let res = self.store.get_opts(&path, Default::default()).await?;
        let data = match res.payload {
            object_store::GetResultPayload::Stream(mut stream) => {
                let mut buf = bytes::BytesMut::new();
                while let Some(data) = stream.try_next().await? {
                    buf.extend_from_slice(&data);
                }
                buf.freeze() // Convert to immutable Bytes
            }
            _ => return Err("StoreFeatures: unexpected payload from get_opts".into()),
        };
        Ok((data, res.meta))
    }

    /// Lists objects in storage with optional prefix and offset filters
    ///
    /// # Arguments
    /// * `prefix` - Optional path prefix to filter results
    /// * `offset` - Optional path to start listing from (exclude)
    pub async fn store_list(
        &self,
        namespace: &Path,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> Result<Vec<ObjectMeta>, BoxError> {
        let prefix = prefix.map(|p| path_lowercase(&namespace.child(p.as_ref())));
        let offset = path_lowercase(&namespace.child(offset.as_ref()));
        let mut res = self.store.list_with_offset(prefix.as_ref(), &offset);
        let mut metas = Vec::new();
        while let Some(meta) = res.try_next().await? {
            metas.push(meta)
        }

        Ok(metas)
    }

    /// Stores data at the specified path with a given write mode
    ///
    /// # Arguments
    /// * `path` - Target storage path
    /// * `mode` - Write mode (Create, Overwrite, etc.)
    /// * `val` - Data to store as bytes
    pub async fn store_put(
        &self,
        namespace: &Path,
        path: &Path,
        mode: PutMode,
        val: bytes::Bytes,
    ) -> Result<PutResult, BoxError> {
        let path = path_lowercase(&namespace.child(path.as_ref()));
        let res = self
            .store
            .put_opts(
                &path,
                val.into(),
                PutOptions {
                    mode,
                    ..Default::default()
                },
            )
            .await?;
        Ok(res)
    }

    /// Renames a storage object if the target path doesn't exist
    ///
    /// # Arguments
    /// * `from` - Source path
    /// * `to` - Destination path
    pub async fn store_rename_if_not_exists(
        &self,
        namespace: &Path,
        from: &Path,
        to: &Path,
    ) -> Result<(), BoxError> {
        let from = path_lowercase(&namespace.child(from.as_ref()));
        let to = path_lowercase(&namespace.child(to.as_ref()));
        self.store.rename_if_not_exists(&from, &to).await?;
        Ok(())
    }

    /// Deletes data at the specified path
    ///
    /// # Arguments
    /// * `path` - Path of the object to delete
    pub async fn store_delete(&self, namespace: &Path, path: &Path) -> Result<(), BoxError> {
        let path = path_lowercase(&namespace.child(path.as_ref()));
        self.store.delete(&path).await?;
        Ok(())
    }
}
