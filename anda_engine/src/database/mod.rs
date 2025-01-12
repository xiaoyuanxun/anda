use anda_core::{join_path, BoxError, BoxPinFut, ObjectMeta, Path, PutMode, PutResult};
use futures::TryStreamExt;
use object_store::{ObjectStore, PutOptions};
use std::sync::Arc;

pub trait VectorSearchFeaturesDyn: Send + Sync + 'static {
    fn top_n(
        &self,
        namespace: Path,
        query: String,
        n: usize,
    ) -> BoxPinFut<Result<Vec<(String, bytes::Bytes)>, BoxError>>;

    fn top_n_ids(
        &self,
        namespace: Path,
        query: String,
        n: usize,
    ) -> BoxPinFut<Result<Vec<String>, BoxError>>;
}

#[derive(Clone)]
pub struct VectorStore {
    store: Arc<dyn ObjectStore>,
}

impl VectorStore {
    pub fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self { store }
    }
}

impl VectorSearchFeaturesDyn for VectorStore {
    fn top_n(
        &self,
        namespace: Path,
        query: String,
        n: usize,
    ) -> BoxPinFut<Result<Vec<(String, bytes::Bytes)>, BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
    }

    fn top_n_ids(
        &self,
        namespace: Path,
        query: String,
        n: usize,
    ) -> BoxPinFut<Result<Vec<String>, BoxError>> {
        Box::pin(futures::future::ready(Err("not implemented".into())))
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
        let path = join_path(namespace, path);
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
        let prefix = prefix.map(|p| join_path(namespace, p));
        let offset = join_path(namespace, offset);
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
        let path = join_path(namespace, path);
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
        let from = join_path(namespace, from);
        let to = join_path(namespace, to);
        self.store.rename_if_not_exists(&from, &to).await?;
        Ok(())
    }

    /// Deletes data at the specified path
    ///
    /// # Arguments
    /// * `path` - Path of the object to delete
    pub async fn store_delete(&self, namespace: &Path, path: &Path) -> Result<(), BoxError> {
        let path = join_path(namespace, path);
        self.store.delete(&path).await?;
        Ok(())
    }
}
