mod agent;

pub use agent::*;

#[cfg(test)]
mod tests {
    use super::*;
    use anda_db::database::{AndaDB, DBConfig};
    use anda_engine::context::{Web3Client, Web3SDK};
    use object_store::memory::InMemory;
    use std::sync::Arc;

    fn assert_send<T: Send>(_: &T) {}

    async fn build_future() {
        let object_store = Arc::new(InMemory::new());

        let db_config = DBConfig::default();

        let db = AndaDB::connect(object_store.clone(), db_config)
            .await
            .unwrap();
        let db = Arc::new(db);
        let web3 = Arc::new(Web3SDK::Web3(Web3Client::not_implemented()));
        let _agent = Assistant::connect(db, web3).await.unwrap();
    }

    #[tokio::test]
    #[ignore = "test is used for compilation errors"]
    async fn test_async_send_lifetime() {
        let fut = build_future();
        assert_send(&fut); // 编译报错信息会更聚焦
        fut.await;
    }
}
