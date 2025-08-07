use anda_core::{BoxError, Xid};
use anda_db::{
    collection::{Collection, CollectionConfig},
    database::AndaDB,
    error::DBError,
    query::{Filter, RangeQuery},
};
use anda_db_schema::Fv;
use anda_db_tfs::jieba_tokenizer;
use async_trait::async_trait;
use candid::Principal;
use std::sync::Arc;

use super::{BaseManagement, Management, Thread, User, UserState, Visibility};

pub struct AndaManagement {
    users: Arc<Collection>,
    threads: Arc<Collection>,
    base: BaseManagement,
}

impl AndaManagement {
    pub async fn connect(base: BaseManagement, db: Arc<AndaDB>) -> Result<Self, BoxError> {
        let schema = User::schema()?;
        let users = db
            .open_or_create_collection(
                schema,
                CollectionConfig {
                    name: "users".to_string(),
                    description: "users collection".to_string(),
                },
                async |collection| {
                    // set tokenizer
                    collection.set_tokenizer(jieba_tokenizer());
                    // create BTree indexes if not exists
                    collection.create_btree_index_nx(&["id"]).await?;

                    Ok::<(), DBError>(())
                },
            )
            .await?;

        let schema = Thread::schema()?;
        let threads = db
            .open_or_create_collection(
                schema,
                CollectionConfig {
                    name: "threads".to_string(),
                    description: "threads collection".to_string(),
                },
                async |collection| {
                    // set tokenizer
                    collection.set_tokenizer(jieba_tokenizer());
                    // create BTree indexes if not exists
                    collection.create_btree_index_nx(&["id"]).await?;
                    collection.create_btree_index_nx(&["managers"]).await?;
                    collection.create_btree_index_nx(&["participants"]).await?;
                    collection
                        .create_bm25_index_nx(&["title", "description"])
                        .await?;

                    Ok::<(), DBError>(())
                },
            )
            .await?;

        Ok(Self {
            users,
            threads,
            base,
        })
    }
}

#[async_trait]
impl Management for AndaManagement {
    /// Returns true if the caller is the controller of the engine.
    fn is_controller(&self, caller: &Principal) -> bool {
        self.base.is_controller(caller)
    }

    /// Returns true if the caller is the controller or a manager of the engine.
    fn is_manager(&self, caller: &Principal) -> bool {
        self.base.is_manager(caller)
    }

    fn check_visibility(&self, caller: &Principal) -> Result<Visibility, BoxError> {
        self.base.check_visibility(caller)
    }

    async fn get_user(&self, user: &Principal) -> Result<UserState, BoxError> {
        let mut ids = self
            .users
            .query_ids(
                Filter::Field((
                    "id".to_string(),
                    RangeQuery::Eq(Fv::Bytes(user.as_slice().to_vec())),
                )),
                None,
            )
            .await?;
        let id = ids.pop().ok_or_else(|| DBError::NotFound {
            name: "user".to_string(),
            path: user.to_text(),
            source: "not found in Btree index".into(),
            _id: 0,
        })?;

        let user: User = self.users.get_as(id).await?;
        Ok(UserState::new(user))
    }

    async fn get_thread(&self, id: &Xid) -> Result<Thread, BoxError> {
        let mut ids = self
            .threads
            .query_ids(
                Filter::Field((
                    "id".to_string(),
                    RangeQuery::Eq(Fv::Bytes(id.as_slice().to_vec())),
                )),
                None,
            )
            .await?;
        let id = ids.pop().ok_or_else(|| DBError::NotFound {
            name: "thread".to_string(),
            path: id.to_string(),
            source: "not found in Btree index".into(),
            _id: 0,
        })?;

        let thread: Thread = self.threads.get_as(id).await?;
        Ok(thread)
    }
}
