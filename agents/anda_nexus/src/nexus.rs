use anda_core::{BoxError, Resource, Xid};
use anda_db::{
    collection::{Collection, CollectionConfig},
    database::AndaDB,
    error::DBError,
    index::BTree,
    query::{Filter, Query, RangeQuery},
};
use anda_db_schema::Fv;
use anda_db_tfs::jieba_tokenizer;
use anda_engine::unix_ms;

use candid::Principal;
use futures::stream::{self, StreamExt};
use parking_lot::RwLock;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use crate::types::*;

#[derive(Debug)]
pub struct NexusNode {
    db: Arc<AndaDB>,
    threads: Arc<Collection>,
    thread_states: RwLock<BTreeMap<u64, Arc<RwLock<ThreadState>>>>,
}

impl NexusNode {
    fn thread_message_collection_name(id: u64) -> String {
        format!("{}_messages", id)
    }

    fn thread_resource_collection_name(id: u64) -> String {
        format!("{}_resources", id)
    }

    pub async fn connect(db: Arc<AndaDB>) -> Result<Self, BoxError> {
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
                    collection.create_btree_index_nx(&["participants"]).await?;
                    collection
                        .create_bm25_index_nx(&["name", "tags", "description"])
                        .await?;

                    Ok::<(), DBError>(())
                },
            )
            .await?;

        let thread_ids = threads.ids();

        let rt = stream::iter(thread_ids.into_iter())
            .map(|id| {
                let threads = threads.clone();

                async move {
                    let thread: Thread = match threads.get_as(id).await {
                        Ok(thread) => thread,
                        Err(DBError::NotFound { .. }) => return None,
                        Err(err) => return Some(Err(err)),
                    };
                    let state = ThreadState {
                        visibility: thread.visibility,
                        status: thread.status,
                        updated_at: thread.updated_at,
                        max_participants: thread.max_participants,
                        latest_message_by: thread.latest_message_by,
                        latest_message_id: thread.latest_message_id,
                        latest_message_at: thread.latest_message_at,
                    };

                    Some(Ok((thread._id, Arc::new(RwLock::new(state)))))
                }
            })
            .buffer_unordered(64)
            .collect::<Vec<Option<Result<_, DBError>>>>()
            .await;

        let mut thread_states: BTreeMap<u64, Arc<RwLock<ThreadState>>> = BTreeMap::new();
        for r in rt.into_iter().flatten() {
            match r {
                Ok((id, state)) => {
                    thread_states.insert(id, state);
                }
                Err(err) => {
                    log::error!("Failed to load thread: {}", err);
                    continue;
                }
            }
        }

        Ok(Self {
            db,
            threads,
            thread_states: RwLock::new(thread_states),
        })
    }

    async fn get_message_collection(&self, thread_id: u64) -> Result<Arc<Collection>, BoxError> {
        let collection = self
            .db
            .open_collection(
                Self::thread_message_collection_name(thread_id),
                async |collection| {
                    collection.set_tokenizer(jieba_tokenizer());

                    Ok::<(), DBError>(())
                },
            )
            .await?;

        Ok(collection)
    }

    async fn get_resource_collection(&self, thread_id: u64) -> Result<Arc<Collection>, BoxError> {
        let collection = self
            .db
            .open_collection(
                Self::thread_resource_collection_name(thread_id),
                async |collection| {
                    collection.set_tokenizer(jieba_tokenizer());

                    Ok::<(), DBError>(())
                },
            )
            .await?;

        Ok(collection)
    }

    pub async fn create_thread(&self, owner: Principal, name: String) -> Result<Thread, BoxError> {
        let updated_at = unix_ms();
        let mut thread = Thread {
            _id: 0,
            id: Xid::new(),
            name,
            controllers: BTreeSet::from([owner]),
            managers: BTreeSet::from([owner]),
            participants: BTreeMap::from([(owner, 0)]),
            created_at: updated_at,
            updated_at,
            ..Default::default()
        };
        let id = self.threads.add_from(&thread).await.unwrap();
        thread._id = id;

        let schema = Message::schema()?;
        let _ = self
            .db
            .open_or_create_collection(
                schema,
                CollectionConfig {
                    name: Self::thread_message_collection_name(id),
                    description: "Thread messages collection".to_string(),
                },
                async |collection| {
                    collection.set_tokenizer(jieba_tokenizer());
                    collection.create_btree_index_nx(&["user"]).await?;
                    collection.create_bm25_index_nx(&["content"]).await?;

                    Ok::<(), DBError>(())
                },
            )
            .await?;
        let schema = Resource::schema()?;
        let _ = self
            .db
            .open_or_create_collection(
                schema,
                CollectionConfig {
                    name: Self::thread_resource_collection_name(id),
                    description: "Thread resources collection".to_string(),
                },
                async |collection| {
                    collection.set_tokenizer(jieba_tokenizer());
                    collection.create_btree_index_nx(&["tags"]).await?;
                    collection.create_btree_index_nx(&["hash"]).await?;
                    collection.create_btree_index_nx(&["mime_type"]).await?;
                    collection
                        .create_bm25_index_nx(&["name", "description", "metadata"])
                        .await?;

                    Ok::<(), DBError>(())
                },
            )
            .await?;

        self.thread_states
            .write()
            .insert(thread._id, Arc::new(RwLock::new(thread.to_state())));

        Ok(thread)
    }

    pub async fn fetch_my_threads_state(&self, user: Principal) -> Vec<ThreadState> {
        let ids: Vec<u64> = self
            .threads
            .search_ids(Query {
                filter: Some(Filter::Field((
                    "participants".to_string(),
                    RangeQuery::Eq(Fv::Bytes(user.as_slice().to_vec())),
                ))),
                ..Default::default()
            })
            .await
            .unwrap_or_default();
        let mut rt = Vec::with_capacity(ids.len());
        let states = self.thread_states.read();
        for id in ids {
            if let Some(s) = states.get(&id) {
                let s = s.read();
                if s.status == ThreadStatus::Active {
                    rt.push(s.clone());
                }
            }
        }
        rt
    }

    pub fn public_threads_state(&self, ids: BTreeSet<u64>) -> Vec<ThreadState> {
        let mut rt = Vec::with_capacity(ids.len());
        let states = self.thread_states.read();
        for id in ids {
            if let Some(s) = states.get(&id) {
                let s = s.read();
                if s.visibility == ThreadVisibility::Public && s.status == ThreadStatus::Active {
                    rt.push(s.clone());
                }
            }
        }
        rt
    }

    pub async fn get_thread(&self, user: Principal, _id: u64) -> Result<Thread, BoxError> {
        let thread: Thread = self.threads.get_as(_id).await?;
        if thread.has_permission(&user, ThreadPermission::Read) {
            Ok(thread)
        } else {
            Err(format!(
                "User {} does not have permission to access thread {}",
                user, _id
            )
            .into())
        }
    }

    pub async fn list_my_threads(
        &self,
        user: Principal,
        cursor: Option<String>,
        limit: Option<usize>,
    ) -> Result<(Vec<ThreadInfo>, Option<String>), BoxError> {
        let limit = limit.unwrap_or(100).min(1000);
        let cursor = match BTree::from_cursor::<u64>(&cursor)? {
            Some(cursor) => cursor,
            None => 0,
        };
        let mut ids: Vec<u64> = self
            .threads
            .search_ids(Query {
                filter: Some(Filter::Field((
                    "participants".to_string(),
                    RangeQuery::Eq(Fv::Bytes(user.as_slice().to_vec())),
                ))),
                ..Default::default()
            })
            .await
            .unwrap_or_default();
        if cursor > 0 {
            ids.sort();
            ids.retain(|&id| id > cursor);
        }
        if ids.len() > limit {
            ids.truncate(limit);
        }
        let cursor = if ids.len() >= limit {
            BTree::to_cursor(&ids.last().unwrap())
        } else {
            None
        };

        let mut threads = Vec::with_capacity(ids.len());
        for id in ids {
            if let Ok(thread) = self.threads.get_as(id).await {
                threads.push(thread);
            }
        }
        Ok((threads, cursor))
    }

    pub async fn public_threads(&self, limit: Option<usize>) -> Result<Vec<ThreadInfo>, BoxError> {
        let limit = limit.unwrap_or(100).min(1000);

        let mut candidates = {
            let states = self.thread_states.read();
            let mut candidates = Vec::with_capacity(states.len() / 2);
            for (id, state) in states.iter() {
                let s = state.read();
                if (s.visibility == ThreadVisibility::Public
                    || s.visibility == ThreadVisibility::Protected)
                    && s.status == ThreadStatus::Active
                {
                    candidates.push((*id, s.updated_at));
                }
            }
            candidates
        };

        candidates.sort_by(|a, b| b.1.cmp(&a.1)); // sort by updated_at desc
        let mut threads = Vec::with_capacity(limit);
        for (id, _) in candidates.into_iter().take(limit) {
            if let Ok(thread) = self.threads.get_as(id).await {
                threads.push(thread);
            }
        }

        Ok(threads)
    }

    pub async fn update_thread(
        &self,
        user: Principal,
        _id: u64,
        mut input: UpdateThreadInfo,
    ) -> Result<(), BoxError> {
        input.validate_and_normalize()?;

        let thread: Thread = self.threads.get_as(_id).await?;
        if !thread.has_permission(&user, ThreadPermission::Manage) {
            return Err(format!(
                "User {} does not have permission to manage thread {}",
                user, _id
            )
            .into());
        }
        let updated_at = unix_ms();
        let mut changes: BTreeMap<String, Fv> =
            BTreeMap::from([("updated_at".to_string(), Fv::U64(updated_at))]);
        if let Some(name) = input.name {
            changes.insert("name".to_string(), Fv::Text(name));
        }
        if let Some(language) = input.language {
            changes.insert("language".to_string(), Fv::Text(language));
        }
        if let Some(image) = input.image {
            changes.insert("image".to_string(), Fv::Text(image));
        }
        if let Some(tags) = input.tags {
            changes.insert(
                "tags".to_string(),
                Fv::Array(tags.into_iter().map(Fv::Text).collect()),
            );
        }
        if let Some(description) = input.description {
            changes.insert("description".to_string(), Fv::Text(description));
        }
        if let Some(visibility) = &input.visibility {
            changes.insert("visibility".to_string(), Fv::Text(visibility.to_string()));
        }

        let _ = self.threads.update(_id, changes).await?;
        if let Some(state) = self.thread_states.write().get_mut(&_id) {
            let mut s = state.write();
            s.updated_at = updated_at;
            if let Some(visibility) = input.visibility {
                s.visibility = visibility;
            }
        }
        Ok(())
    }

    pub async fn update_thread_controllers(
        &self,
        user: Principal,
        _id: u64,
        controllers: BTreeSet<Principal>,
    ) -> Result<(), BoxError> {
        let mut thread: Thread = self.threads.get_as(_id).await?;
        if !thread.has_permission(&user, ThreadPermission::Control) {
            return Err(format!(
                "User {} does not have permission to control thread {}",
                user, _id
            )
            .into());
        }
        if controllers.is_empty() {
            return Err("Controllers cannot be empty".to_string().into());
        }
        if controllers.len() > 5 {
            return Err("Controllers cannot be more than 5".to_string().into());
        }
        for p in &controllers {
            thread.participants.entry(p.clone()).or_insert(0);
        }
        let controllers_fv = Fv::Array(
            controllers
                .into_iter()
                .map(|p| p.as_ref().to_vec().into())
                .collect(),
        );
        let updated_at = unix_ms();
        let _ = self
            .threads
            .update(
                _id,
                BTreeMap::from([
                    ("controllers".to_string(), controllers_fv),
                    (
                        "participants".to_string(),
                        Fv::Map(
                            thread
                                .participants
                                .into_iter()
                                .map(|(k, v)| (k.as_ref().into(), v.into()))
                                .collect(),
                        ),
                    ),
                    ("updated_at".to_string(), Fv::U64(updated_at)),
                ]),
            )
            .await?;
        if let Some(state) = self.thread_states.write().get_mut(&_id) {
            let mut s = state.write();
            s.updated_at = updated_at;
        }
        Ok(())
    }

    pub async fn update_thread_managers(
        &self,
        user: Principal,
        _id: u64,
        managers: BTreeSet<Principal>,
    ) -> Result<(), BoxError> {
        let mut thread: Thread = self.threads.get_as(_id).await?;
        if !thread.has_permission(&user, ThreadPermission::Control) {
            return Err(format!(
                "User {} does not have permission to control thread {}",
                user, _id
            )
            .into());
        }
        if managers.is_empty() {
            return Err("Managers cannot be empty".to_string().into());
        }
        if managers.len() > 5 {
            return Err("Managers cannot be more than 5".to_string().into());
        }

        for p in &managers {
            thread.participants.entry(p.clone()).or_insert(0);
        }
        let managers_fv = Fv::Array(
            managers
                .into_iter()
                .map(|p| p.as_ref().to_vec().into())
                .collect(),
        );
        let updated_at = unix_ms();
        let _ = self
            .threads
            .update(
                _id,
                BTreeMap::from([
                    ("managers".to_string(), managers_fv),
                    (
                        "participants".to_string(),
                        Fv::Map(
                            thread
                                .participants
                                .into_iter()
                                .map(|(k, v)| (k.as_ref().into(), v.into()))
                                .collect(),
                        ),
                    ),
                    ("updated_at".to_string(), Fv::U64(updated_at)),
                ]),
            )
            .await?;
        if let Some(state) = self.thread_states.write().get_mut(&_id) {
            let mut s = state.write();
            s.updated_at = updated_at;
        }
        Ok(())
    }

    pub async fn add_thread_participants(
        &self,
        user: Principal,
        _id: u64,
        participants: BTreeSet<Principal>,
    ) -> Result<(), BoxError> {
        let mut thread: Thread = self.threads.get_as(_id).await?;
        if !thread.has_permission(&user, ThreadPermission::Manage) {
            return Err(format!(
                "User {} does not have permission to manage thread {}",
                user, _id
            )
            .into());
        }
        if participants.is_empty() {
            return Err("Participants cannot be empty".to_string().into());
        }

        if thread.participants.len() + participants.len() > thread.max_participants as usize {
            return Err(
                format!("Exceed max participants limit: {}", thread.max_participants).into(),
            );
        }

        for p in participants {
            thread.participants.entry(p).or_insert(0);
        }
        let updated_at = unix_ms();
        let _ = self
            .threads
            .update(
                _id,
                BTreeMap::from([
                    (
                        "participants".to_string(),
                        Fv::Map(
                            thread
                                .participants
                                .into_iter()
                                .map(|(k, v)| (k.as_ref().into(), v.into()))
                                .collect(),
                        ),
                    ),
                    ("updated_at".to_string(), Fv::U64(updated_at)),
                ]),
            )
            .await?;
        if let Some(state) = self.thread_states.write().get_mut(&_id) {
            let mut s = state.write();
            s.updated_at = updated_at;
        }
        Ok(())
    }

    pub async fn remove_thread_participants(
        &self,
        user: Principal,
        _id: u64,
        participants: BTreeSet<Principal>,
    ) -> Result<(), BoxError> {
        let mut thread: Thread = self.threads.get_as(_id).await?;
        if !thread.has_permission(&user, ThreadPermission::Manage) {
            return Err(format!(
                "User {} does not have permission to manage thread {}",
                user, _id
            )
            .into());
        }
        if participants.is_empty() {
            return Err("Participants cannot be empty".to_string().into());
        }
        if thread.controllers.intersection(&participants).count() > 0 {
            return Err("Cannot remove controllers from participants"
                .to_string()
                .into());
        }
        if thread.managers.intersection(&participants).count() > 0 {
            return Err("Cannot remove managers from participants"
                .to_string()
                .into());
        }

        for p in participants {
            thread.participants.remove(&p);
        }
        let updated_at = unix_ms();
        let _ = self
            .threads
            .update(
                _id,
                BTreeMap::from([
                    (
                        "participants".to_string(),
                        Fv::Map(
                            thread
                                .participants
                                .into_iter()
                                .map(|(k, v)| (k.as_ref().into(), v.into()))
                                .collect(),
                        ),
                    ),
                    ("updated_at".to_string(), Fv::U64(updated_at)),
                ]),
            )
            .await?;
        if let Some(state) = self.thread_states.write().get_mut(&_id) {
            let mut s = state.write();
            s.updated_at = updated_at;
        }
        Ok(())
    }

    pub async fn quit_thread(&self, user: Principal, _id: u64) -> Result<(), BoxError> {
        let mut thread: Thread = self.threads.get_as(_id).await?;
        if !thread.participants.contains_key(&user) {
            return Err(format!("User {} is not a participant of thread {}", user, _id).into());
        }

        let updated_at = unix_ms();
        let mut changes: BTreeMap<String, Fv> =
            BTreeMap::from([("updated_at".to_string(), Fv::U64(updated_at))]);
        if thread.controllers.contains(&user) {
            thread.controllers.remove(&user);
            if thread.controllers.is_empty() {
                return Err("Cannot quit thread as the last controller"
                    .to_string()
                    .into());
            }
            changes.insert(
                "controllers".to_string(),
                Fv::Array(
                    thread
                        .controllers
                        .into_iter()
                        .map(|p| p.as_ref().to_vec().into())
                        .collect(),
                ),
            );
        }

        if thread.managers.contains(&user) {
            thread.managers.remove(&user);
            changes.insert(
                "managers".to_string(),
                Fv::Array(
                    thread
                        .managers
                        .into_iter()
                        .map(|p| p.as_ref().to_vec().into())
                        .collect(),
                ),
            );
        }

        thread.participants.remove(&user);
        changes.insert(
            "participants".to_string(),
            Fv::Map(
                thread
                    .participants
                    .into_iter()
                    .map(|(k, v)| (k.as_ref().into(), v.into()))
                    .collect(),
            ),
        );

        let _ = self.threads.update(_id, changes).await?;
        if let Some(state) = self.thread_states.write().get_mut(&_id) {
            let mut s = state.write();
            s.updated_at = updated_at;
        }
        Ok(())
    }

    pub async fn delete_thread(&self, user: Principal, _id: u64) -> Result<(), BoxError> {
        let thread: Thread = self.threads.get_as(_id).await?;
        if !thread.has_permission(&user, ThreadPermission::Control) {
            return Err(format!(
                "User {} does not have permission to control thread {}",
                user, _id
            )
            .into());
        }

        Err("Not implemented".to_string().into())
    }

    pub async fn sys_set_thread_status(
        &self,
        _id: u64,
        status: ThreadStatus,
    ) -> Result<(), BoxError> {
        let updated_at = unix_ms();
        let _ = self
            .threads
            .update(
                _id,
                BTreeMap::from([
                    ("status".to_string(), Fv::Text(status.to_string())),
                    ("updated_at".to_string(), Fv::U64(updated_at)),
                ]),
            )
            .await?;
        if let Some(state) = self.thread_states.write().get_mut(&_id) {
            let mut s = state.write();
            s.updated_at = updated_at;
        }
        Ok(())
    }

    pub async fn sys_set_thread_max_participants(
        &self,
        _id: u64,
        max_participants: u64,
    ) -> Result<(), BoxError> {
        let updated_at = unix_ms();
        let _ = self
            .threads
            .update(
                _id,
                BTreeMap::from([
                    ("max_participants".to_string(), Fv::U64(max_participants)),
                    ("updated_at".to_string(), Fv::U64(updated_at)),
                ]),
            )
            .await?;
        if let Some(state) = self.thread_states.write().get_mut(&_id) {
            let mut s = state.write();
            s.updated_at = updated_at;
            s.max_participants = max_participants;
        }
        Ok(())
    }
}
