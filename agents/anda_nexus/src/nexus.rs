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

                    Some(Ok((thread._id, thread.to_state())))
                }
            })
            .buffer_unordered(16)
            .collect::<Vec<Option<Result<_, DBError>>>>()
            .await;

        let mut thread_states: BTreeMap<u64, Arc<RwLock<ThreadState>>> = BTreeMap::new();
        for r in rt.into_iter().flatten() {
            match r {
                Ok((id, state)) => {
                    thread_states.insert(id, Arc::new(RwLock::new(state)));
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

        let latest_message_id = {
            if let Some(state) = self.thread_states.read().get(&thread_id) {
                state.read().latest_message_id
            } else {
                0
            }
        };
        if latest_message_id == 0
            && let Some(id) = collection.latest_document_id()
            && let Ok(message) = collection.get_as::<Message>(id).await
            && let Some(state) = self.thread_states.write().get_mut(&thread_id)
        {
            let mut s = state.write();
            if message._id > s.latest_message_id {
                s.latest_message_by = message.user;
                s.latest_message_id = message._id;
                s.latest_message_at = message.timestamp;
            }
        }

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

    pub async fn fetch_my_threads_state(&self, user: &Principal) -> Vec<ThreadState> {
        let ids: Vec<u64> = self.my_thread_ids(user).await;
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

    pub async fn get_thread(&self, user: &Principal, _id: u64) -> Result<Thread, BoxError> {
        self.check_thread_state(_id)?;

        let thread: Thread = self.threads.get_as(_id).await?;
        if thread.has_permission(user, ThreadPermission::Read) {
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
        user: &Principal,
        cursor: Option<String>,
        limit: Option<usize>,
    ) -> Result<(Vec<ThreadInfo>, Option<String>), BoxError> {
        let limit = limit.unwrap_or(100).min(1000);
        let cursor = (BTree::from_cursor::<u64>(&cursor)?).unwrap_or_default();
        let mut ids: Vec<u64> = self.my_thread_ids(user).await;
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
        user: &Principal,
        _id: u64,
        mut input: UpdateThreadInfo,
    ) -> Result<(), BoxError> {
        input.validate_and_normalize()?;
        self.check_thread_state(_id)?;

        let thread: Thread = self.threads.get_as(_id).await?;
        if !thread.has_permission(user, ThreadPermission::Manage) {
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

        self.threads.update(_id, changes).await?;
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
        user: &Principal,
        _id: u64,
        controllers: BTreeSet<Principal>,
    ) -> Result<(), BoxError> {
        if controllers.is_empty() {
            return Err("Controllers cannot be empty".to_string().into());
        }
        if controllers.len() > 5 {
            return Err("Controllers cannot be more than 5".to_string().into());
        }
        self.check_thread_state(_id)?;

        let mut thread: Thread = self.threads.get_as(_id).await?;
        if !thread.has_permission(user, ThreadPermission::Control) {
            return Err(format!(
                "User {} does not have permission to control thread {}",
                user, _id
            )
            .into());
        }

        for p in &controllers {
            thread.participants.entry(*p).or_insert(0);
        }
        let controllers_fv = Fv::Array(
            controllers
                .into_iter()
                .map(|p| p.as_ref().to_vec().into())
                .collect(),
        );
        let updated_at = unix_ms();
        let participants = thread.participants.len() as u64;
        self.threads
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
            s.participants = participants;
            s.updated_at = updated_at;
        }
        Ok(())
    }

    pub async fn update_thread_managers(
        &self,
        user: &Principal,
        _id: u64,
        managers: BTreeSet<Principal>,
    ) -> Result<(), BoxError> {
        if managers.is_empty() {
            return Err("Managers cannot be empty".to_string().into());
        }
        if managers.len() > 5 {
            return Err("Managers cannot be more than 5".to_string().into());
        }
        self.check_thread_state(_id)?;

        let mut thread: Thread = self.threads.get_as(_id).await?;
        if !thread.has_permission(user, ThreadPermission::Control) {
            return Err(format!(
                "User {} does not have permission to control thread {}",
                user, _id
            )
            .into());
        }

        for p in &managers {
            thread.participants.entry(*p).or_insert(0);
        }
        let managers_fv = Fv::Array(
            managers
                .into_iter()
                .map(|p| p.as_ref().to_vec().into())
                .collect(),
        );
        let updated_at = unix_ms();
        let participants = thread.participants.len() as u64;
        self.threads
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
            s.participants = participants;
            s.updated_at = updated_at;
        }
        Ok(())
    }

    pub async fn add_thread_participants(
        &self,
        user: &Principal,
        _id: u64,
        participants: BTreeSet<Principal>,
    ) -> Result<(), BoxError> {
        if participants.is_empty() {
            return Err("Participants cannot be empty".to_string().into());
        }

        let (num_participants, max_participants) = {
            match self.thread_states.read().get(&_id) {
                Some(state) => {
                    let s = state.read();
                    if s.status != ThreadStatus::Active {
                        return Err(format!("Thread {} is not active", _id).into());
                    }
                    (s.participants, s.max_participants)
                }
                None => return Err(format!("Thread {} not found", _id).into()),
            }
        };

        if num_participants + participants.len() as u64 > max_participants {
            return Err(format!("Exceed max participants limit: {}", max_participants).into());
        }

        let mut thread: Thread = self.threads.get_as(_id).await?;
        if !thread.has_permission(user, ThreadPermission::Manage) {
            return Err(format!(
                "User {} does not have permission to manage thread {}",
                user, _id
            )
            .into());
        }

        for p in participants {
            thread.participants.entry(p).or_insert(0);
        }
        let updated_at = unix_ms();
        let participants = thread.participants.len() as u64;
        self.threads
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
            s.participants = participants;
            s.updated_at = updated_at;
        }
        Ok(())
    }

    pub async fn remove_thread_participants(
        &self,
        user: &Principal,
        _id: u64,
        participants: BTreeSet<Principal>,
    ) -> Result<(), BoxError> {
        if participants.is_empty() {
            return Err("Participants cannot be empty".to_string().into());
        }

        self.check_thread_state(_id)?;
        let mut thread: Thread = self.threads.get_as(_id).await?;
        if !thread.has_permission(user, ThreadPermission::Manage) {
            return Err(format!(
                "User {} does not have permission to manage thread {}",
                user, _id
            )
            .into());
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
        let participants = thread.participants.len() as u64;
        self.threads
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
            s.participants = participants;
            s.updated_at = updated_at;
        }
        Ok(())
    }

    pub async fn quit_thread(&self, user: &Principal, _id: u64) -> Result<(), BoxError> {
        {
            match self.thread_states.read().get(&_id) {
                Some(state) => {
                    if state.read().status == ThreadStatus::Unavailable {
                        return Err(format!("Thread {} is unavailable", _id).into());
                    }
                }
                None => return Err(format!("Thread {} not found", _id).into()),
            }
        }

        let mut thread: Thread = self.threads.get_as(_id).await?;
        if !thread.participants.contains_key(user) {
            return Err(format!("User {} is not a participant of thread {}", user, _id).into());
        }

        let updated_at = unix_ms();
        let mut changes: BTreeMap<String, Fv> =
            BTreeMap::from([("updated_at".to_string(), Fv::U64(updated_at))]);
        if thread.controllers.contains(user) {
            thread.controllers.remove(user);
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

        if thread.managers.contains(user) {
            thread.managers.remove(user);
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

        thread.participants.remove(user);
        let participants = thread.participants.len() as u64;
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
        self.threads.update(_id, changes).await?;
        if let Some(state) = self.thread_states.write().get_mut(&_id) {
            let mut s = state.write();
            s.participants = participants;
            s.updated_at = updated_at;
        }
        Ok(())
    }

    pub async fn delete_thread(&self, user: &Principal, _id: u64) -> Result<(), BoxError> {
        {
            match self.thread_states.read().get(&_id) {
                Some(state) => {
                    if state.read().status == ThreadStatus::Unavailable {
                        return Err(format!("Thread {} is unavailable", _id).into());
                    }
                }
                None => return Err(format!("Thread {} not found", _id).into()),
            }
        }

        let thread: Thread = self.threads.get_as(_id).await?;
        if !thread.has_permission(user, ThreadPermission::Control) {
            return Err(format!(
                "User {} does not have permission to control thread {}",
                user, _id
            )
            .into());
        }

        {
            self.thread_states.write().remove(&_id);
        }

        self.threads.remove(_id).await?;
        self.db
            .delete_collection(Self::thread_resource_collection_name(_id).as_str())
            .await?;
        self.db
            .delete_collection(Self::thread_message_collection_name(_id).as_str())
            .await?;

        Ok(())
    }

    pub async fn sys_set_thread_status(
        &self,
        _id: u64,
        status: ThreadStatus,
    ) -> Result<(), BoxError> {
        let updated_at = unix_ms();
        self.threads
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
            s.status = status;
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
        self.threads
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
            s.max_participants = max_participants;
            s.updated_at = updated_at;
        }
        Ok(())
    }

    fn check_thread_state(&self, thread_id: u64) -> Result<ThreadVisibility, BoxError> {
        match self.thread_states.read().get(&thread_id) {
            Some(state) => {
                let s = state.read();
                if s.status != ThreadStatus::Active {
                    return Err(format!("Thread {} is not active", thread_id).into());
                }
                Ok(s.visibility)
            }
            None => Err(format!("Thread {} not found", thread_id).into()),
        }
    }

    async fn my_thread_ids(&self, user: &Principal) -> Vec<u64> {
        self.threads
            .search_ids(Query {
                filter: Some(Filter::Field((
                    "participants".to_string(),
                    RangeQuery::Eq(Fv::Bytes(user.as_slice().to_vec())),
                ))),
                ..Default::default()
            })
            .await
            .unwrap_or_default()
    }
}

impl NexusNode {
    pub async fn add_message(
        &self,
        user: &Principal,
        thread_id: u64,
        mut message: Message,
    ) -> Result<Message, BoxError> {
        self.check_thread_state(thread_id)?;
        let ids = self.my_thread_ids(user).await;
        if !ids.contains(&thread_id) {
            return Err(
                format!("User {} is not a participant of thread {}", user, thread_id).into(),
            );
        }

        let timestamp = unix_ms();
        message._id = 0;
        message.user = Some(*user);
        message.timestamp = timestamp;

        let collection = self.get_message_collection(thread_id).await?;
        let _id = collection.add_from(&message).await?;
        message._id = _id;

        if let Some(state) = self.thread_states.write().get_mut(&thread_id) {
            let mut s = state.write();
            s.latest_message_by = message.user;
            s.latest_message_id = message._id;
            s.latest_message_at = timestamp;
        }

        Ok(message)
    }

    pub async fn get_message(
        &self,
        user: &Principal,
        thread_id: u64,
        message_id: u64,
    ) -> Result<Message, BoxError> {
        let v = self.check_thread_state(thread_id)?;
        let ids = self.my_thread_ids(user).await;
        if !ids.contains(&thread_id) && v != ThreadVisibility::Public {
            return Err(
                format!("User {} is not a participant of thread {}", user, thread_id).into(),
            );
        }

        let collection = self.get_message_collection(thread_id).await?;
        let message: Message = collection.get_as(message_id).await?;

        Ok(message)
    }

    pub async fn list_messages(
        &self,
        user: &Principal,
        thread_id: u64,
        cursor: Option<String>,
        limit: Option<usize>,
    ) -> Result<(Vec<Message>, Option<String>), BoxError> {
        let v = self.check_thread_state(thread_id)?;
        let ids = self.my_thread_ids(user).await;
        if !ids.contains(&thread_id) && v != ThreadVisibility::Public {
            return Err(
                format!("User {} is not a participant of thread {}", user, thread_id).into(),
            );
        }

        let limit = limit.unwrap_or(100).min(1000);
        let cursor = (BTree::from_cursor::<u64>(&cursor)?).unwrap_or_default();

        let collection = self.get_message_collection(thread_id).await?;
        let mut message_ids = collection.ids();
        if cursor > 0 {
            message_ids.retain(|id| *id < cursor);
        }

        if message_ids.len() > limit {
            message_ids.drain(0..message_ids.len() - limit);
        }

        let mut messages = Vec::with_capacity(message_ids.len());
        for id in message_ids {
            if let Ok(message) = collection.get_as::<Message>(id).await {
                messages.push(message);
            }
        }
        let cursor = if messages.len() >= limit {
            BTree::to_cursor(&messages.first().unwrap()._id)
        } else {
            None
        };

        Ok((messages, cursor))
    }
}
