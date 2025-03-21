use anda_core::{
    ANONYMOUS, BaseContext, BoxError, CacheStoreFeatures, MyThreads, RequestMeta, ThreadMeta,
    ToolInput, UpdateVersion, Xid,
};
use candid::Principal;
use serde_json::json;
use std::collections::BTreeSet;
use structured_logger::unix_ms;

use crate::context::BaseCtx;

mod state;
mod thread;

pub use state::*;
pub use thread::*;

pub static SYSTEM_PATH: &str = "_";

#[derive(Clone)]
/// Represents system management tools for the Anda engine.
pub struct Management {
    ctx: BaseCtx,
    controller: Principal,
    managers: BTreeSet<Principal>,
    visibility: Visibility, // 0: private, 1: protected, 2: public
}

/// The visibility of the engine.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    /// private, can only be accessed by the controller and managers;
    Private = 0,

    /// protected, can be accessed by the controller, managers, and users who have permission;
    Protected = 1,

    /// public, can be accessed by anyone.
    Public = 2,
}

#[derive(Clone)]
/// Builder for creating a new management instance.
pub struct ManagementBuilder {
    /// The visibility of the engine.
    pub(crate) visibility: Visibility,

    /// The controller of the engine.
    pub(crate) controller: Principal,

    /// The managers of the engine.
    pub(crate) managers: BTreeSet<Principal>,
}

impl ManagementBuilder {
    pub fn new(visibility: Visibility, controller: Principal) -> Self {
        Self {
            controller,
            managers: BTreeSet::new(),
            visibility,
        }
    }

    /// Sets the managers for the engine.
    pub fn with_managers(mut self, managers: BTreeSet<Principal>) -> Self {
        self.managers = managers;
        self
    }

    pub fn build(self, ctx: &BaseCtx) -> Management {
        Management {
            ctx: ctx
                .child_with(
                    ctx.id,
                    SYSTEM_PATH.to_string(),
                    RequestMeta {
                        engine: None,
                        thread: None,
                        user: Some(ctx.name.clone()),
                    },
                )
                .expect("failed to create system context"),
            controller: self.controller,
            managers: self.managers,
            visibility: self.visibility,
        }
    }
}

impl Management {
    fn user_state_path(user_id: &Principal) -> String {
        format!("US_{}.cbor", user_id.to_text())
    }

    fn thread_meta_path(thread_id: &Xid) -> String {
        format!("TH_{}.meta.cbor", thread_id.xid())
    }

    fn my_threads_path(id: &Principal) -> String {
        format!("MYTH_{}.cbor", id.to_text())
    }

    /// Returns true if the caller is the controller of the engine.
    pub fn is_controller(&self, caller: &Principal) -> bool {
        caller == &self.controller
    }

    /// Returns true if the caller is the controller or a manager of the engine.
    pub fn is_manager(&self, caller: &Principal) -> bool {
        caller == &self.controller || self.managers.contains(caller)
    }

    pub fn try_get_visibility(&self, caller: &Principal) -> Result<Visibility, BoxError> {
        if self.visibility != Visibility::Public && caller == &ANONYMOUS {
            return Err("anonymous caller not allowed".into());
        }

        if self.visibility == Visibility::Private && !self.is_manager(caller) {
            return Err("caller is not allowed".into());
        }

        Ok(self.visibility)
    }

    /// Retrieves the user state from the cache store.
    /// It does not check the permission of the caller for the thread.
    pub async fn get_user_state(&self, user: &Principal) -> Result<UserState, BoxError> {
        let state_key = Self::user_state_path(user);
        let (mut val, version) = self.ctx.cache_store_get::<UserState>(&state_key).await?;
        // the version in the user state is prev version, we need to update it here
        val.version = Some(version);
        Ok(val)
    }

    /// Loads the user state from the cache store. If the user state does not exist, a new user state will be created.
    pub(crate) async fn load_user_state(
        &self,
        user: &Principal,
    ) -> Result<UserStateWrapper, BoxError> {
        match self.get_user_state(user).await {
            Ok(state) => Ok(UserStateWrapper { state }),
            Err(_) => Ok(UserStateWrapper::new(*user)),
        }
    }

    /// Saves the user state to the cache store.
    pub(crate) async fn save_user_state(
        &self,
        state: UserState,
    ) -> Result<UpdateVersion, BoxError> {
        let state_key = Self::user_state_path(&state.user);
        let ver = state.version.clone();
        self.ctx.cache_store_set(&state_key, state, ver).await
    }

    /// Deletes the user state from the cache store.
    pub(crate) async fn delete_user_state(&self, user: &Principal) -> Result<(), BoxError> {
        let state_key = Self::user_state_path(user);
        self.ctx.cache_store_delete(&state_key).await
    }

    /// Retrieves the thread metadata from the cache store.
    /// It does not check the permission of the caller for the thread.
    pub async fn get_thread_meta(&self, thread_id: &Xid) -> Result<ThreadMeta, BoxError> {
        let thread_key = Self::thread_meta_path(thread_id);
        let (mut meta, version) = self.ctx.cache_store_get::<ThreadMeta>(&thread_key).await?;
        // the version in the metadata is prev version, we need to update it here
        meta.version = Some(version);
        Ok(meta)
    }

    /// Loads the thread metadata from the cache store or remote engine.
    /// If the thread does not exist, a new thread will be created.
    pub(crate) async fn load_thread_meta(
        &self,
        caller: &Principal,
        thread_id: &Option<Xid>,
    ) -> Result<ThreadMeta, BoxError> {
        match thread_id {
            // Create a new thread if the thread_id is not provided.
            None => Ok(ThreadMeta::new(Xid::new(), self.ctx.id, *caller, unix_ms())),
            Some(id) => {
                match self.get_thread_meta(id).await {
                    Ok(thread) => {
                        // Check if the caller has permission to access the thread.
                        if thread.has_permission(caller) {
                            Ok(thread)
                        } else {
                            Err(format!(
                                "caller {} does not have permission to access the thread {}",
                                caller.to_text(),
                                id
                            )
                            .into())
                        }
                    }
                    Err(_) => {
                        let threads = self.load_my_threads().await?;
                        if let Some(agent) = threads.get_agent_by(id) {
                            let endpoint =
                                self.ctx.remote.get_endpoint_by_id(agent).ok_or_else(|| {
                                    format!(
                                        "failed to get the engine endpoint: {}",
                                        agent.to_text()
                                    )
                                })?;
                            // Call the remote agent engine to get the thread metadata.
                            let output = self
                                .ctx
                                .remote_tool_call(
                                    &endpoint,
                                    ToolInput::new(
                                        ThreadMetaTool::NAME.to_string(),
                                        json!(ThreadMetaToolArgs {
                                            method: ThreadMetaToolMethod::GetThreadMeta,
                                            thread_id: id.to_string(),
                                            user_id: None,
                                        }),
                                    ),
                                )
                                .await?;
                            let thread: ThreadMeta = serde_json::from_value(output.output)?;
                            return Ok(thread);
                        }

                        // Create a new thread with parent if the thread does not exist.
                        let mut thread =
                            ThreadMeta::new(Xid::new(), self.ctx.id, *caller, unix_ms());
                        thread.parent = Some(id.to_owned());
                        Ok(thread)
                    }
                }
            }
        }
    }

    /// Saves the thread metadata to the cache store.
    pub(crate) async fn save_thread_meta(
        &self,
        mut thread: ThreadMeta,
    ) -> Result<UpdateVersion, BoxError> {
        let thread_key = Self::thread_meta_path(&thread.id);
        let ver = thread.version.clone();
        thread.updated_at = unix_ms();
        self.ctx.cache_store_set(&thread_key, thread, ver).await
    }

    /// Deletes the thread metadata from the cache store.
    pub(crate) async fn delete_thread_meta(
        &self,
        caller: &Principal,
        thread_id: &Xid,
    ) -> Result<(), BoxError> {
        match self.get_thread_meta(thread_id).await {
            Ok(thread) => {
                if thread.has_permission(caller) {
                    self.ctx
                        .cache_store_delete(&Self::thread_meta_path(&thread.id))
                        .await
                } else {
                    Err(format!(
                        "caller {} does not have permission to delete the thread {}",
                        caller.to_text(),
                        thread_id
                    )
                    .into())
                }
            }
            Err(_) => Ok(()),
        }
    }

    /// Loads my threads index that participating in.
    pub(crate) async fn load_my_threads(&self) -> Result<MyThreads, BoxError> {
        let my_threads_key = Self::my_threads_path(&self.ctx.id);
        match self.ctx.cache_store_get::<MyThreads>(&my_threads_key).await {
            Ok((mut threads, ver)) => {
                threads.version = Some(ver);
                Ok(threads)
            }
            Err(_) => Ok(MyThreads::new(self.ctx.id)),
        }
    }

    /// Saves my threads index that participating in.
    pub(crate) async fn save_my_threads(
        &self,
        threads: MyThreads,
    ) -> Result<UpdateVersion, BoxError> {
        let my_threads_key = Self::my_threads_path(&self.ctx.id);
        let ver = threads.version.clone();
        self.ctx
            .cache_store_set(&my_threads_key, threads, ver)
            .await
    }
}
