use anda_core::{BoxError, CacheStoreFeatures, RequestMeta, ThreadId, ThreadMeta};
use candid::Principal;
use std::collections::BTreeSet;
use structured_logger::unix_ms;

use crate::context::BaseCtx;

pub static SYSTEM_PATH: &str = "_";

/// Represents a system management tool for the Anda engine.
pub struct Management {
    ctx: BaseCtx,
    controller: Principal,
    managers: BTreeSet<Principal>,
}

impl Management {
    pub(crate) fn new(ctx: &BaseCtx, controller: Principal) -> Self {
        Self {
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
            controller,
            managers: BTreeSet::new(),
        }
    }

    fn thread_meta_path(thread_id: &ThreadId) -> String {
        format!("TH_{}.meta", thread_id.xid())
    }

    pub fn is_controller(&self, caller: &Principal) -> bool {
        caller == &self.controller
    }

    pub fn is_manager(&self, caller: &Principal) -> bool {
        caller == &self.controller || self.managers.contains(caller)
    }

    pub async fn get_thread_meta(&self, thread_id: &ThreadId) -> Result<ThreadMeta, BoxError> {
        let thread_key = Self::thread_meta_path(thread_id);
        self.ctx.cache_store_get::<ThreadMeta>(&thread_key).await
    }

    pub async fn load_thread_meta(
        &self,
        caller: Principal,
        thread_id: &Option<ThreadId>,
    ) -> Result<ThreadMeta, BoxError> {
        match thread_id {
            None => Ok(ThreadMeta::new(self.ctx.id, caller, unix_ms())),
            Some(id) => {
                let thread_key = Self::thread_meta_path(id);
                match self.ctx.cache_store_get::<ThreadMeta>(&thread_key).await {
                    Ok(thread) => {
                        // Check if the caller has permission to access the thread.
                        if thread.has_permission(&caller) {
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
                        // Create a new thread with parent if the thread does not exist.
                        let mut thread = ThreadMeta::new(self.ctx.id, caller, unix_ms());
                        thread.parent = Some(id.to_owned());
                        Ok(thread)
                    }
                }
            }
        }
    }

    pub async fn save_thread_meta(&self, mut thread: ThreadMeta) -> Result<(), BoxError> {
        let thread_key = Self::thread_meta_path(&thread.id);
        thread.updated_at = unix_ms();
        self.ctx.cache_store_set_and_wait(&thread_key, thread).await
    }
}
