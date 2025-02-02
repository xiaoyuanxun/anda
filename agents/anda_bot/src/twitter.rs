use agent_twitter_client::{models::Tweet, scraper::Scraper, search::SearchMode};
use anda_core::{
    Agent, BoxError, CacheFeatures, CompletionFeatures, Path, PutMode, StateFeatures, StoreFeatures,
};
use anda_engine::{
    context::AgentCtx, engine::Engine, extension::character::CharacterAgent, rand_number,
};
use anda_lancedb::knowledge::KnowledgeStore;
use ciborium::from_reader;
use ic_cose_types::to_cbor_bytes;
use std::sync::Arc;
use tokio::{
    sync::RwLock,
    time::{sleep, Duration},
};
use tokio_util::sync::CancellationToken;

use crate::handler::ServiceStatus;

const MAX_HISTORY_TWEETS: i64 = 21;
const MAX_SEEN_TWEET_IDS: usize = 10000;

static LOG_TARGET: &str = "twitter";

pub struct TwitterDaemon {
    engine: Arc<Engine>,
    agent: Arc<CharacterAgent<KnowledgeStore>>,
    scraper: Scraper,
    status: Arc<RwLock<ServiceStatus>>,
}

impl TwitterDaemon {
    pub fn new(
        engine: Arc<Engine>,
        agent: Arc<CharacterAgent<KnowledgeStore>>,
        scraper: Scraper,
        status: Arc<RwLock<ServiceStatus>>,
    ) -> Self {
        Self {
            engine,
            agent,
            scraper,
            status,
        }
    }

    async fn init_seen_tweet_ids<F>(&self, ctx: &F) -> usize
    where
        F: CacheFeatures + StoreFeatures,
    {
        // load seen_tweet_ids from store
        let seen_tweet_ids: Vec<String> = ctx
            .store_get(&Path::from("seen_tweet_ids"))
            .await
            .map(|(v, _)| from_reader(&v[..]).unwrap_or_default())
            .unwrap_or_default();
        let count = seen_tweet_ids.len();
        ctx.cache_set("seen_tweet_ids", (seen_tweet_ids, None))
            .await;
        count
    }

    async fn get_seen_tweet_ids<F>(&self, ctx: &F) -> Vec<String>
    where
        F: CacheFeatures + StoreFeatures,
    {
        ctx.cache_get("seen_tweet_ids").await.unwrap_or_default()
    }

    async fn set_seen_tweet_ids<F>(&self, ctx: F, val: Vec<String>)
    where
        F: CacheFeatures + StoreFeatures + Send + Sync + 'static,
    {
        ctx.cache_set("seen_tweet_ids", (val.clone(), None)).await;
        tokio::spawn(async move {
            let _ = ctx
                .store_put(
                    &Path::from("seen_tweet_ids"),
                    PutMode::Overwrite,
                    to_cbor_bytes(&val).into(),
                )
                .await;
        });
    }

    pub async fn run(&self, cancel_token: CancellationToken) -> Result<(), BoxError> {
        {
            let ctx = self.engine.ctx_with(self.agent.as_ref(), None, None)?;
            // load seen_tweet_ids from store
            let count = self.init_seen_tweet_ids(&ctx).await;

            log::info!(target: LOG_TARGET, "starting Twitter bot with {} seen tweets", count);
        }

        loop {
            {
                let status = self.status.read().await;
                if *status == ServiceStatus::Stopped {
                    log::info!(target: LOG_TARGET, "Twitter task stopped");
                    tokio::select! {
                        _ = cancel_token.cancelled() => {
                            return Ok(());
                        },
                        _ = sleep(Duration::from_secs(60)) => {},
                    }
                    continue;
                }
                log::info!(target: LOG_TARGET, "run a Twitter task");
                // release read lock
            }

            match self
                .scraper
                .search_tweets(
                    &format!("@{}", self.agent.character.username.clone()),
                    20,
                    SearchMode::Latest,
                    None,
                )
                .await
            {
                Ok(mentions) => {
                    log::info!(target: LOG_TARGET, "fetch mentions: {} tweets", mentions.tweets.len());
                    for tweet in mentions.tweets {
                        if let Err(err) = self.handle_mention(tweet).await {
                            log::error!(target: LOG_TARGET, "handle mention error: {err:?}");
                        }

                        tokio::select! {
                            _ = cancel_token.cancelled() => {
                                return Ok(());
                            },
                            _ = sleep(Duration::from_secs(rand_number(3..=10))) => {},
                        }
                    }
                }
                Err(err) => {
                    log::error!(target: LOG_TARGET, "fetch mentions error: {err:?}");
                }
            }

            match rand_number(0..=10) {
                0 => {
                    if let Err(err) = self.handle_home_timeline().await {
                        log::error!(target: LOG_TARGET, "handle_home_timeline error: {err:?}");
                    }
                }
                n => {
                    log::info!(target: LOG_TARGET, "skip home timeline task by random {n}");
                }
            }

            match rand_number(0..=20) {
                0 => {
                    if let Err(err) = self.post_new_tweet().await {
                        log::error!(target: LOG_TARGET, "post_new_tweet error: {err:?}");
                    }
                }
                n => {
                    log::info!(target: LOG_TARGET, "skip post new tweet task by random {n}");
                }
            }

            // Sleep between tasks
            tokio::select! {
                _ = cancel_token.cancelled() => {
                    return Ok(());
                },
                _ = sleep(Duration::from_secs(rand_number(60..=5 * 60))) => {},
            }
        }
    }

    async fn post_new_tweet(&self) -> Result<(), BoxError> {
        let knowledges = self.agent.latest_knowledge(60 * 30, 3, None).await?;
        if knowledges.is_empty() {
            return Ok(());
        }

        log::info!(target: LOG_TARGET, "post new tweet with {} knowledges", knowledges.len());
        let ctx = self.engine.ctx_with(
            self.agent.as_ref(),
            Some(self.agent.character.username.clone()),
            None,
        )?;
        let req = self
            .agent
            .character
            .to_request(
                "\
                Share a single brief thought or observation in one short sentence.\
                Be direct and concise. No questions, hashtags, or emojis.\
                "
                .to_string(),
                ctx.user(),
            )
            .append_documents(knowledges.into());
        let res = ctx.completion(req).await?;
        match res.failed_reason {
            Some(reason) => Err(format!("Failed to generate response for tweet: {reason}").into()),
            None => {
                let _ = self.scraper.send_tweet(&res.content, None, None).await?;
                log::info!(target: LOG_TARGET,
                    time_elapsed = ctx.time_elapsed().as_millis() as u64;
                    "post new tweet: {}",
                    res.content.chars().take(100).collect::<String>()
                );
                Ok(())
            }
        }
    }

    async fn handle_home_timeline(&self) -> Result<(), BoxError> {
        let ctx = self.engine.ctx_with(
            self.agent.as_ref(),
            Some(self.agent.character.username.clone()),
            None,
        )?;

        let mut seen_tweet_ids: Vec<String> = self.get_seen_tweet_ids(&ctx).await;
        if seen_tweet_ids.len() >= MAX_SEEN_TWEET_IDS {
            seen_tweet_ids.drain(0..MAX_SEEN_TWEET_IDS / 2);
        }
        let ids = if seen_tweet_ids.len() > 42 {
            seen_tweet_ids[(seen_tweet_ids.len() - 42)..].to_vec()
        } else {
            seen_tweet_ids.clone()
        };
        let tweets = self.scraper.get_home_timeline(1, ids).await?;
        log::info!(target: LOG_TARGET, "process home timeline, {} tweets", tweets.len());

        let mut likes = 0;
        let mut retweets = 0;
        let mut quotes = 0;
        for tweet in tweets {
            let tweet_user = tweet["core"]["user_results"]["result"]["legacy"]["screen_name"]
                .as_str()
                .unwrap_or_else(|| tweet["legacy"]["user_id_str"].as_str().unwrap_or_default())
                .to_string();
            let tweet_content = tweet["legacy"]["full_text"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            let tweet_id = tweet["legacy"]["id_str"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            if tweet_content.is_empty() || tweet_id.is_empty() {
                continue;
            }

            if tweet_user.to_lowercase() == self.agent.character.username.to_lowercase() {
                // not replying to bot itself
                continue;
            }

            if seen_tweet_ids.contains(&tweet_id) {
                continue;
            }
            seen_tweet_ids.push(tweet_id.clone());

            let res: Result<(), BoxError> = async {
                if self.handle_like(&ctx, &tweet_content, &tweet_id).await? {
                    likes += 1;
                    if self.handle_quote(&ctx, &tweet_content, &tweet_id).await? {
                        // TODO: save tweet to knowledge store
                        quotes += 1;
                    } else {
                        self.handle_retweet(&ctx, &tweet_content, &tweet_id).await?;
                        retweets += 1;
                    }
                }
                Ok(())
            }
            .await;

            if let Err(err) = res {
                log::error!(target: LOG_TARGET, "handle home timeline {tweet_id} error: {err:?}");
            }

            sleep(Duration::from_secs(rand_number(3..=10))).await;
        }

        self.set_seen_tweet_ids(ctx, seen_tweet_ids).await;
        log::info!(target: LOG_TARGET, "home timeline: likes {}, retweets {}, quotes {}", likes, retweets, quotes);
        Ok(())
    }

    async fn handle_mention(&self, tweet: Tweet) -> Result<(), BoxError> {
        let tweet_id = tweet.id.clone().unwrap_or_default();
        let tweet_text = tweet.text.clone().unwrap_or_default();
        let tweet_user = tweet.username.clone().unwrap_or_default();
        if tweet_text.is_empty() || tweet_user.is_empty() {
            return Ok(());
        }
        if tweet_user.to_lowercase() == self.agent.character.username.to_lowercase() {
            // not replying to bot itself
            return Ok(());
        }
        let ctx = self
            .engine
            .ctx_with(self.agent.as_ref(), Some(tweet_user.clone()), None)?;
        let mut seen_tweet_ids: Vec<String> = self.get_seen_tweet_ids(&ctx).await;

        if seen_tweet_ids.contains(&tweet_id) {
            return Ok(());
        }

        seen_tweet_ids.push(tweet_id.clone());

        let thread = self.build_conversation_thread(&tweet).await?;
        let messages: Vec<String> = thread
            .into_iter()
            .map(|t| {
                format!(
                    "{}: {:?}",
                    t.username.unwrap_or_default(),
                    t.text.unwrap_or_default()
                )
            })
            .collect();

        let tweet_text = if messages.len() <= 1 {
            tweet_text
        } else {
            messages.join("\n")
        };

        let res = self.agent.run(ctx.clone(), tweet_text, None).await?;
        if res.failed_reason.is_none() {
            // Reply to the original tweet
            let tweet: Option<&str> = tweet.id.as_deref();
            let _ = self.scraper.send_tweet(&res.content, tweet, None).await?;

            log::info!(target: LOG_TARGET,
                tweet_user = tweet_user,
                tweet_id = tweet_id,
                chars = res.content.chars().count(),
                time_elapsed = ctx.time_elapsed().as_millis() as u64;
                "handle mention");
        }

        self.set_seen_tweet_ids(ctx, seen_tweet_ids.clone()).await;

        Ok(())
    }

    async fn build_conversation_thread(&self, tweet: &Tweet) -> Result<Vec<Tweet>, BoxError> {
        let mut thread = Vec::new();
        let mut current_tweet = Some(tweet.clone());
        let mut depth = 0;

        while let Some(tweet) = current_tweet {
            if tweet.text.is_some() {
                thread.push(tweet.clone());
            }

            if depth >= MAX_HISTORY_TWEETS {
                break;
            }

            sleep(Duration::from_secs(rand_number(1..=3))).await;
            current_tweet = match tweet.in_reply_to_status_id {
                Some(parent_id) => match self.scraper.get_tweet(&parent_id).await {
                    Ok(parent_tweet) => Some(parent_tweet),
                    Err(_) => None,
                },
                None => None,
            };

            depth += 1;
        }

        thread.reverse();
        Ok(thread)
    }

    async fn handle_like(
        &self,
        ctx: &AgentCtx,
        tweet_content: &str,
        tweet_id: &str,
    ) -> Result<bool, BoxError> {
        if self.agent.should_like(ctx, tweet_content).await {
            let _ = self.scraper.like_tweet(tweet_id).await?;
            return Ok(true);
        }
        Ok(false)
    }

    async fn handle_retweet(
        &self,
        ctx: &AgentCtx,
        tweet_content: &str,
        tweet_id: &str,
    ) -> Result<bool, BoxError> {
        if self
            .agent
            .attention
            .should_retweet(ctx, tweet_content)
            .await
        {
            let req = self
                .agent
                .character
                .to_request(
                    "\
                    Reply the tweet with a single clear, natural sentence.\
                    "
                    .to_string(),
                    ctx.user(),
                )
                .context(
                    tweet_id.to_string(),
                    format!("Tweet content:\n{tweet_content}"),
                );
            let res = ctx.completion(req).await?;
            match res.failed_reason {
                Some(reason) => {
                    return Err(format!("Failed to generate response for tweet: {reason}").into());
                }
                None => {
                    let _ = self
                        .scraper
                        .send_tweet(&res.content, Some(tweet_id), None)
                        .await?;
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    async fn handle_quote(
        &self,
        ctx: &AgentCtx,
        tweet_content: &str,
        tweet_id: &str,
    ) -> Result<bool, BoxError> {
        if self.agent.attention.should_quote(ctx, tweet_content).await {
            let req = self
                .agent
                .character
                .to_request(
                    "\
                    Reply the tweet with a single clear, natural sentence.\
                    "
                    .to_string(),
                    ctx.user(),
                )
                .context(
                    tweet_id.to_string(),
                    format!("Tweet content:\n{tweet_content}"),
                );
            let res = ctx.completion(req).await?;
            match res.failed_reason {
                Some(reason) => {
                    return Err(format!("Failed to generate response for tweet: {reason}").into());
                }
                None => {
                    let _ = self
                        .scraper
                        .send_quote_tweet(&res.content, tweet_id, None)
                        .await?;
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "current_thread")]
    #[ignore]
    async fn test_x_api() {
        dotenv::dotenv().ok();

        let mut scraper = Scraper::new().await.unwrap();
        let cookie_string = std::env::var("TWITTER_COOKIES").expect("TWITTER_COOKIES is not set");

        scraper
            .set_from_cookie_string(&cookie_string)
            .await
            .unwrap();

        // scraper
        //     .login(
        //         std::env::var("TWITTER_USERNAME").unwrap(),
        //         std::env::var("TWITTER_PASSWORD").unwrap(),
        //         std::env::var("TWITTER_EMAIL").ok(),
        //         std::env::var("TWITTER_2FA_SECRET").ok(),
        //     )
        //     .await
        //     .unwrap();

        {
            let res = scraper
                .search_tweets(&format!("@{}", "ICPandaDAO"), 5, SearchMode::Latest, None)
                .await
                .unwrap();
            for tweet in res.tweets {
                // let data = serde_json::to_string_pretty(&tweet).unwrap();
                // println!("{}", data);
                let tweet_user = tweet.username.unwrap_or_default();
                let tweet_content = tweet.text.unwrap_or_default();
                let tweet_id = tweet.id.unwrap_or_default();
                println!("\n\n{}: {} - {}", tweet_user, tweet_id, tweet_content);

                println!("----------------------");
            }
        }

        {
            let tweets = scraper.get_home_timeline(1, Vec::new()).await.unwrap();
            for tweet in &tweets {
                let tweet_user = tweet["core"]["user_results"]["result"]["legacy"]["screen_name"]
                    .as_str()
                    .unwrap_or_else(|| tweet["legacy"]["user_id_str"].as_str().unwrap_or_default())
                    .to_string();
                let tweet_content = tweet["legacy"]["full_text"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                let tweet_id = tweet["legacy"]["id_str"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                println!("{}: {} - {}", tweet_user, tweet_id, tweet_content);
            }
        }
        // let tweets = serde_json::to_string_pretty(&tweets).unwrap();
        // std::fs::write("home_timeline_tweets.json", tweets).unwrap();
    }
}
