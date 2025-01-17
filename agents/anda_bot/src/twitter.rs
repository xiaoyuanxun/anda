use agent_twitter_client::{models::Tweet, scraper::Scraper, search::SearchMode};
use anda_core::{Agent, BoxError, CacheExpiry, CacheFeatures, CompletionFeatures, StateFeatures};
use anda_engine::{
    context::AgentCtx, engine::Engine, extension::character::CharacterAgent, rand_number,
};
use anda_lancedb::knowledge::KnowledgeStore;
use log::{debug, error, info};
use std::sync::Arc;
use tokio::{
    sync::RwLock,
    time::{sleep, Duration},
};
use tokio_util::sync::CancellationToken;

use crate::handler::ServiceStatus;

const MAX_TWEET_LENGTH: usize = 280;
const MAX_HISTORY_TWEETS: i64 = 21;

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

    pub async fn run(&self, cancel_token: CancellationToken) -> Result<(), BoxError> {
        info!(target: LOG_TARGET, "starting Twitter bot");

        loop {
            sleep(Duration::from_secs(60)).await;

            {
                let status = self.status.read().await;
                if *status == ServiceStatus::Stopped {
                    continue;
                }
                // release read lock
            }

            match rand_number(0..=2) {
                0 => {
                    if let Err(err) = self.post_new_tweet().await {
                        error!(target: LOG_TARGET, "post_new_tweet error: {err:?}");
                    }
                }
                1 => {
                    if let Err(err) = self.handle_home_timeline().await {
                        error!(target: LOG_TARGET, "handle_home_timeline error: {err:?}");
                    }
                }
                2 => {
                    match self
                        .scraper
                        .search_tweets(
                            &format!("@{}", self.agent.character.username.clone()),
                            5,
                            SearchMode::Latest,
                            None,
                        )
                        .await
                    {
                        Ok(mentions) => {
                            for tweet in mentions.tweets {
                                if let Err(err) = self.handle_mention(tweet).await {
                                    error!(target: LOG_TARGET, "handle mention error: {err:?}");
                                }

                                sleep(Duration::from_secs(rand_number(60..=180))).await;
                            }
                        }
                        Err(err) => {
                            error!(target: LOG_TARGET, "fetch mentions error: {err:?}");
                        }
                    }
                }
                _ => unreachable!(),
            }

            if cancel_token.is_cancelled() {
                break;
            }

            // Sleep between tasks
            sleep(Duration::from_secs(rand_number(30 * 60..=60 * 60))).await;
        }

        Ok(())
    }

    async fn post_new_tweet(&self) -> Result<(), BoxError> {
        let knowledges = self.agent.latest_knowledge(60 * 30, 3, None).await?;
        if knowledges.is_empty() {
            return Ok(());
        }
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
                Keep responses concise and under 280 characters.\n\
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
                Ok(())
            }
        }
    }

    async fn handle_home_timeline(&self) -> Result<(), BoxError> {
        let tweets = self.scraper.get_home_timeline(1, Vec::new()).await?;
        let ctx = self.engine.ctx_with(
            self.agent.as_ref(),
            Some(self.agent.character.username.clone()),
            None,
        )?;
        debug!(target: LOG_TARGET, "process home timeline, {} tweets", tweets.len());

        for tweet in tweets {
            let tweet_user = tweet["legacy"]["user_id_str"]
                .as_str()
                .unwrap_or_default()
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
                return Ok(());
            }

            let handle_key = format!("D_{}", tweet_id);
            if ctx.cache_contains(&handle_key) {
                continue;
            }
            ctx.cache_set(
                &handle_key,
                (
                    true,
                    Some(CacheExpiry::TTL(Duration::from_secs(3600 * 24 * 7))),
                ),
            )
            .await;

            if self.handle_like(&ctx, &tweet_content, &tweet_id).await? {
                if self.handle_quote(&ctx, &tweet_content, &tweet_id).await? {
                    // TODO: save tweet to knowledge store
                } else {
                    self.handle_retweet(&ctx, &tweet_content, &tweet_id).await?;
                }
            }

            sleep(Duration::from_secs(rand_number(60..=180))).await;
        }
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
            .ctx_with(self.agent.as_ref(), Some(tweet_user), None)?;

        let handle_key = format!("D_{}", tweet_id);
        if ctx.cache_contains(&handle_key) {
            return Ok(());
        }
        ctx.cache_set(
            &handle_key,
            (
                true,
                Some(CacheExpiry::TTL(Duration::from_secs(3600 * 24 * 7))),
            ),
        )
        .await;

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

        let res = self.agent.run(ctx, tweet_text, None).await?;

        if res.failed_reason.is_some() {
            return Ok(());
        }

        // Split response into tweet-sized chunks if necessary
        let chunks: Vec<String> = res
            .content
            .chars()
            .collect::<Vec<char>>()
            .chunks(MAX_TWEET_LENGTH)
            .map(|chunk| chunk.iter().collect::<String>())
            .collect();

        // Reply to the original tweet
        let tweet_id: Option<&str> = tweet.id.as_deref();
        for chunk in &chunks {
            let _ = self
                .scraper
                .send_tweet(chunk.as_str(), tweet_id, None)
                .await?;
            sleep(Duration::from_secs(rand_number(1..=10))).await;
        }

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
        if self.agent.attention.should_like(ctx, tweet_content).await {
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
            let _ = self.scraper.retweet(tweet_id).await;
            return Ok(true);
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
                    Reply with a single clear, natural sentence.\
                    If the tweet contains ASCII art or stylized text formatting, respond with similar creative formatting.\n\
                    Keep responses concise and under 280 characters.\
                    "
                    .to_string(),
                    ctx.user(),
                )
                .context(tweet_id.to_string(), format!("Quote tweet content:\n{tweet_content}"));
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
