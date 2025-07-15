//! Extension module providing core AI agent capabilities
//!
//! This module contains essential extensions that enhance AI agent functionality,
//! including attention management, character definition, content extraction,
//! and document segmentation.
//!
//! # Key Components
//!
//! - **Attention Management**: Controls how agents focus on and respond to content
//! - **Character System**: Defines agent personalities and communication styles
//! - **Extraction Tools**: Enables structured data extraction from unstructured text
//! - **Google Web Search Tool**: Enables web searches and retrieve results.
//! - **Document Segmentation**: Breaks down large documents into manageable chunks
//!
//! # Usage
//!
//! These extensions are typically used together to create fully-featured AI agents:
//! 1. Define agent personality using the Character system
//! 2. Configure attention management for response behavior
//! 3. Use extraction tools for structured data processing
//! 4. Apply document segmentation for large content processing
//!

pub mod attention;
pub mod character;
pub mod extractor;
pub mod google;
pub mod segmenter;
