//! Library entrypoint for the `classifier` crate.
//!
//! This file exposes the internal modules as public library modules so unit tests
//! and other consumers can reference components like `nb::train_naive_bayes`,
//! `text::tokenize`, and `data::LabeledRow` via `crate::...`.

//! The binary (`src/main.rs`) continues to exist and may shadow its own module
//! declarations; exposing these as `pub mod` here ensures the library crate API
//! is available for tests and integration.

pub mod artifact;
pub mod cli;
pub mod commands;
pub mod context;
pub mod data;
pub mod metrics;
pub mod model;
pub mod nb;
pub mod text;

pub use data::{LabeledRow, read_labeled_csv};
/// Re-exports of commonly used types/functions for convenience in tests and consumers.
pub use model::ModelArtifact;
pub use nb::{predict_with_probs, train_naive_bayes};
pub use text::{normalize_text, tokenize};

/// Library-level version constant derived from Cargo package metadata.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests;
