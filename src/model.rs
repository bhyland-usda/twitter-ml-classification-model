use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelArtifact {
    pub version: String,
    pub labels: Vec<String>,
    pub vocab: HashMap<String, usize>,   // token -> index
    pub class_log_prior: Vec<f64>,       // len = number_of_labels
    pub feature_log_prob: Vec<Vec<f64>>, // [label_index][token_index]
    pub alpha: f64,
    pub min_df: usize,
    pub max_features: usize,
    pub metadata: HashMap<String, String>,
}

pub fn validate_model(model_artifact: &ModelArtifact) -> Result<()> {
    if model_artifact.version.trim().is_empty() {
        bail!("invalid model: version is empty");
    }

    if model_artifact.labels.is_empty() {
        bail!("invalid model: labels are empty");
    }

    let mut seen_label_names = HashSet::with_capacity(model_artifact.labels.len());
    for label_name in &model_artifact.labels {
        let normalized_label_name = label_name.trim();
        if normalized_label_name.is_empty() {
            bail!("invalid model: found empty label value");
        }

        if !seen_label_names.insert(normalized_label_name.to_string()) {
            bail!(
                "invalid model: duplicate label '{}' found",
                normalized_label_name
            );
        }
    }

    if model_artifact.vocab.is_empty() {
        bail!("invalid model: vocabulary is empty");
    }

    if !model_artifact.alpha.is_finite() || model_artifact.alpha <= 0.0 {
        bail!("invalid model: alpha must be finite and > 0");
    }

    if model_artifact.min_df == 0 {
        bail!("invalid model: min_df must be >= 1");
    }

    if model_artifact.max_features == 0 {
        bail!("invalid model: max_features must be >= 1");
    }

    let uses_parent_text_value = model_artifact
        .metadata
        .get("uses_parent_text")
        .map(|value| value.trim())
        .unwrap_or("");
    if uses_parent_text_value != "true" {
        bail!(
            "invalid model: metadata.uses_parent_text must be 'true' (got '{}')",
            uses_parent_text_value
        );
    }

    let number_of_labels = model_artifact.labels.len();

    if model_artifact.class_log_prior.len() != number_of_labels {
        bail!(
            "invalid model: class_log_prior length mismatch (got {}, expected {})",
            model_artifact.class_log_prior.len(),
            number_of_labels
        );
    }

    if model_artifact.feature_log_prob.len() != number_of_labels {
        bail!(
            "invalid model: feature_log_prob label dimension mismatch (got {}, expected {})",
            model_artifact.feature_log_prob.len(),
            number_of_labels
        );
    }

    let vocabulary_size = model_artifact.vocab.len();

    let mut seen_vocabulary_indices = HashSet::with_capacity(vocabulary_size);
    for (token_text, token_index) in &model_artifact.vocab {
        if token_text.trim().is_empty() {
            bail!("invalid model: vocabulary contains empty token text");
        }

        if *token_index >= vocabulary_size {
            bail!(
                "invalid model: token '{}' has out-of-bounds index {} (vocabulary_size={})",
                token_text,
                token_index,
                vocabulary_size
            );
        }

        if !seen_vocabulary_indices.insert(*token_index) {
            bail!(
                "invalid model: duplicate vocabulary index {} detected",
                token_index
            );
        }
    }

    if seen_vocabulary_indices.len() != vocabulary_size {
        bail!(
            "invalid model: vocabulary index set is incomplete (unique_indices={}, vocabulary_size={})",
            seen_vocabulary_indices.len(),
            vocabulary_size
        );
    }

    for (label_index, class_prior_log_probability) in
        model_artifact.class_log_prior.iter().enumerate()
    {
        if !class_prior_log_probability.is_finite() {
            bail!(
                "invalid model: non-finite class_log_prior at label_index={}",
                label_index
            );
        }

        if *class_prior_log_probability > 0.0 {
            bail!(
                "invalid model: class_log_prior must be <= 0 at label_index={} (got {})",
                label_index,
                class_prior_log_probability
            );
        }
    }

    let sum_of_class_prior_probabilities: f64 = model_artifact
        .class_log_prior
        .iter()
        .map(|log_probability| log_probability.exp())
        .sum();

    if !sum_of_class_prior_probabilities.is_finite()
        || (sum_of_class_prior_probabilities - 1.0).abs() > 1e-6
    {
        bail!(
            "invalid model: class prior probabilities must sum to ~1.0 (got {})",
            sum_of_class_prior_probabilities
        );
    }

    for (label_index, token_log_probabilities_for_label) in
        model_artifact.feature_log_prob.iter().enumerate()
    {
        if token_log_probabilities_for_label.len() != vocabulary_size {
            bail!(
                "invalid model: token probability dimension mismatch at label_index={} (got {}, expected {})",
                label_index,
                token_log_probabilities_for_label.len(),
                vocabulary_size
            );
        }

        for (token_index, token_log_probability) in
            token_log_probabilities_for_label.iter().enumerate()
        {
            if !token_log_probability.is_finite() {
                bail!(
                    "invalid model: non-finite token log probability at label_index={}, token_index={}",
                    label_index,
                    token_index
                );
            }

            if *token_log_probability > 0.0 {
                bail!(
                    "invalid model: token log probability must be <= 0 at label_index={}, token_index={} (got {})",
                    label_index,
                    token_index,
                    token_log_probability
                );
            }
        }

        let sum_of_token_probabilities_for_label: f64 = token_log_probabilities_for_label
            .iter()
            .map(|log_probability| log_probability.exp())
            .sum();

        if !sum_of_token_probabilities_for_label.is_finite()
            || (sum_of_token_probabilities_for_label - 1.0).abs() > 1e-3
        {
            bail!(
                "invalid model: token probabilities must sum to ~1.0 at label_index={} (got {})",
                label_index,
                sum_of_token_probabilities_for_label
            );
        }
    }

    Ok(())
}
