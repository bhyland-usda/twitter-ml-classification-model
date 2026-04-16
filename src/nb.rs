use crate::{
    context::build_combined_text,
    data::LabeledRow,
    model::ModelArtifact,
    text::{normalize_text, tokenize},
};
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::{Arc, Mutex},
};

pub fn train_naive_bayes(
    training_rows: &[LabeledRow],
    min_df: usize,
    max_features: usize,
    alpha: f64,
) -> Result<ModelArtifact> {
    if training_rows.is_empty() {
        bail!("training rows are empty");
    }
    if min_df == 0 {
        bail!("min_df must be >= 1");
    }
    if max_features == 0 {
        bail!("max_features must be >= 1");
    }
    if !alpha.is_finite() || alpha <= 0.0 {
        bail!("alpha must be finite and > 0");
    }

    // Collect unique labels deterministically (lexical)
    let unique_labels: Vec<String> = {
        let mut unique_label_set = BTreeSet::new();
        for training_row in training_rows {
            let normalized_label = training_row.label.trim();
            if normalized_label.is_empty() {
                bail!("training row has empty label");
            }
            unique_label_set.insert(normalized_label.to_string());
        }
        unique_label_set.into_iter().collect()
    };

    if unique_labels.len() < 2 {
        bail!("training requires at least 2 unique labels");
    }

    let label_to_index: HashMap<String, usize> = unique_labels
        .iter()
        .enumerate()
        .map(|(label_index, label_name)| (label_name.clone(), label_index))
        .collect();

    // Tokenize texts in parallel for speed.
    let tokenized_training_texts: Vec<Vec<String>> = training_rows
        .par_iter()
        .map(|training_row| {
            let combined_text = build_combined_text(&training_row.parent_text, &training_row.text);
            tokenize(&normalize_text(&combined_text))
        })
        .collect();

    // Build vocabulary (uses deterministic ordering after counting).
    let vocabulary_index_by_token = build_vocab(&tokenized_training_texts, min_df, max_features);
    if vocabulary_index_by_token.is_empty() {
        bail!("empty vocabulary: lower --min-df or inspect input text");
    }

    let number_of_labels = unique_labels.len();
    let vocabulary_size = vocabulary_index_by_token.len();
    let number_of_training_rows = training_rows.len();

    // Use shared accumulators protected by a Mutex to allow parallel updates.
    #[derive(Default)]
    struct Accumulators {
        training_rows_per_label: Vec<usize>,
        total_token_count_per_label: Vec<f64>,
        token_count_per_label_and_token: Vec<Vec<f64>>,
    }

    let accum = Accumulators {
        training_rows_per_label: vec![0usize; number_of_labels],
        total_token_count_per_label: vec![0f64; number_of_labels],
        token_count_per_label_and_token: vec![vec![0f64; vocabulary_size]; number_of_labels],
    };

    let accum = Arc::new(Mutex::new(accum));

    // Parallel iterate over rows, vectorize and update accumulators.
    training_rows.par_iter().for_each(|training_row| {
        // Local handling: if label lookup fails, we cannot easily return Result from for_each.
        // So we catch panics by bubbling through unwraps only for programming errors.
        let label_index_opt = label_to_index.get(&training_row.label).copied();
        if label_index_opt.is_none() {
            // Unknown label in training rows should be considered a programming error
            // (should have been validated earlier). We bail by panicking inside parallel
            // iterator so the outer function returns an error via join.
            panic!("unknown label in training rows: {}", training_row.label);
        }
        let label_index = label_index_opt.unwrap();

        // Vectorize counts for this row
        let combined_text = build_combined_text(&training_row.parent_text, &training_row.text);
        let token_counts = vectorize_counts(&combined_text, &vocabulary_index_by_token);

        // Update shared accumulators
        let mut guard = accum.lock().unwrap();
        guard.training_rows_per_label[label_index] += 1;
        for (token_index, token_count) in token_counts {
            if token_index >= vocabulary_size {
                panic!(
                    "internal error: token index out of bounds (token_index={}, vocabulary_size={})",
                    token_index, vocabulary_size
                );
            }
            let token_count_as_f64 = token_count as f64;
            guard.token_count_per_label_and_token[label_index][token_index] += token_count_as_f64;
            guard.total_token_count_per_label[label_index] += token_count_as_f64;
        }
        // guard dropped here
    });

    // Extract accumulators
    let guard = Arc::try_unwrap(accum)
        .map_err(|_| anyhow::anyhow!("internal error unwrapping accumulators"))?
        .into_inner()
        .map_err(|_| anyhow::anyhow!("internal error unwrapping accumulators"))?;

    let training_rows_per_label = guard.training_rows_per_label;
    let total_token_count_per_label = guard.total_token_count_per_label;
    let token_count_per_label_and_token = guard.token_count_per_label_and_token;

    // Compute priors and feature log probabilities
    let mut class_log_prior = vec![0f64; number_of_labels];
    let mut feature_log_prob = vec![vec![0f64; vocabulary_size]; number_of_labels];

    for label_index in 0..number_of_labels {
        let smoothed_label_row_count = training_rows_per_label[label_index] as f64 + alpha;
        let smoothed_total_row_count =
            number_of_training_rows as f64 + alpha * number_of_labels as f64;

        if !smoothed_label_row_count.is_finite()
            || !smoothed_total_row_count.is_finite()
            || smoothed_total_row_count <= 0.0
        {
            bail!("invalid class prior normalization state");
        }

        let label_prior_probability = smoothed_label_row_count / smoothed_total_row_count;
        if !label_prior_probability.is_finite() || label_prior_probability <= 0.0 {
            bail!(
                "invalid computed class prior at label_index={}",
                label_index
            );
        }
        class_log_prior[label_index] = label_prior_probability.ln();

        let smoothed_total_token_count_for_label =
            total_token_count_per_label[label_index] + alpha * vocabulary_size as f64;

        if !smoothed_total_token_count_for_label.is_finite()
            || smoothed_total_token_count_for_label <= 0.0
        {
            bail!(
                "invalid token likelihood denominator at label_index={}",
                label_index
            );
        }

        for token_index in 0..vocabulary_size {
            let token_probability_given_label =
                (token_count_per_label_and_token[label_index][token_index] + alpha)
                    / smoothed_total_token_count_for_label;

            if !token_probability_given_label.is_finite() || token_probability_given_label <= 0.0 {
                bail!(
                    "invalid token probability at label_index={}, token_index={}",
                    label_index,
                    token_index
                );
            }

            feature_log_prob[label_index][token_index] = token_probability_given_label.ln();
        }
    }

    Ok(ModelArtifact {
        version: "nb_text_classifier_v1".to_string(),
        labels: unique_labels,
        vocab: vocabulary_index_by_token,
        class_log_prior,
        feature_log_prob,
        alpha,
        min_df,
        max_features,
        metadata: HashMap::from([("uses_parent_text".to_string(), "true".to_string())]),
    })
}

pub fn predict_with_context(
    model: &ModelArtifact,
    parent_text: &str,
    comment_text: &str,
) -> Result<(String, Vec<f64>)> {
    if parent_text.trim().is_empty() {
        bail!("parent_text is required for prediction");
    }

    let combined_text = build_combined_text(parent_text, comment_text);
    predict_with_probs(model, &combined_text)
}

pub fn predict_with_probs(model: &ModelArtifact, input_text: &str) -> Result<(String, Vec<f64>)> {
    validate_runtime_model(model)?;

    let token_count_by_index = vectorize_counts(input_text, &model.vocab);
    let number_of_labels = model.labels.len();

    let mut unnormalized_log_score_by_label = vec![0f64; number_of_labels];
    for label_index in 0..number_of_labels {
        let mut label_log_score = model.class_log_prior[label_index];
        for (token_index, token_count) in &token_count_by_index {
            label_log_score +=
                *token_count as f64 * model.feature_log_prob[label_index][*token_index];
        }

        if !label_log_score.is_finite() {
            bail!("non-finite class score at label_index={}", label_index);
        }

        unnormalized_log_score_by_label[label_index] = label_log_score;
    }

    let maximum_log_score = unnormalized_log_score_by_label
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    if !maximum_log_score.is_finite() {
        bail!("non-finite max score during softmax");
    }

    let exponentiated_shifted_scores: Vec<f64> = unnormalized_log_score_by_label
        .iter()
        .map(|log_score| (log_score - maximum_log_score).exp())
        .collect();

    if exponentiated_shifted_scores
        .iter()
        .any(|value| !value.is_finite())
    {
        bail!("non-finite value in exponentiated scores");
    }

    let normalization_denominator: f64 = exponentiated_shifted_scores.iter().sum();
    if !normalization_denominator.is_finite() || normalization_denominator <= 0.0 {
        bail!("invalid softmax normalization constant");
    }

    let probability_by_label: Vec<f64> = exponentiated_shifted_scores
        .iter()
        .map(|value| value / normalization_denominator)
        .collect();

    if probability_by_label
        .iter()
        .any(|probability| !probability.is_finite() || *probability < 0.0 || *probability > 1.0)
    {
        bail!("invalid probability vector produced by softmax");
    }

    let (predicted_label_index, _) = probability_by_label
        .iter()
        .copied()
        .enumerate()
        .max_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal))
        .context("failed to select predicted class")?;

    Ok((
        model.labels[predicted_label_index].clone(),
        probability_by_label,
    ))
}

fn build_vocab(
    tokenized_training_texts: &[Vec<String>],
    min_df: usize,
    max_features: usize,
) -> HashMap<String, usize> {
    // Perform parallel counting using mutex-guarded hash maps and then consume
    // those maps into local variables. This avoids creating temporary maps that
    // are overwritten later and removes unused imports.
    let corpus_mutex = Arc::new(Mutex::new(HashMap::<String, usize>::new()));
    let doc_mutex = Arc::new(Mutex::new(HashMap::<String, usize>::new()));

    tokenized_training_texts
        .par_iter()
        .for_each(|tokenized_text| {
            let mut local_seen = BTreeSet::new();
            for token in tokenized_text {
                // update corpus frequency
                {
                    let mut corpus_guard = corpus_mutex.lock().unwrap();
                    *corpus_guard.entry(token.clone()).or_insert(0) += 1;
                }
                local_seen.insert(token.clone());
            }
            // update document frequency
            {
                let mut doc_guard = doc_mutex.lock().unwrap();
                for token in local_seen {
                    *doc_guard.entry(token).or_insert(0) += 1;
                }
            }
        });

    // Move mutex contents into local maps for processing
    let corpus_frequency_by_token = Arc::try_unwrap(corpus_mutex)
        .expect("corpus mutex unwrap")
        .into_inner()
        .expect("corpus mutex into_inner");
    let document_frequency_by_token = Arc::try_unwrap(doc_mutex)
        .expect("doc mutex unwrap")
        .into_inner()
        .expect("doc mutex into_inner");

    let mut candidate_tokens_with_frequency: Vec<(String, usize)> = corpus_frequency_by_token
        .into_iter()
        .filter(|(token, _)| document_frequency_by_token.get(token).copied().unwrap_or(0) >= min_df)
        .collect();

    // Deterministic ordering: highest corpus frequency first, then lexical order.
    candidate_tokens_with_frequency
        .sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));

    candidate_tokens_with_frequency.truncate(max_features);

    let mut vocabulary_index_by_token =
        HashMap::with_capacity(candidate_tokens_with_frequency.len());
    for (token_index, (token, _)) in candidate_tokens_with_frequency.into_iter().enumerate() {
        vocabulary_index_by_token.insert(token, token_index);
    }

    vocabulary_index_by_token
}

fn vectorize_counts(
    input_text: &str,
    vocabulary_index_by_token: &HashMap<String, usize>,
) -> Vec<(usize, u32)> {
    let normalized_text = normalize_text(input_text);
    let tokens = tokenize(&normalized_text);
    let mut token_count_by_index: BTreeMap<usize, u32> = BTreeMap::new();

    for token in tokens {
        if let Some(&token_index) = vocabulary_index_by_token.get(&token) {
            *token_count_by_index.entry(token_index).or_insert(0) += 1;
        }
    }

    token_count_by_index.into_iter().collect()
}

fn validate_runtime_model(model: &ModelArtifact) -> Result<()> {
    if model.labels.is_empty() {
        bail!("runtime model invalid: labels empty");
    }
    if model.vocab.is_empty() {
        bail!("runtime model invalid: vocab empty");
    }
    if model.class_log_prior.len() != model.labels.len() {
        bail!("runtime model invalid: class_log_prior length mismatch");
    }
    if model.feature_log_prob.len() != model.labels.len() {
        bail!("runtime model invalid: feature_log_prob class dimension mismatch");
    }

    let vocabulary_size = model.vocab.len();
    for row in &model.feature_log_prob {
        if row.len() != vocabulary_size {
            bail!("runtime model invalid: feature_log_prob token dimension mismatch");
        }
    }

    Ok(())
}
