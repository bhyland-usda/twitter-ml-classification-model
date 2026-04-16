use crate::{data::LabeledRow, model::ModelArtifact, nb::predict_with_context};
use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::collections::HashMap;

const UNKNOWN_LABEL_MAX_RATIO: f64 = 0.05;
const UNKNOWN_LABEL_SAMPLE_LIMIT: usize = 10;

#[derive(Debug, Serialize)]
pub struct PerClassMetrics {
    pub label: String,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

#[derive(Debug, Serialize)]
pub struct EvalMetrics {
    pub total: usize,
    pub accuracy: f64,
    pub macro_f1: f64,
    pub per_class: Vec<PerClassMetrics>,
    pub confusion_matrix: Vec<Vec<usize>>, // [true][pred]
    pub labels: Vec<String>,
}

pub fn evaluate(model: &ModelArtifact, rows: &[LabeledRow]) -> Result<EvalMetrics> {
    if rows.is_empty() {
        bail!("evaluation rows empty");
    }

    let label_to_index_map: HashMap<String, usize> = model
        .labels
        .iter()
        .enumerate()
        .map(|(label_index, label_name)| (label_name.clone(), label_index))
        .collect();

    let number_of_labels = model.labels.len();
    let mut confusion_matrix = vec![vec![0usize; number_of_labels]; number_of_labels];
    let mut evaluated_row_count = 0usize;
    let mut correctly_predicted_row_count = 0usize;

    let mut unknown_label_rows: Vec<(String, String)> = Vec::new();

    for row in rows {
        let Some(&true_label_index) = label_to_index_map.get(&row.label) else {
            unknown_label_rows.push((row.comment_id.clone(), row.label.clone()));
            continue;
        };

        let (predicted_label_name, _) = predict_with_context(model, &row.parent_text, &row.text)?;
        let predicted_label_index = *label_to_index_map
            .get(&predicted_label_name)
            .with_context(|| format!("predicted unknown label {}", predicted_label_name))?;

        confusion_matrix[true_label_index][predicted_label_index] += 1;
        if true_label_index == predicted_label_index {
            correctly_predicted_row_count += 1;
        }
        evaluated_row_count += 1;
    }

    let unknown_label_count = unknown_label_rows.len();
    if unknown_label_count > 0 {
        let unknown_label_ratio = unknown_label_count as f64 / rows.len() as f64;
        let unknown_label_sample = unknown_label_rows
            .iter()
            .take(UNKNOWN_LABEL_SAMPLE_LIMIT)
            .map(|(comment_id, label)| format!("{}:{}", comment_id, label))
            .collect::<Vec<_>>()
            .join(", ");

        eprintln!(
            "[WARN] skipped_rows_with_unknown_labels={} ratio={:.4} sample=[{}]",
            unknown_label_count, unknown_label_ratio, unknown_label_sample
        );

        if unknown_label_ratio > UNKNOWN_LABEL_MAX_RATIO {
            bail!(
                "too many rows have unknown labels: skipped={} total={} ratio={:.4} threshold={:.4} sample=[{}]",
                unknown_label_count,
                rows.len(),
                unknown_label_ratio,
                UNKNOWN_LABEL_MAX_RATIO,
                unknown_label_sample
            );
        }
    }

    if evaluated_row_count == 0 {
        bail!("no evaluable rows: eval labels not present in model labels");
    }

    let overall_accuracy = correctly_predicted_row_count as f64 / evaluated_row_count as f64;
    let mut per_class_metrics = Vec::with_capacity(number_of_labels);
    let mut total_class_f1_sum = 0.0;

    for label_index in 0..number_of_labels {
        let true_positive_count = confusion_matrix[label_index][label_index] as f64;

        let false_positive_count: f64 = (0..number_of_labels)
            .filter(|&row_index| row_index != label_index)
            .map(|row_index| confusion_matrix[row_index][label_index] as f64)
            .sum();

        let false_negative_count: f64 = (0..number_of_labels)
            .filter(|&column_index| column_index != label_index)
            .map(|column_index| confusion_matrix[label_index][column_index] as f64)
            .sum();

        let support_count: usize = confusion_matrix[label_index].iter().sum();

        let precision = if true_positive_count + false_positive_count > 0.0 {
            true_positive_count / (true_positive_count + false_positive_count)
        } else {
            0.0
        };

        let recall = if true_positive_count + false_negative_count > 0.0 {
            true_positive_count / (true_positive_count + false_negative_count)
        } else {
            0.0
        };

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        total_class_f1_sum += f1;
        per_class_metrics.push(PerClassMetrics {
            label: model.labels[label_index].clone(),
            precision,
            recall,
            f1,
            support: support_count,
        });
    }

    let macro_f1 = total_class_f1_sum / number_of_labels as f64;

    Ok(EvalMetrics {
        total: evaluated_row_count,
        accuracy: overall_accuracy,
        macro_f1,
        per_class: per_class_metrics,
        confusion_matrix,
        labels: model.labels.clone(),
    })
}

pub fn print_metrics(name: &str, metrics: &EvalMetrics) {
    println!(
        "[METRICS] split={} total={} accuracy={:.6} macro_f1={:.6}",
        name, metrics.total, metrics.accuracy, metrics.macro_f1
    );

    for class_metrics in &metrics.per_class {
        println!(
            "[CLASS] split={} label={} precision={:.6} recall={:.6} f1={:.6} support={}",
            name,
            class_metrics.label,
            class_metrics.precision,
            class_metrics.recall,
            class_metrics.f1,
            class_metrics.support
        );
    }
}
