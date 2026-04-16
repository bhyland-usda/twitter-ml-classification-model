use anyhow::{Context, Result, bail};
use csv::{ReaderBuilder, StringRecord, WriterBuilder};
use rand::rngs::StdRng;
use rand::{SeedableRng, seq::SliceRandom};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct LabeledRow {
    pub comment_id: String,
    pub text: String,
    pub parent_text: String,
    pub label: String,
}

/// Find a column index by name using case-insensitive matching.
/// This accepts header names like `Text`, `TEXT`, or `text` and will match them
/// against the canonical `name` argument in a case-insensitive way.
pub fn find_col(headers: &StringRecord, name: &str) -> Option<usize> {
    headers
        .iter()
        .position(|column_name| column_name.trim().eq_ignore_ascii_case(name))
}

/// Split a CSV deterministically into train/val/test files using the provided
/// split proportions and a fixed seed for reproducibility. Splitting is done
/// on raw CSV records (before any label-informed transforms).
///
/// `split` is the tuple (train_frac, val_frac, test_frac). The fractions do not
/// need to sum to 1.0; counts are computed as rounded fractions of the total,
/// and any remaining rows are assigned to the test split.
///
/// This function preserves the input CSV header in each output file.
pub fn split_labeled_csv(
    input_path: &Path,
    train_out: &Path,
    val_out: &Path,
    test_out: &Path,
    split: (f64, f64, f64),
    seed: u64,
) -> Result<()> {
    let (train_frac, val_frac, test_frac) = split;

    if train_frac < 0.0 || val_frac < 0.0 || test_frac < 0.0 {
        bail!("invalid split proportions: fractions must be >= 0");
    }

    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(input_path)
        .with_context(|| format!("opening {}", input_path.display()))?;

    let headers = reader
        .headers()
        .with_context(|| format!("reading headers {}", input_path.display()))?
        .clone();

    // Collect records (raw rows) to perform deterministic shuffle and split.
    let mut records: Vec<StringRecord> = Vec::new();
    for (row_idx, record_result) in reader.records().enumerate() {
        let _ = row_idx; // keep for potential debug
        let record = record_result.with_context(|| {
            format!(
                "parsing CSV record at {} row {}",
                input_path.display(),
                row_idx + 2
            )
        })?;
        records.push(record);
    }

    let total = records.len();
    if total == 0 {
        bail!("input CSV has no data rows: {}", input_path.display());
    }

    // Deterministic shuffle using the provided seed.
    let mut indices: Vec<usize> = (0..total).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);

    // Compute counts (rounded), ensure we don't exceed total.
    let train_count = ((train_frac * total as f64).round() as usize).min(total);
    let val_count =
        ((val_frac * total as f64).round() as usize).min(total.saturating_sub(train_count));

    // Ensure output directories exist.
    if let Some(parent) = train_out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating output directory {}", parent.display()))?;
        }
    }
    if let Some(parent) = val_out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating output directory {}", parent.display()))?;
        }
    }
    if let Some(parent) = test_out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating output directory {}", parent.display()))?;
        }
    }

    let mut writer_train = WriterBuilder::new()
        .has_headers(true)
        .from_path(train_out)
        .with_context(|| format!("creating train CSV {}", train_out.display()))?;
    let mut writer_val = WriterBuilder::new()
        .has_headers(true)
        .from_path(val_out)
        .with_context(|| format!("creating val CSV {}", val_out.display()))?;
    let mut writer_test = WriterBuilder::new()
        .has_headers(true)
        .from_path(test_out)
        .with_context(|| format!("creating test CSV {}", test_out.display()))?;

    // Write headers to each output.
    writer_train.write_record(&headers)?;
    writer_val.write_record(&headers)?;
    writer_test.write_record(&headers)?;

    for (i, &shuffled_idx) in indices.iter().enumerate() {
        let record = &records[shuffled_idx];
        if i < train_count {
            writer_train.write_record(record.iter())?;
        } else if i < train_count + val_count {
            writer_val.write_record(record.iter())?;
        } else {
            writer_test.write_record(record.iter())?;
        }
    }

    writer_train.flush()?;
    writer_val.flush()?;
    writer_test.flush()?;

    Ok(())
}

pub fn read_labeled_csv(path: &Path) -> Result<Vec<LabeledRow>> {
    let mut csv_reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .with_context(|| format!("opening {}", path.display()))?;

    let header_record = csv_reader
        .headers()
        .with_context(|| format!("reading headers {}", path.display()))?
        .clone();

    let comment_id_column_index = find_col(&header_record, "comment_id")
        .with_context(|| format!("{} missing 'comment_id' column", path.display()))?;
    let text_column_index = find_col(&header_record, "comment_text")
        .or_else(|| find_col(&header_record, "text"))
        .with_context(|| {
            format!(
                "{} missing required 'comment_text' or 'text' column",
                path.display()
            )
        })?;
    let parent_text_column_index = find_col(&header_record, "parent_text").with_context(|| {
        format!(
            "CSV missing required header: parent_text ({})",
            path.display()
        )
    })?;
    let label_column_index = find_col(&header_record, "label")
        .with_context(|| format!("{} missing 'label' column", path.display()))?;

    let mut parsed_rows = Vec::new();

    let mut seen_comment_ids: HashSet<String> = HashSet::new();
    let mut duplicate_comment_id_count = 0usize;
    let mut duplicate_comment_id_samples: Vec<String> = Vec::new();

    let mut skipped_rows_missing_comment_id = 0usize;
    let mut skipped_rows_missing_text = 0usize;
    let mut skipped_rows_missing_label = 0usize;
    let mut skipped_row_count = 0usize;
    let mut total_data_row_count = 0usize;

    for (row_offset, csv_record_result) in csv_reader.records().enumerate() {
        // Header is line 1, first data row is line 2.
        let line_number = row_offset + 2;
        total_data_row_count += 1;

        let csv_record = csv_record_result.with_context(|| {
            format!(
                "parsing CSV record at {} line {}",
                path.display(),
                line_number
            )
        })?;

        let comment_id_value = csv_record
            .get(comment_id_column_index)
            .unwrap_or("")
            .trim()
            .to_string();
        let text_value = csv_record
            .get(text_column_index)
            .unwrap_or("")
            .trim()
            .to_string();
        let parent_text_value = csv_record
            .get(parent_text_column_index)
            .unwrap_or("")
            .trim()
            .to_string();
        let label_value = csv_record
            .get(label_column_index)
            .unwrap_or("")
            .trim()
            .to_string();

        let mut row_is_missing_required_field = false;

        if comment_id_value.is_empty() {
            skipped_rows_missing_comment_id += 1;
            row_is_missing_required_field = true;
        }
        if text_value.is_empty() {
            skipped_rows_missing_text += 1;
            row_is_missing_required_field = true;
        }
        if label_value.is_empty() {
            skipped_rows_missing_label += 1;
            row_is_missing_required_field = true;
        }

        if parent_text_value.is_empty() {
            bail!(
                "row {} in {} has empty required field: parent_text",
                line_number,
                path.display()
            );
        }

        if row_is_missing_required_field {
            skipped_row_count += 1;
            continue;
        }

        if !seen_comment_ids.insert(comment_id_value.clone()) {
            duplicate_comment_id_count += 1;
            if duplicate_comment_id_samples.len() < 5 {
                duplicate_comment_id_samples.push(comment_id_value.clone());
            }
        }

        parsed_rows.push(LabeledRow {
            comment_id: comment_id_value,
            text: text_value,
            parent_text: parent_text_value,
            label: label_value,
        });
    }

    if parsed_rows.is_empty() {
        bail!(
            "no valid rows found in {} (total_data_rows={}, skipped_rows={}, missing_comment_id={}, missing_text={}, missing_label={})",
            path.display(),
            total_data_row_count,
            skipped_row_count,
            skipped_rows_missing_comment_id,
            skipped_rows_missing_text,
            skipped_rows_missing_label
        );
    }

    if skipped_row_count > 0 {
        eprintln!(
            "[WARN] csv_validation file={} total_data_rows={} valid_rows={} skipped_rows={} missing_comment_id={} missing_text={} missing_label={}",
            path.display(),
            total_data_row_count,
            parsed_rows.len(),
            skipped_row_count,
            skipped_rows_missing_comment_id,
            skipped_rows_missing_text,
            skipped_rows_missing_label
        );
    }

    if duplicate_comment_id_count > 0 {
        eprintln!(
            "[WARN] duplicate_comment_id file={} duplicate_rows={} sample_ids=[{}]",
            path.display(),
            duplicate_comment_id_count,
            duplicate_comment_id_samples.join(", ")
        );
    }

    Ok(parsed_rows)
}
