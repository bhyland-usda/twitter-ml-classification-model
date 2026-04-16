use crate::{
    artifact::{load_model, write_model},
    cli::{Cli, Commands},
    data::{find_col, read_labeled_csv, split_labeled_csv},
    metrics::{evaluate, print_metrics},
    nb::{predict_with_context, train_naive_bayes},
};
use anyhow::{Context, Result, bail};
use csv::{ReaderBuilder, WriterBuilder};
use serde::Serialize;
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};
use toml;

fn parse_split_string(split: Option<String>) -> Result<(f64, f64, f64)> {
    if let Some(s) = split {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 3 {
            bail!("--split must be three comma-separated floats: train,val,test");
        }
        let a: f64 = parts[0]
            .trim()
            .parse()
            .with_context(|| "parsing train fraction")?;
        let b: f64 = parts[1]
            .trim()
            .parse()
            .with_context(|| "parsing val fraction")?;
        let c: f64 = parts[2]
            .trim()
            .parse()
            .with_context(|| "parsing test fraction")?;
        if a < 0.0 || b < 0.0 || c < 0.0 {
            bail!("split fractions must be non-negative");
        }
        Ok((a, b, c))
    } else {
        Ok((1.0, 0.0, 0.0))
    }
}

pub fn run(command_line_arguments: Cli) -> Result<()> {
    match command_line_arguments.command {
        Commands::Train {
            train_csv,
            model_out,
            val_csv,
            min_df,
            max_features,
            alpha,
            split,
            seed,
        } => cmd_train(
            train_csv,
            model_out,
            val_csv,
            min_df,
            max_features,
            alpha,
            split,
            seed,
        ),
        Commands::Split {
            input_csv,
            train_out,
            val_out,
            test_out,
            split,
            seed,
        } => cmd_split(input_csv, train_out, val_out, test_out, split, seed),
        Commands::Eval {
            model_path,
            input_csv,
        } => cmd_eval(model_path, input_csv),
        Commands::PredictText {
            model_path,
            parent_text,
            text,
        } => cmd_predict_text(model_path, parent_text, text),
        Commands::PredictCsv {
            model_path,
            input_csv,
            output_csv,
        } => cmd_predict_csv(model_path, input_csv, output_csv),
    }
}

fn cmd_split(
    input_csv: PathBuf,
    train_out: PathBuf,
    val_out: PathBuf,
    test_out: PathBuf,
    split: Option<String>,
    seed: u64,
) -> Result<()> {
    // Reuse the existing parser helper to validate split format
    let (train_frac, val_frac, test_frac) = parse_split_string(split)?;
    split_labeled_csv(
        &input_csv,
        &train_out,
        &val_out,
        &test_out,
        (train_frac, val_frac, test_frac),
        seed,
    )?;
    println!(
        "[INFO] split_completed input={} train={} val={} test={}",
        input_csv.display(),
        train_out.display(),
        val_out.display(),
        test_out.display()
    );
    Ok(())
}

fn cmd_train(
    training_csv_path: PathBuf,
    model_output_path: PathBuf,
    validation_csv_path: Option<PathBuf>,
    min_df: usize,
    max_features: usize,
    alpha: f64,
    split: Option<String>,
    seed: u64,
) -> Result<()> {
    if alpha <= 0.0 {
        bail!("--alpha must be > 0");
    }
    if min_df == 0 {
        bail!("--min-df must be >= 1");
    }
    if max_features == 0 {
        bail!("--max-features must be >= 1");
    }

    // If split is provided, perform a reproducible split into ml_artifacts/training/<run_id>/
    // and use the generated train/val files for subsequent steps.
    let mut train_csv_to_use = training_csv_path.clone();
    let mut val_csv_to_use: Option<PathBuf> = validation_csv_path.clone();
    let mut run_dir: Option<PathBuf> = None;

    if split.is_some() {
        let (train_frac, val_frac, test_frac) = parse_split_string(split.clone())?;
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let run_id = format!("run_{}_s{}", now_secs, seed);
        let base_dir = PathBuf::from("ml_artifacts").join("training").join(&run_id);
        fs::create_dir_all(&base_dir)
            .with_context(|| format!("creating run directory {}", base_dir.display()))?;

        let train_out = base_dir.join("train.csv");
        let val_out = base_dir.join("val.csv");
        let test_out = base_dir.join("test.csv");

        split_labeled_csv(
            &training_csv_path,
            &train_out,
            &val_out,
            &test_out,
            (train_frac, val_frac, test_frac),
            seed,
        )
        .with_context(|| {
            format!(
                "splitting {} into {}",
                training_csv_path.display(),
                base_dir.display()
            )
        })?;

        train_csv_to_use = train_out;
        // prefer CLI-provided validation CSV if present; otherwise use split val
        if validation_csv_path.is_none() {
            val_csv_to_use = Some(val_out);
        }
        run_dir = Some(base_dir);
    }

    let training_rows = read_labeled_csv(&train_csv_to_use)
        .with_context(|| format!("reading train CSV {}", train_csv_to_use.display()))?;

    if training_rows.is_empty() {
        bail!("train CSV has no valid rows");
    }

    // Train model
    let trained_model = train_naive_bayes(&training_rows, min_df, max_features, alpha)?;
    write_model(&model_output_path, &trained_model)?;

    // If a run directory was created, also save the model into the run directory for artifact versioning.
    if let Some(ref base_dir) = run_dir {
        let run_model_path = base_dir.join("model.json");
        let serialized =
            serde_json::to_string_pretty(&trained_model).context("serializing model artifact")?;
        fs::write(&run_model_path, serialized)
            .with_context(|| format!("writing run model {}", run_model_path.display()))?;
    }

    println!("[INFO] model_written={}", model_output_path.display());
    println!("[INFO] labels={:?}", trained_model.labels);
    println!("[INFO] vocab_size={}", trained_model.vocab.len());

    // Evaluate on training data
    let training_metrics = evaluate(&trained_model, &training_rows)?;
    print_metrics("train", &training_metrics);

    // Optionally evaluate on validation data (from CLI arg or split)
    let mut validation_metrics_opt: Option<crate::metrics::EvalMetrics> = None;
    if let Some(validation_path) = val_csv_to_use {
        let validation_rows = read_labeled_csv(&validation_path)
            .with_context(|| format!("reading val CSV {}", validation_path.display()))?;

        if !validation_rows.is_empty() {
            let validation_metrics = evaluate(&trained_model, &validation_rows)?;
            print_metrics("validation", &validation_metrics);
            validation_metrics_opt = Some(validation_metrics);
        } else {
            println!(
                "[WARN] validation CSV has no valid rows: {}",
                validation_path.display()
            );
        }
    }

    // Write structured TOML training metadata into run dir if available, else into ml_artifacts/training/latest/
    if let Some(base_dir) = run_dir.or_else(|| {
        Some(
            PathBuf::from("ml_artifacts")
                .join("training")
                .join("run_latest"),
        )
    }) {
        if !base_dir.exists() {
            fs::create_dir_all(&base_dir)
                .with_context(|| format!("creating metadata directory {}", base_dir.display()))?;
        }

        #[derive(Serialize)]
        struct TrainingMetadata<'a> {
            run_id: String,
            seed: u64,
            split: Option<String>,
            min_df: usize,
            max_features: usize,
            alpha: f64,
            model_output: String,
            train_rows: usize,
            validation_rows: Option<usize>,
            train_metrics: &'a crate::metrics::EvalMetrics,
            validation_metrics: Option<&'a crate::metrics::EvalMetrics>,
        }

        let run_id_str = base_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("run")
            .to_string();
        let metadata = TrainingMetadata {
            run_id: run_id_str,
            seed,
            split: split.clone(),
            min_df,
            max_features,
            alpha,
            model_output: model_output_path.display().to_string(),
            train_rows: training_rows.len(),
            validation_rows: validation_metrics_opt.as_ref().map(|m| m.total),
            train_metrics: &training_metrics,
            validation_metrics: validation_metrics_opt.as_ref(),
        };

        let toml_serialized =
            toml::to_string(&metadata).context("serializing training metadata to TOML")?;
        let metadata_path = base_dir.join("training_metadata.toml");
        fs::write(&metadata_path, toml_serialized)
            .with_context(|| format!("writing metadata {}", metadata_path.display()))?;
        println!(
            "[INFO] training_metadata_written={}",
            metadata_path.display()
        );
    }

    Ok(())
}

fn cmd_eval(model_path: PathBuf, evaluation_csv_path: PathBuf) -> Result<()> {
    let model_artifact = load_model(&model_path)?;
    let evaluation_rows = read_labeled_csv(&evaluation_csv_path)
        .with_context(|| format!("reading eval CSV {}", evaluation_csv_path.display()))?;

    let evaluation_metrics = evaluate(&model_artifact, &evaluation_rows)?;
    print_metrics("evaluation", &evaluation_metrics);
    Ok(())
}

fn cmd_predict_text(model_path: PathBuf, parent_text: String, input_text: String) -> Result<()> {
    let model_artifact = load_model(&model_path)?;
    let (predicted_label_name, predicted_probabilities) =
        predict_with_context(&model_artifact, &parent_text, &input_text)?;

    println!("[PREDICTION] {}", predicted_label_name);
    for (label_name, probability) in model_artifact
        .labels
        .iter()
        .zip(predicted_probabilities.iter())
    {
        println!("[PROB] {}={:.6}", label_name, probability);
    }

    // CLI-level confidence mapping: if input text is empty/whitespace-only, reported confidence must be 0.0
    let cli_confidence = if input_text.trim().is_empty() {
        0.0_f64
    } else {
        predicted_probabilities
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(0.0)
    };

    println!(
        "[CONFIDENCE] {cli_confidence:.6}",
        cli_confidence = cli_confidence
    );

    Ok(())
}

fn cmd_predict_csv(
    model_path: PathBuf,
    input_csv_path: PathBuf,
    output_csv_path: PathBuf,
) -> Result<()> {
    let model_artifact = load_model(&model_path)?;

    let mut csv_reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(&input_csv_path)
        .with_context(|| format!("opening input CSV {}", input_csv_path.display()))?;

    let input_headers = csv_reader
        .headers()
        .with_context(|| format!("reading headers {}", input_csv_path.display()))?
        .clone();

    let text_column_index = find_col(&input_headers, "text")
        .with_context(|| format!("{} missing 'text' column", input_csv_path.display()))?;
    let comment_id_column_index = find_col(&input_headers, "comment_id");
    let parent_text_column_index = find_col(&input_headers, "parent_text")
        .with_context(|| format!("{} missing 'parent_text' column", input_csv_path.display()))?;

    if let Some(output_directory_path) = output_csv_path.parent() {
        if !output_directory_path.as_os_str().is_empty() {
            fs::create_dir_all(output_directory_path).with_context(|| {
                format!(
                    "creating output directory {}",
                    output_directory_path.display()
                )
            })?;
        }
    }

    let mut csv_writer = WriterBuilder::new()
        .has_headers(true)
        .from_path(&output_csv_path)
        .with_context(|| format!("creating output CSV {}", output_csv_path.display()))?;

    csv_writer.write_record([
        "comment_id",
        "text",
        "parent_text",
        "predicted_label",
        "confidence",
    ])?;

    for (row_offset, csv_record_result) in csv_reader.records().enumerate() {
        let line_number = row_offset + 2;
        let csv_record = csv_record_result?;
        let input_text = csv_record.get(text_column_index).unwrap_or("").trim();

        let comment_id_value = comment_id_column_index
            .and_then(|column_index| csv_record.get(column_index))
            .unwrap_or("")
            .trim();
        let parent_text_value = csv_record
            .get(parent_text_column_index)
            .unwrap_or("")
            .trim();

        if parent_text_value.is_empty() {
            bail!(
                "row {} in {} has empty required field: parent_text",
                line_number,
                input_csv_path.display()
            );
        }

        let (predicted_label_name, predicted_probabilities) =
            predict_with_context(&model_artifact, parent_text_value, input_text)?;

        // CLI-level behavior: if the text is empty/whitespace-only, report confidence 0.0
        let prediction_confidence = if input_text.trim().is_empty() {
            0.0_f64
        } else {
            predicted_probabilities
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
                .max(0.0)
        };

        let output_record = vec![
            comment_id_value.to_string(),
            input_text.to_string(),
            parent_text_value.to_string(),
            predicted_label_name,
            format!("{:.6}", prediction_confidence),
        ];

        csv_writer.write_record(output_record.iter().map(String::as_str))?;
    }

    csv_writer.flush()?;
    println!("[INFO] predictions_written={}", output_csv_path.display());
    Ok(())
}
