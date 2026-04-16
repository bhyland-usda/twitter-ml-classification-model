use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "classifier")]
#[command(about = "Naive Bayes text classifier (train/eval/predict/split)")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Train a model from a labeled CSV. Optionally perform an internal
    /// reproducible split if `--split` is provided.
    Train {
        #[arg(long)]
        train_csv: PathBuf,
        #[arg(long)]
        model_out: PathBuf,
        /// Optional validation CSV. If provided, validation will be run.
        #[arg(long)]
        val_csv: Option<PathBuf>,
        #[arg(long, default_value_t = 2)]
        min_df: usize,
        #[arg(long, default_value_t = 20_000)]
        max_features: usize,
        #[arg(long, default_value_t = 1.0)]
        alpha: f64,
        /// Optional split proportions as comma-separated floats train,val,test (e.g. 0.8,0.1,0.1).
        /// When provided, the tool will split `train_csv` reproducibly using `--seed`.
        #[arg(long)]
        split: Option<String>,
        /// RNG seed for reproducible splitting and other stochastic steps (default 42).
        #[arg(long, default_value_t = 42u64)]
        seed: u64,
    },

    /// Standalone split utility: split an input CSV into train/val/test files
    /// deterministically using the provided seed and optional split proportions.
    Split {
        #[arg(long)]
        input_csv: PathBuf,
        #[arg(long)]
        train_out: PathBuf,
        #[arg(long)]
        val_out: PathBuf,
        #[arg(long)]
        test_out: PathBuf,
        /// Optional split proportions as comma-separated floats train,val,test (e.g. 0.8,0.1,0.1).
        #[arg(long)]
        split: Option<String>,
        /// RNG seed for reproducible splitting (default 42).
        #[arg(long, default_value_t = 42u64)]
        seed: u64,
    },

    Eval {
        #[arg(long)]
        model_path: PathBuf,
        #[arg(long)]
        input_csv: PathBuf,
    },
    PredictText {
        #[arg(long)]
        model_path: PathBuf,
        #[arg(long)]
        parent_text: String,
        #[arg(long)]
        text: String,
    },
    PredictCsv {
        #[arg(long)]
        model_path: PathBuf,
        #[arg(long)]
        input_csv: PathBuf,
        #[arg(long)]
        output_csv: PathBuf,
    },
}
