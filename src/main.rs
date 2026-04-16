mod artifact;
mod cli;
mod commands;
mod context;
mod data;
mod metrics;
mod model;
mod nb;
mod text;

use anyhow::Result;
use clap::Parser;
use cli::Cli;
use commands::run;

fn main() -> Result<()> {
    let command_line_arguments = Cli::parse();
    run(command_line_arguments)
}
