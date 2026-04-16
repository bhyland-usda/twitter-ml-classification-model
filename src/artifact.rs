use crate::model::{ModelArtifact, validate_model};
use anyhow::{Context, Result};
use std::{fs, path::Path};

pub fn write_model(model_output_path: &Path, model_artifact: &ModelArtifact) -> Result<()> {
    if let Some(output_directory_path) = model_output_path.parent() {
        if !output_directory_path.as_os_str().is_empty() {
            fs::create_dir_all(output_directory_path).with_context(|| {
                format!(
                    "creating model output directory {}",
                    output_directory_path.display()
                )
            })?;
        }
    }

    let serialized_model_json =
        serde_json::to_string_pretty(model_artifact).context("serializing model artifact")?;

    fs::write(model_output_path, serialized_model_json)
        .with_context(|| format!("writing model artifact {}", model_output_path.display()))?;

    Ok(())
}

pub fn load_model(model_input_path: &Path) -> Result<ModelArtifact> {
    let serialized_model_json = fs::read_to_string(model_input_path)
        .with_context(|| format!("reading {}", model_input_path.display()))?;

    let parsed_model_artifact: ModelArtifact = serde_json::from_str(&serialized_model_json)
        .with_context(|| format!("parsing {}", model_input_path.display()))?;

    validate_model(&parsed_model_artifact)?;
    Ok(parsed_model_artifact)
}
