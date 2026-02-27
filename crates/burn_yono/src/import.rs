use std::path::{Path, PathBuf};

use burn::{
    module::Module,
    prelude::Backend,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, RecorderError},
};
use burn_store::{
    ApplyResult, BurnpackStore, KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};

use crate::model::{
    CrocoStyleBackbone, CrocoStyleBackboneConfig, YonoHeadConfig, YonoHeadPipeline,
};

#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error("invalid key remap rule `{0}` -> `{1}`: {2}")]
    InvalidRemap(String, String, String),
    #[error("failed to apply checkpoint tensors: {0}")]
    Apply(String),
    #[error("failed to save burn record: {0}")]
    Save(#[from] RecorderError),
    #[error("failed to save burnpack record: {0}")]
    SaveBurnpack(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckpointFormat {
    Mpk,
    Bpk,
}

pub fn load_yono_head_from_safetensors<B: Backend>(
    device: &B::Device,
    config: YonoHeadConfig,
    path: &Path,
) -> Result<(YonoHeadPipeline<B>, ApplyResult), ImportError> {
    let mut model = YonoHeadPipeline::new(device, config);
    let mut store = build_store(path)?;

    let result = model
        .load_from(&mut store)
        .map_err(|err| ImportError::Apply(format!("{err:?}")))?;

    Ok((model, result))
}

pub fn load_yono_head_from_burnpack<B: Backend>(
    device: &B::Device,
    config: YonoHeadConfig,
    path: &Path,
) -> Result<(YonoHeadPipeline<B>, ApplyResult), ImportError> {
    let mut model = YonoHeadPipeline::new(device, config);
    let mut store = BurnpackStore::from_file(path)
        .auto_extension(false)
        .validate(true);

    let result = model
        .load_from(&mut store)
        .map_err(|err| ImportError::Apply(format!("{err:?}")))?;

    Ok((model, result))
}

pub fn load_yono_backbone_from_safetensors<B: Backend>(
    device: &B::Device,
    config: CrocoStyleBackboneConfig,
    path: &Path,
) -> Result<(CrocoStyleBackbone<B>, ApplyResult), ImportError> {
    let mut model = CrocoStyleBackbone::new(device, config);
    let mut store = build_backbone_store(path)?;

    let result = model
        .load_from(&mut store)
        .map_err(|err| ImportError::Apply(format!("{err:?}")))?;

    Ok((model, result))
}

pub fn load_yono_backbone_from_burnpack<B: Backend>(
    device: &B::Device,
    config: CrocoStyleBackboneConfig,
    path: &Path,
) -> Result<(CrocoStyleBackbone<B>, ApplyResult), ImportError> {
    let mut model = CrocoStyleBackbone::new(device, config);
    let mut store = BurnpackStore::from_file(path)
        .auto_extension(false)
        .validate(true);

    let result = model
        .load_from(&mut store)
        .map_err(|err| ImportError::Apply(format!("{err:?}")))?;

    Ok((model, result))
}

pub fn save_yono_head_record<B: Backend>(
    model: &YonoHeadPipeline<B>,
    format: CheckpointFormat,
    output_base: &Path,
) -> Result<PathBuf, ImportError> {
    match format {
        CheckpointFormat::Mpk => save_yono_head_record_mpk(model, output_base),
        CheckpointFormat::Bpk => save_yono_head_record_bpk(model, output_base),
    }
}

pub fn save_yono_backbone_record<B: Backend>(
    model: &CrocoStyleBackbone<B>,
    format: CheckpointFormat,
    output_base: &Path,
) -> Result<PathBuf, ImportError> {
    match format {
        CheckpointFormat::Mpk => save_yono_backbone_record_mpk(model, output_base),
        CheckpointFormat::Bpk => save_yono_backbone_record_bpk(model, output_base),
    }
}

pub fn save_yono_head_record_mpk<B: Backend>(
    model: &YonoHeadPipeline<B>,
    output_base: &Path,
) -> Result<PathBuf, ImportError> {
    let output = normalize_extension(output_base, "mpk");
    let base = output.with_extension("");

    model.clone().save_file(
        base.clone(),
        &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
    )?;

    Ok(base.with_extension("mpk"))
}

pub fn save_yono_backbone_record_mpk<B: Backend>(
    model: &CrocoStyleBackbone<B>,
    output_base: &Path,
) -> Result<PathBuf, ImportError> {
    let output = normalize_extension(output_base, "mpk");
    let base = output.with_extension("");

    model.clone().save_file(
        base.clone(),
        &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
    )?;

    Ok(base.with_extension("mpk"))
}

pub fn save_yono_head_record_bpk<B: Backend>(
    model: &YonoHeadPipeline<B>,
    output_base: &Path,
) -> Result<PathBuf, ImportError> {
    let output = normalize_extension(output_base, "bpk");
    let mut store = BurnpackStore::from_file(&output)
        .auto_extension(false)
        .overwrite(true);

    model
        .save_into(&mut store)
        .map_err(|err| ImportError::SaveBurnpack(err.to_string()))?;

    Ok(output)
}

pub fn save_yono_backbone_record_bpk<B: Backend>(
    model: &CrocoStyleBackbone<B>,
    output_base: &Path,
) -> Result<PathBuf, ImportError> {
    let output = normalize_extension(output_base, "bpk");
    let mut store = BurnpackStore::from_file(&output)
        .auto_extension(false)
        .overwrite(true);

    model
        .save_into(&mut store)
        .map_err(|err| ImportError::SaveBurnpack(err.to_string()))?;

    Ok(output)
}

pub fn report_apply_result(prefix: &str, result: &ApplyResult) {
    println!(
        "[IMPORT] {prefix} tensors applied: {} (missing {}, unused {}, skipped {})",
        result.applied.len(),
        result.missing.len(),
        result.unused.len(),
        result.skipped.len()
    );

    if !result.missing.is_empty() {
        println!("[IMPORT] Missing {} tensor(s):", result.missing.len());
        for key in &result.missing {
            println!("  - {key}");
        }
    }

    if !result.unused.is_empty() {
        println!("[IMPORT] Unused {} tensor(s):", result.unused.len());
        for key in &result.unused {
            println!("  - {key}");
        }
    }

    if !result.skipped.is_empty() {
        println!("[IMPORT] Skipped {} tensor(s):", result.skipped.len());
        for key in &result.skipped {
            println!("  - {key}");
        }
    }
}

fn build_store(path: &Path) -> Result<SafetensorsStore, ImportError> {
    let mut remapper = KeyRemapper::new();
    for &(from, to) in head_key_remap_rules() {
        remapper = remapper.add_pattern(from, to).map_err(|err| {
            ImportError::InvalidRemap(from.to_string(), to.to_string(), err.to_string())
        })?;
    }

    Ok(SafetensorsStore::from_file(path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(true)
        .remap(remapper)
        .validate(true))
}

fn build_backbone_store(path: &Path) -> Result<SafetensorsStore, ImportError> {
    let mut remapper = KeyRemapper::new();
    for &(from, to) in backbone_key_remap_rules() {
        remapper = remapper.add_pattern(from, to).map_err(|err| {
            ImportError::InvalidRemap(from.to_string(), to.to_string(), err.to_string())
        })?;
    }

    Ok(SafetensorsStore::from_file(path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(true)
        .remap(remapper)
        .validate(true))
}

fn normalize_extension(path: &Path, extension: &str) -> PathBuf {
    if path
        .extension()
        .map(|ext| ext.eq_ignore_ascii_case(extension))
        .unwrap_or(false)
    {
        path.to_path_buf()
    } else {
        path.with_extension(extension)
    }
}

fn head_key_remap_rules() -> &'static [(&'static str, &'static str)] {
    &[
        (r"^encoder\.(.*)$", "$1"),
        (
            r"^camera_head\.more_mlps\.0\.(weight|bias)$",
            "camera_head.mlp_0.$1",
        ),
        (
            r"^camera_head\.more_mlps\.2\.(weight|bias)$",
            "camera_head.mlp_1.$1",
        ),
        (r"^(.*\.norm\d?)\.weight$", "$1.gamma"),
        (r"^(.*\.norm\d?)\.bias$", "$1.beta"),
        (r"^(.*\.q_norm)\.weight$", "$1.gamma"),
        (r"^(.*\.q_norm)\.bias$", "$1.beta"),
        (r"^(.*\.k_norm)\.weight$", "$1.gamma"),
        (r"^(.*\.k_norm)\.bias$", "$1.beta"),
        (r"^(.*\.norm)\.weight$", "$1.gamma"),
        (r"^(.*\.norm)\.bias$", "$1.beta"),
    ]
}

fn backbone_key_remap_rules() -> &'static [(&'static str, &'static str)] {
    &[
        (r"^encoder\.(.*)$", "$1"),
        (r"^backbone\.(.*)$", "$1"),
        (r"^decoder\.(.*)$", "blocks.$1"),
        (r"^intrinsics_embed_layer\.(.*)$", "intrinsics_embed.$1"),
        (r"^(.*\.norm\d?)\.weight$", "$1.gamma"),
        (r"^(.*\.norm\d?)\.bias$", "$1.beta"),
        (r"^(.*\.q_norm)\.weight$", "$1.gamma"),
        (r"^(.*\.q_norm)\.bias$", "$1.beta"),
        (r"^(.*\.k_norm)\.weight$", "$1.gamma"),
        (r"^(.*\.k_norm)\.bias$", "$1.beta"),
        (r"^(.*\.norm)\.weight$", "$1.gamma"),
        (r"^(.*\.norm)\.bias$", "$1.beta"),
    ]
}
