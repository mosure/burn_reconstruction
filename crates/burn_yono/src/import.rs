use std::collections::BTreeSet;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use burn::{
    module::Module,
    prelude::Backend,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, RecorderError},
    tensor::{Bytes, DType, TensorData},
};
use burn_store::{
    ApplyResult, BurnpackStore, KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};
use serde::{Deserialize, Serialize};

use crate::model::{
    CrocoStyleBackbone, CrocoStyleBackboneConfig, YonoHeadConfig, YonoHeadPipeline,
};
use crate::parts::{
    load_model_from_burnpack_parts_with_progress, write_burnpack_parts_for_wasm,
    BurnpackPartsReport,
};
use crate::{burnpack_path_for_precision, YonoWeightPrecision};

pub const DEFAULT_PART_SIZE_MIB: u64 = 64;
const BURNPACK_MAGIC_NUMBER: u32 = 0x4255_524E;
const BURNPACK_HEADER_SIZE: usize = 10;

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
    #[error("burnpack parts error: {0}")]
    Parts(String),
    #[error("burnpack precision conversion error: {0}")]
    BurnpackConversion(String),
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
    load_yono_head_from_burnpack_candidates(device, config, &[path.to_path_buf()])
}

pub fn load_yono_head_from_burnpack_candidates<B: Backend>(
    device: &B::Device,
    config: YonoHeadConfig,
    paths: &[PathBuf],
) -> Result<(YonoHeadPipeline<B>, ApplyResult), ImportError> {
    load_yono_head_from_burnpack_candidates_with_progress(device, config, paths, |_| {})
}

pub fn load_yono_head_from_burnpack_candidates_with_progress<B: Backend, F>(
    device: &B::Device,
    config: YonoHeadConfig,
    paths: &[PathBuf],
    mut progress: F,
) -> Result<(YonoHeadPipeline<B>, ApplyResult), ImportError>
where
    F: FnMut(String),
{
    if paths.is_empty() {
        return Err(ImportError::Apply(
            "no burnpack candidate paths supplied for YoNo head".to_string(),
        ));
    }

    if let Some((model, result)) = load_model_from_burnpack_parts_with_progress(
        paths,
        "YoNo head",
        true,
        || YonoHeadPipeline::new(device, config.clone()),
        |model, part_bytes| apply_burnpack_part_bytes(model, part_bytes),
        &mut progress,
    )
    .map_err(ImportError::Parts)?
    {
        return Ok((model, result));
    }

    let fallback = first_existing_or_first(paths);
    progress("loading YoNo head monolithic burnpack".to_string());
    let mut model = YonoHeadPipeline::new(device, config);
    let mut store = BurnpackStore::from_file(fallback)
        .auto_extension(false)
        .validate(true);

    let result = model
        .load_from(&mut store)
        .map_err(|err| ImportError::Apply(format!("{err:?}")))?;

    Ok((model, result))
}

pub fn load_yono_head_from_burnpack_part_bytes<B: Backend>(
    device: &B::Device,
    config: YonoHeadConfig,
    parts: &[Vec<u8>],
) -> Result<(YonoHeadPipeline<B>, ApplyResult), ImportError> {
    load_yono_head_from_burnpack_part_bytes_with_progress(device, config, parts, |_| {})
}

pub fn load_yono_head_from_burnpack_part_bytes_with_progress<B: Backend, F>(
    device: &B::Device,
    config: YonoHeadConfig,
    parts: &[Vec<u8>],
    mut progress: F,
) -> Result<(YonoHeadPipeline<B>, ApplyResult), ImportError>
where
    F: FnMut(String),
{
    let mut model = YonoHeadPipeline::new(device, config);
    let result = apply_burnpack_parts_bytes_with_progress(&mut model, parts, |index, total| {
        progress(format!("loading yono head part {index}/{total}"))
    })?;
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
    load_yono_backbone_from_burnpack_candidates(device, config, &[path.to_path_buf()])
}

pub fn load_yono_backbone_from_burnpack_candidates<B: Backend>(
    device: &B::Device,
    config: CrocoStyleBackboneConfig,
    paths: &[PathBuf],
) -> Result<(CrocoStyleBackbone<B>, ApplyResult), ImportError> {
    load_yono_backbone_from_burnpack_candidates_with_progress(device, config, paths, |_| {})
}

pub fn load_yono_backbone_from_burnpack_candidates_with_progress<B: Backend, F>(
    device: &B::Device,
    config: CrocoStyleBackboneConfig,
    paths: &[PathBuf],
    mut progress: F,
) -> Result<(CrocoStyleBackbone<B>, ApplyResult), ImportError>
where
    F: FnMut(String),
{
    if paths.is_empty() {
        return Err(ImportError::Apply(
            "no burnpack candidate paths supplied for YoNo backbone".to_string(),
        ));
    }

    if let Some((model, result)) = load_model_from_burnpack_parts_with_progress(
        paths,
        "YoNo backbone",
        true,
        || CrocoStyleBackbone::new(device, config.clone()),
        |model, part_bytes| apply_burnpack_part_bytes(model, part_bytes),
        &mut progress,
    )
    .map_err(ImportError::Parts)?
    {
        return Ok((model, result));
    }

    let fallback = first_existing_or_first(paths);
    progress("loading YoNo backbone monolithic burnpack".to_string());
    let mut model = CrocoStyleBackbone::new(device, config);
    let mut store = BurnpackStore::from_file(fallback)
        .auto_extension(false)
        .validate(true);

    let result = model
        .load_from(&mut store)
        .map_err(|err| ImportError::Apply(format!("{err:?}")))?;

    Ok((model, result))
}

pub fn load_yono_backbone_from_burnpack_part_bytes<B: Backend>(
    device: &B::Device,
    config: CrocoStyleBackboneConfig,
    parts: &[Vec<u8>],
) -> Result<(CrocoStyleBackbone<B>, ApplyResult), ImportError> {
    load_yono_backbone_from_burnpack_part_bytes_with_progress(device, config, parts, |_| {})
}

pub fn load_yono_backbone_from_burnpack_part_bytes_with_progress<B: Backend, F>(
    device: &B::Device,
    config: CrocoStyleBackboneConfig,
    parts: &[Vec<u8>],
    mut progress: F,
) -> Result<(CrocoStyleBackbone<B>, ApplyResult), ImportError>
where
    F: FnMut(String),
{
    let mut model = CrocoStyleBackbone::new(device, config);
    let result = apply_burnpack_parts_bytes_with_progress(&mut model, parts, |index, total| {
        progress(format!("loading yono backbone part {index}/{total}"))
    })?;
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
    save_yono_head_record_bpk_with_precision(model, output_base, YonoWeightPrecision::F32)
}

pub fn save_yono_head_record_bpk_with_precision<B: Backend>(
    model: &YonoHeadPipeline<B>,
    output_base: &Path,
    precision: YonoWeightPrecision,
) -> Result<PathBuf, ImportError> {
    let base = normalize_extension(output_base, "bpk");
    match precision {
        YonoWeightPrecision::F32 => {
            let output = burnpack_path_for_precision(base.as_path(), YonoWeightPrecision::F32);
            let mut store = BurnpackStore::from_file(&output)
                .auto_extension(false)
                .overwrite(true);
            model
                .save_into(&mut store)
                .map_err(|err| ImportError::SaveBurnpack(err.to_string()))?;
            Ok(output)
        }
        YonoWeightPrecision::F16 => {
            let source = burnpack_path_for_precision(base.as_path(), YonoWeightPrecision::F32);
            let mut store = BurnpackStore::from_file(&source)
                .auto_extension(false)
                .overwrite(true);
            model
                .save_into(&mut store)
                .map_err(|err| ImportError::SaveBurnpack(err.to_string()))?;
            convert_burnpack_to_f16(source.as_path(), base.as_path())
        }
    }
}

pub fn save_yono_backbone_record_bpk<B: Backend>(
    model: &CrocoStyleBackbone<B>,
    output_base: &Path,
) -> Result<PathBuf, ImportError> {
    save_yono_backbone_record_bpk_with_precision(model, output_base, YonoWeightPrecision::F32)
}

pub fn save_yono_backbone_record_bpk_with_precision<B: Backend>(
    model: &CrocoStyleBackbone<B>,
    output_base: &Path,
    precision: YonoWeightPrecision,
) -> Result<PathBuf, ImportError> {
    let base = normalize_extension(output_base, "bpk");
    match precision {
        YonoWeightPrecision::F32 => {
            let output = burnpack_path_for_precision(base.as_path(), YonoWeightPrecision::F32);
            let mut store = BurnpackStore::from_file(&output)
                .auto_extension(false)
                .overwrite(true);
            model
                .save_into(&mut store)
                .map_err(|err| ImportError::SaveBurnpack(err.to_string()))?;
            Ok(output)
        }
        YonoWeightPrecision::F16 => {
            let source = burnpack_path_for_precision(base.as_path(), YonoWeightPrecision::F32);
            let mut store = BurnpackStore::from_file(&source)
                .auto_extension(false)
                .overwrite(true);
            model
                .save_into(&mut store)
                .map_err(|err| ImportError::SaveBurnpack(err.to_string()))?;
            convert_burnpack_to_f16(source.as_path(), base.as_path())
        }
    }
}

pub fn ensure_burnpack_parts(
    burnpack_path: &Path,
    max_part_size_mib: u64,
    overwrite: bool,
) -> Result<Option<BurnpackPartsReport>, ImportError> {
    write_burnpack_parts_for_wasm(burnpack_path, max_part_size_mib, overwrite)
        .map_err(ImportError::Parts)
}

pub fn convert_burnpack_to_f16(
    source_burnpack: &Path,
    output_base: &Path,
) -> Result<PathBuf, ImportError> {
    let output = burnpack_path_for_precision(
        normalize_extension(output_base, "bpk").as_path(),
        YonoWeightPrecision::F16,
    );
    convert_burnpack_float_precision(source_burnpack, output.as_path(), DType::F16)?;
    Ok(output)
}

fn convert_burnpack_float_precision(
    source_burnpack: &Path,
    output_burnpack: &Path,
    target_dtype: DType,
) -> Result<(), ImportError> {
    let mut source = fs::File::open(source_burnpack).map_err(|err| {
        ImportError::BurnpackConversion(format!(
            "failed to open source burnpack {}: {err}",
            source_burnpack.display()
        ))
    })?;
    let (version, mut metadata, data_start) = read_burnpack_metadata(&mut source, source_burnpack)?;

    let mut descriptors = metadata.tensors.into_iter().collect::<Vec<_>>();
    descriptors.sort_by_key(|(_, descriptor)| descriptor.data_offsets.0);
    let mut converted_descriptors = std::collections::BTreeMap::new();
    let mut converted_payloads = Vec::with_capacity(descriptors.len());
    let mut next_offset = 0u64;

    for (name, mut descriptor) in descriptors {
        let source_bytes = read_tensor_payload(
            &mut source,
            data_start,
            descriptor.data_offsets,
            source_burnpack,
            &name,
        )?;
        let converted_dtype = convert_float_dtype(descriptor.dtype, target_dtype);
        let converted_bytes = if converted_dtype == descriptor.dtype {
            source_bytes
        } else {
            let shape = descriptor
                .shape
                .iter()
                .map(|&dim| {
                    usize::try_from(dim).map_err(|_| {
                        ImportError::BurnpackConversion(format!(
                            "tensor `{name}` shape dimension overflow: {dim}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let source_data = TensorData::from_bytes_vec(source_bytes, shape, descriptor.dtype);
            let converted = convert_tensor_data_float_precision(source_data, target_dtype);
            converted.bytes.to_vec()
        };

        let tensor_len = u64::try_from(converted_bytes.len()).map_err(|_| {
            ImportError::BurnpackConversion(format!(
                "tensor `{name}` byte length overflow: {}",
                converted_bytes.len()
            ))
        })?;
        let end_offset = next_offset.checked_add(tensor_len).ok_or_else(|| {
            ImportError::BurnpackConversion(format!(
                "tensor `{name}` data offset overflow: {next_offset} + {tensor_len}"
            ))
        })?;
        descriptor.dtype = converted_dtype;
        descriptor.data_offsets = (next_offset, end_offset);
        next_offset = end_offset;
        converted_descriptors.insert(name, descriptor);
        converted_payloads.push(converted_bytes);
    }

    metadata.tensors = converted_descriptors;
    metadata.metadata.insert(
        "precision".to_string(),
        dtype_precision_label(target_dtype).to_string(),
    );
    write_burnpack_file(output_burnpack, version, &metadata, converted_payloads)?;
    Ok(())
}

fn convert_tensor_data_float_precision(data: TensorData, target_dtype: DType) -> TensorData {
    match target_dtype {
        DType::F16 => match data.dtype {
            DType::F64 | DType::F32 | DType::BF16 | DType::Flex32 => data.convert_dtype(DType::F16),
            _ => data,
        },
        DType::F32 => match data.dtype {
            DType::F16 | DType::BF16 | DType::F64 | DType::Flex32 => data.convert_dtype(DType::F32),
            _ => data,
        },
        _ => data,
    }
}

fn convert_float_dtype(source: DType, target_dtype: DType) -> DType {
    match target_dtype {
        DType::F16 => {
            if source.is_float() {
                DType::F16
            } else {
                source
            }
        }
        DType::F32 => {
            if source.is_float() {
                DType::F32
            } else {
                source
            }
        }
        _ => source,
    }
}

fn dtype_precision_label(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => "f16",
        DType::F32 => "f32",
        _ => "mixed",
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct RawBurnpackMetadata {
    tensors: std::collections::BTreeMap<String, RawTensorDescriptor>,
    #[serde(default)]
    metadata: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct RawTensorDescriptor {
    dtype: DType,
    shape: Vec<u64>,
    data_offsets: (u64, u64),
    #[serde(default, skip_serializing_if = "Option::is_none")]
    param_id: Option<u64>,
}

fn read_burnpack_metadata(
    source: &mut fs::File,
    source_path: &Path,
) -> Result<(u16, RawBurnpackMetadata, u64), ImportError> {
    source.seek(SeekFrom::Start(0)).map_err(|err| {
        ImportError::BurnpackConversion(format!("failed to seek {}: {err}", source_path.display()))
    })?;
    let mut header = [0u8; BURNPACK_HEADER_SIZE];
    source.read_exact(&mut header).map_err(|err| {
        ImportError::BurnpackConversion(format!(
            "failed to read burnpack header {}: {err}",
            source_path.display()
        ))
    })?;

    let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    if magic != BURNPACK_MAGIC_NUMBER {
        return Err(ImportError::BurnpackConversion(format!(
            "invalid burnpack magic in {}: expected {BURNPACK_MAGIC_NUMBER:#x}, got {magic:#x}",
            source_path.display()
        )));
    }
    let version = u16::from_le_bytes([header[4], header[5]]);
    let metadata_size = u32::from_le_bytes([header[6], header[7], header[8], header[9]]);
    let mut metadata_bytes = vec![0u8; metadata_size as usize];
    source.read_exact(&mut metadata_bytes).map_err(|err| {
        ImportError::BurnpackConversion(format!(
            "failed to read burnpack metadata {}: {err}",
            source_path.display()
        ))
    })?;
    let metadata = ciborium::de::from_reader(metadata_bytes.as_slice()).map_err(|err| {
        ImportError::BurnpackConversion(format!(
            "failed to parse burnpack metadata {}: {err}",
            source_path.display()
        ))
    })?;
    Ok((
        version,
        metadata,
        BURNPACK_HEADER_SIZE as u64 + metadata_size as u64,
    ))
}

fn read_tensor_payload(
    source: &mut fs::File,
    data_start: u64,
    data_offsets: (u64, u64),
    source_path: &Path,
    tensor_name: &str,
) -> Result<Vec<u8>, ImportError> {
    let (start, end) = data_offsets;
    if end < start {
        return Err(ImportError::BurnpackConversion(format!(
            "tensor `{tensor_name}` has invalid data offsets ({start}, {end}) in {}",
            source_path.display()
        )));
    }
    let len = end - start;
    let len_usize = usize::try_from(len).map_err(|_| {
        ImportError::BurnpackConversion(format!(
            "tensor `{tensor_name}` byte length overflow: {len}"
        ))
    })?;
    let seek_offset = data_start.checked_add(start).ok_or_else(|| {
        ImportError::BurnpackConversion(format!(
            "tensor `{tensor_name}` data offset overflow: {data_start} + {start}"
        ))
    })?;
    source.seek(SeekFrom::Start(seek_offset)).map_err(|err| {
        ImportError::BurnpackConversion(format!(
            "failed to seek tensor `{tensor_name}` in {}: {err}",
            source_path.display()
        ))
    })?;
    let mut bytes = vec![0u8; len_usize];
    source.read_exact(&mut bytes).map_err(|err| {
        ImportError::BurnpackConversion(format!(
            "failed to read tensor `{tensor_name}` bytes from {}: {err}",
            source_path.display()
        ))
    })?;
    Ok(bytes)
}

fn write_burnpack_file(
    output_path: &Path,
    version: u16,
    metadata: &RawBurnpackMetadata,
    payloads: Vec<Vec<u8>>,
) -> Result<(), ImportError> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            ImportError::BurnpackConversion(format!(
                "failed to create output directory {}: {err}",
                parent.display()
            ))
        })?;
    }

    let mut metadata_bytes = Vec::new();
    ciborium::ser::into_writer(metadata, &mut metadata_bytes).map_err(|err| {
        ImportError::BurnpackConversion(format!(
            "failed to serialize burnpack metadata for {}: {err}",
            output_path.display()
        ))
    })?;
    let metadata_size = u32::try_from(metadata_bytes.len()).map_err(|_| {
        ImportError::BurnpackConversion(format!(
            "burnpack metadata too large for {}: {} bytes",
            output_path.display(),
            metadata_bytes.len()
        ))
    })?;

    let mut out = fs::File::create(output_path).map_err(|err| {
        ImportError::BurnpackConversion(format!(
            "failed to create output burnpack {}: {err}",
            output_path.display()
        ))
    })?;
    let mut header = [0u8; BURNPACK_HEADER_SIZE];
    header[0..4].copy_from_slice(&BURNPACK_MAGIC_NUMBER.to_le_bytes());
    header[4..6].copy_from_slice(&version.to_le_bytes());
    header[6..10].copy_from_slice(&metadata_size.to_le_bytes());
    out.write_all(&header).map_err(|err| {
        ImportError::BurnpackConversion(format!(
            "failed to write burnpack header {}: {err}",
            output_path.display()
        ))
    })?;
    out.write_all(metadata_bytes.as_slice()).map_err(|err| {
        ImportError::BurnpackConversion(format!(
            "failed to write burnpack metadata {}: {err}",
            output_path.display()
        ))
    })?;
    for bytes in payloads {
        out.write_all(bytes.as_slice()).map_err(|err| {
            ImportError::BurnpackConversion(format!(
                "failed to write burnpack tensor bytes {}: {err}",
                output_path.display()
            ))
        })?;
    }
    out.flush().map_err(|err| {
        ImportError::BurnpackConversion(format!(
            "failed to flush output burnpack {}: {err}",
            output_path.display()
        ))
    })?;
    Ok(())
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

fn apply_burnpack_part_bytes<M, B: Backend>(
    model: &mut M,
    burnpack_bytes: Vec<u8>,
) -> Result<ApplyResult, String>
where
    M: Module<B>,
{
    let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(burnpack_bytes)))
        .allow_partial(true)
        .validate(true);
    model
        .load_from(&mut store)
        .map_err(|err| format!("{err:?}"))
}

fn apply_burnpack_parts_bytes_with_progress<M, B: Backend, F>(
    model: &mut M,
    parts: &[Vec<u8>],
    mut progress: F,
) -> Result<ApplyResult, ImportError>
where
    M: Module<B>,
    F: FnMut(usize, usize),
{
    let mut applied = BTreeSet::new();
    let total = parts.len();
    for (index, part) in parts.iter().enumerate() {
        progress(index + 1, total);
        let result = apply_burnpack_part_bytes(model, part.clone()).map_err(ImportError::Apply)?;
        for key in result.applied {
            applied.insert(key);
        }
    }
    Ok(ApplyResult {
        applied: applied.into_iter().collect(),
        skipped: Vec::new(),
        missing: Vec::new(),
        unused: Vec::new(),
        errors: Vec::new(),
    })
}

fn first_existing_or_first(paths: &[PathBuf]) -> &Path {
    paths
        .iter()
        .find(|path| path.exists())
        .map(PathBuf::as_path)
        .unwrap_or(paths[0].as_path())
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use burn::tensor::TensorData;

    use super::{
        convert_burnpack_to_f16, read_burnpack_metadata, read_tensor_payload, write_burnpack_file,
        RawBurnpackMetadata, RawTensorDescriptor,
    };

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}_{}_{}", std::process::id(), stamp))
    }

    #[test]
    fn convert_burnpack_to_f16_converts_only_float_tensors() {
        let root = unique_temp_dir("burn_yono_import_precision");
        std::fs::create_dir_all(&root).expect("create test directory");
        let source = root.join("weights.bpk");
        let output_base = root.join("weights.bpk");

        let float_values = [1.5f32, -2.25f32];
        let int_values = [7_i32, -3_i32];
        let float_bytes = float_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let int_bytes = int_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "float.weight".to_string(),
            RawTensorDescriptor {
                dtype: burn::tensor::DType::F32,
                shape: vec![2],
                data_offsets: (0, float_bytes.len() as u64),
                param_id: None,
            },
        );
        tensors.insert(
            "index.bias".to_string(),
            RawTensorDescriptor {
                dtype: burn::tensor::DType::I32,
                shape: vec![2],
                data_offsets: (
                    float_bytes.len() as u64,
                    (float_bytes.len() + int_bytes.len()) as u64,
                ),
                param_id: None,
            },
        );
        let mut metadata = BTreeMap::new();
        metadata.insert("format".to_string(), "burnpack".to_string());
        metadata.insert("producer".to_string(), "burn".to_string());
        let source_metadata = RawBurnpackMetadata { tensors, metadata };
        write_burnpack_file(
            source.as_path(),
            1,
            &source_metadata,
            vec![float_bytes.clone(), int_bytes.clone()],
        )
        .expect("write source burnpack");

        let output = convert_burnpack_to_f16(source.as_path(), output_base.as_path())
            .expect("convert source burnpack to f16");
        assert!(
            output
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with("_f16.bpk")),
            "output path should contain _f16 suffix"
        );

        let mut output_file = std::fs::File::open(&output).expect("open converted burnpack");
        let (version, converted_metadata, data_start) =
            read_burnpack_metadata(&mut output_file, output.as_path()).expect("read converted");
        assert_eq!(version, 1);
        assert_eq!(
            converted_metadata
                .metadata
                .get("precision")
                .map(String::as_str),
            Some("f16")
        );

        let float_descriptor = converted_metadata
            .tensors
            .get("float.weight")
            .expect("float descriptor");
        let int_descriptor = converted_metadata
            .tensors
            .get("index.bias")
            .expect("int descriptor");
        assert_eq!(float_descriptor.dtype, burn::tensor::DType::F16);
        assert_eq!(int_descriptor.dtype, burn::tensor::DType::I32);

        let float_payload = read_tensor_payload(
            &mut output_file,
            data_start,
            float_descriptor.data_offsets,
            output.as_path(),
            "float.weight",
        )
        .expect("read float payload");
        let float_roundtrip =
            TensorData::from_bytes_vec(float_payload, vec![2], burn::tensor::DType::F16)
                .convert_dtype(burn::tensor::DType::F32)
                .to_vec::<f32>()
                .expect("decode float payload");
        let max_abs_float = float_roundtrip
            .iter()
            .zip(float_values.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_float <= 1e-3,
            "unexpected f16 conversion error: {max_abs_float}"
        );

        let int_payload = read_tensor_payload(
            &mut output_file,
            data_start,
            int_descriptor.data_offsets,
            output.as_path(),
            "index.bias",
        )
        .expect("read int payload");
        let int_roundtrip =
            TensorData::from_bytes_vec(int_payload, vec![2], burn::tensor::DType::I32)
                .to_vec::<i32>()
                .expect("decode int payload");
        assert_eq!(int_roundtrip.as_slice(), int_values.as_slice());

        let _ = std::fs::remove_dir_all(root);
    }
}
