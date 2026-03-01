use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use burn_store::ApplyResult;
use ciborium::Value;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const ONE_MIB: u64 = 1024 * 1024;
const HEADER_SIZE: usize = 10;
const MAGIC_NUMBER: u32 = 0x4255_524E;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurnpackPartsManifest {
    #[serde(default = "default_manifest_version")]
    pub version: u32,
    #[serde(default)]
    pub source_file: String,
    #[serde(default)]
    pub source_modified_unix_ms: u64,
    #[serde(default)]
    pub total_bytes: u64,
    #[serde(default)]
    pub max_part_bytes: u64,
    #[serde(default)]
    pub parts: Vec<BurnpackPartEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurnpackPartEntry {
    pub path: String,
    #[serde(default)]
    pub bytes: u64,
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub tensors: usize,
}

#[derive(Debug, Clone)]
pub struct BurnpackPartsReport {
    pub manifest_path: PathBuf,
    pub part_paths: Vec<PathBuf>,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct RawBurnpackMetadata {
    tensors: BTreeMap<String, RawTensorDescriptor>,
    #[serde(default)]
    metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct RawTensorDescriptor {
    dtype: Value,
    shape: Vec<u64>,
    data_offsets: (u64, u64),
    #[serde(default, skip_serializing_if = "Option::is_none")]
    param_id: Option<u64>,
}

#[derive(Debug, Clone)]
struct TensorRecord {
    name: String,
    descriptor: RawTensorDescriptor,
}

const fn default_manifest_version() -> u32 {
    1
}

pub fn burnpack_parts_manifest_path(burnpack_path: &Path) -> PathBuf {
    let file_name = burnpack_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.bpk");
    burnpack_path.with_file_name(format!("{file_name}.parts.json"))
}

pub fn write_burnpack_parts_for_wasm(
    burnpack_path: &Path,
    max_part_size_mib: u64,
    overwrite: bool,
) -> Result<Option<BurnpackPartsReport>, String> {
    if !burnpack_path.exists() {
        return Err(format!(
            "burnpack does not exist for parting: {}",
            burnpack_path.display()
        ));
    }

    let max_part_bytes = max_part_size_mib
        .max(1)
        .checked_mul(ONE_MIB)
        .ok_or_else(|| "max part size overflow".to_string())?;

    let total_bytes = fs::metadata(burnpack_path)
        .map_err(|err| format!("failed to read {} metadata: {err}", burnpack_path.display()))?
        .len();
    let source_modified_unix_ms = file_modified_unix_ms(burnpack_path).unwrap_or(0);
    let manifest_path = burnpack_parts_manifest_path(burnpack_path);
    if manifest_path.exists()
        && !overwrite
        && manifest_has_all_parts(&manifest_path, Some(burnpack_path))
    {
        let manifest = read_parts_manifest(&manifest_path)?;
        let part_paths = manifest
            .parts
            .iter()
            .map(|entry| resolve_part_entry_path(&manifest_path, &entry.path))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Some(BurnpackPartsReport {
            manifest_path,
            part_paths,
            total_bytes: manifest.total_bytes,
        }));
    }

    if overwrite {
        cleanup_existing_parts(&manifest_path)?;
    }
    ensure_parent_dir(&manifest_path)?;

    let mut source = fs::File::open(burnpack_path)
        .map_err(|err| format!("failed to open burnpack {}: {err}", burnpack_path.display()))?;
    let (version, metadata_size, metadata) = read_burnpack_metadata(&mut source, burnpack_path)?;
    let data_start = HEADER_SIZE as u64 + metadata_size as u64;

    let mut tensor_records = metadata
        .tensors
        .iter()
        .map(|(name, descriptor)| TensorRecord {
            name: name.clone(),
            descriptor: descriptor.clone(),
        })
        .collect::<Vec<_>>();
    if tensor_records.is_empty() {
        return Err(format!(
            "burnpack '{}' contains no tensor descriptors",
            burnpack_path.display()
        ));
    }
    tensor_records.sort_by_key(|record| record.descriptor.data_offsets.0);
    let groups = split_tensor_records(tensor_records, max_part_bytes, &metadata.metadata);

    let source_file_name = burnpack_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid burnpack name '{}'", burnpack_path.display()))?;
    let mut part_entries = Vec::with_capacity(groups.len());
    let mut part_paths = Vec::with_capacity(groups.len());
    for (index, group) in groups.iter().enumerate() {
        let part_name = format!("{source_file_name}.part-{index:05}.bpk");
        let part_path = burnpack_path.with_file_name(&part_name);
        if part_path.exists() && overwrite {
            fs::remove_file(&part_path).map_err(|err| {
                format!(
                    "failed to replace stale burnpack part {}: {err}",
                    part_path.display()
                )
            })?;
        }

        write_burnpack_part(
            &mut source,
            &part_path,
            version,
            data_start,
            &metadata.metadata,
            group,
        )?;
        let bytes = fs::metadata(&part_path)
            .map_err(|err| {
                format!(
                    "failed to stat burnpack part {}: {err}",
                    part_path.display()
                )
            })?
            .len();
        let sha256 = sha256_file(&part_path)?;
        part_entries.push(BurnpackPartEntry {
            path: part_name,
            bytes,
            sha256,
            tensors: group.len(),
        });
        part_paths.push(part_path);
    }

    let manifest = BurnpackPartsManifest {
        version: default_manifest_version(),
        source_file: source_file_name.to_string(),
        source_modified_unix_ms,
        total_bytes,
        max_part_bytes,
        parts: part_entries,
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|err| format!("failed to serialize parts manifest: {err}"))?;
    fs::write(&manifest_path, manifest_json).map_err(|err| {
        format!(
            "failed to write burnpack parts manifest {}: {err}",
            manifest_path.display()
        )
    })?;

    Ok(Some(BurnpackPartsReport {
        manifest_path,
        part_paths,
        total_bytes,
    }))
}

pub fn load_model_from_burnpack_parts<M, Init, Apply>(
    burnpack_candidates: &[PathBuf],
    label: &str,
    verify_checksums: bool,
    mut init_model: Init,
    mut apply_part: Apply,
) -> Result<Option<(M, ApplyResult)>, String>
where
    Init: FnMut() -> M,
    Apply: FnMut(&mut M, Vec<u8>) -> Result<ApplyResult, String>,
{
    load_model_from_burnpack_parts_with_progress(
        burnpack_candidates,
        label,
        verify_checksums,
        &mut init_model,
        &mut apply_part,
        |_| {},
    )
}

pub fn load_model_from_burnpack_parts_with_progress<M, Init, Apply, Progress>(
    burnpack_candidates: &[PathBuf],
    label: &str,
    verify_checksums: bool,
    mut init_model: Init,
    mut apply_part: Apply,
    mut progress: Progress,
) -> Result<Option<(M, ApplyResult)>, String>
where
    Init: FnMut() -> M,
    Apply: FnMut(&mut M, Vec<u8>) -> Result<ApplyResult, String>,
    Progress: FnMut(String),
{
    let any_candidate_exists = burnpack_candidates
        .iter()
        .any(|candidate| candidate.exists());

    for candidate in burnpack_candidates {
        if any_candidate_exists && !candidate.exists() {
            continue;
        }
        let manifest_path = burnpack_parts_manifest_path(candidate);
        if !manifest_path.exists() {
            continue;
        }
        let manifest = read_parts_manifest(&manifest_path)?;
        if candidate.exists() && !manifest_matches_source_file(&manifest, candidate) {
            continue;
        }
        if manifest.parts.is_empty() {
            return Err(format!(
                "burnpack parts manifest {} contains no parts for {label}",
                manifest_path.display()
            ));
        }

        let mut model = init_model();
        let mut applied = BTreeSet::new();
        let total_parts = manifest.parts.len();
        for (index, part) in manifest.parts.iter().enumerate() {
            let part_path = resolve_part_entry_path(&manifest_path, &part.path)?;
            progress(format!(
                "loading {label} part {}/{}",
                index + 1,
                total_parts
            ));
            let bytes = fs::read(&part_path).map_err(|err| {
                format!(
                    "failed to read {} part {}: {err}",
                    label,
                    part_path.display()
                )
            })?;
            if part.bytes > 0 && bytes.len() as u64 != part.bytes {
                return Err(format!(
                    "{label} part {} expected {} bytes but found {}",
                    part_path.display(),
                    part.bytes,
                    bytes.len()
                ));
            }
            if verify_checksums && !part.sha256.trim().is_empty() {
                let actual_sha = sha256_bytes(bytes.as_slice());
                if !actual_sha.eq_ignore_ascii_case(part.sha256.trim()) {
                    return Err(format!(
                        "{label} part {} checksum mismatch: expected {}, got {}",
                        part_path.display(),
                        part.sha256.trim(),
                        actual_sha
                    ));
                }
            }
            let apply_result = apply_part(&mut model, bytes).map_err(|err| {
                format!(
                    "failed to apply {label} part {}/{} ({}): {err}",
                    index + 1,
                    manifest.parts.len(),
                    part_path.display()
                )
            })?;
            for key in apply_result.applied {
                applied.insert(key);
            }
        }
        progress(format!(
            "loaded {label} parts ({total_parts}/{total_parts})"
        ));

        return Ok(Some((
            model,
            ApplyResult {
                applied: applied.into_iter().collect(),
                skipped: Vec::new(),
                missing: Vec::new(),
                unused: Vec::new(),
                errors: Vec::new(),
            },
        )));
    }
    Ok(None)
}

pub fn read_parts_manifest(path: &Path) -> Result<BurnpackPartsManifest, String> {
    let bytes = fs::read(path).map_err(|err| {
        format!(
            "failed to read burnpack parts manifest {}: {err}",
            path.display()
        )
    })?;
    serde_json::from_slice(&bytes).map_err(|err| {
        format!(
            "failed to parse burnpack parts manifest {}: {err}",
            path.display()
        )
    })
}

pub fn resolve_part_entry_path(manifest_path: &Path, entry_path: &str) -> Result<PathBuf, String> {
    let entry_path = Path::new(entry_path);
    if entry_path.is_absolute() {
        return Ok(entry_path.to_path_buf());
    }
    manifest_path
        .parent()
        .map(|parent| parent.join(entry_path))
        .ok_or_else(|| format!("invalid manifest path '{}'", manifest_path.display()))
}

pub fn manifest_is_complete(manifest_path: &Path) -> Result<bool, String> {
    if !manifest_path.exists() {
        return Ok(false);
    }
    let manifest = match read_parts_manifest(manifest_path) {
        Ok(manifest) => manifest,
        Err(_) => return Ok(false),
    };
    if manifest.parts.is_empty() {
        return Ok(false);
    }
    for part in &manifest.parts {
        let path = resolve_part_entry_path(manifest_path, &part.path)?;
        if !part_matches_cache(&path, part)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn read_burnpack_metadata(
    source: &mut fs::File,
    burnpack_path: &Path,
) -> Result<(u16, u32, RawBurnpackMetadata), String> {
    source
        .seek(SeekFrom::Start(0))
        .map_err(|err| format!("failed to seek {}: {err}", burnpack_path.display()))?;
    let mut header = [0u8; HEADER_SIZE];
    source.read_exact(&mut header).map_err(|err| {
        format!(
            "failed to read burnpack header {}: {err}",
            burnpack_path.display()
        )
    })?;

    let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    if magic != MAGIC_NUMBER {
        return Err(format!(
            "invalid burnpack magic in {}: expected {MAGIC_NUMBER:#x}, found {magic:#x}",
            burnpack_path.display()
        ));
    }
    let version = u16::from_le_bytes([header[4], header[5]]);
    let metadata_size = u32::from_le_bytes([header[6], header[7], header[8], header[9]]);
    let mut metadata_bytes = vec![0u8; metadata_size as usize];
    source.read_exact(&mut metadata_bytes).map_err(|err| {
        format!(
            "failed to read burnpack metadata {}: {err}",
            burnpack_path.display()
        )
    })?;
    let metadata = ciborium::de::from_reader(metadata_bytes.as_slice()).map_err(|err| {
        format!(
            "failed to parse burnpack metadata {}: {err}",
            burnpack_path.display()
        )
    })?;
    Ok((version, metadata_size, metadata))
}

fn split_tensor_records(
    records: Vec<TensorRecord>,
    max_part_bytes: u64,
    source_metadata: &BTreeMap<String, String>,
) -> Vec<Vec<TensorRecord>> {
    let mut groups = Vec::new();
    let mut current_group = Vec::new();

    for record in records {
        let mut candidate_group = current_group.clone();
        candidate_group.push(record.clone());

        let candidate_bytes =
            estimate_part_total_bytes(candidate_group.as_slice(), source_metadata)
                .unwrap_or(u64::MAX);
        let would_exceed = !current_group.is_empty() && candidate_bytes > max_part_bytes;
        if would_exceed {
            groups.push(current_group);
            current_group = vec![record];
        } else {
            current_group = candidate_group;
        }
    }
    if !current_group.is_empty() {
        groups.push(current_group);
    }
    groups
}

fn estimate_part_total_bytes(
    records: &[TensorRecord],
    source_metadata: &BTreeMap<String, String>,
) -> Result<u64, String> {
    let mut tensors = BTreeMap::new();
    let mut payload_bytes = 0u64;
    for record in records {
        let tensor_bytes = record
            .descriptor
            .data_offsets
            .1
            .saturating_sub(record.descriptor.data_offsets.0);
        let mut descriptor = record.descriptor.clone();
        descriptor.data_offsets = (payload_bytes, payload_bytes.saturating_add(tensor_bytes));
        payload_bytes = descriptor.data_offsets.1;
        tensors.insert(record.name.clone(), descriptor);
    }

    let metadata = RawBurnpackMetadata {
        tensors,
        metadata: source_metadata.clone(),
    };
    let mut metadata_bytes = Vec::new();
    ciborium::ser::into_writer(&metadata, &mut metadata_bytes)
        .map_err(|err| format!("failed to estimate burnpack part metadata size: {err}"))?;
    Ok(HEADER_SIZE as u64 + metadata_bytes.len() as u64 + payload_bytes)
}

fn write_burnpack_part(
    source: &mut fs::File,
    destination: &Path,
    version: u16,
    data_start: u64,
    source_metadata: &BTreeMap<String, String>,
    records: &[TensorRecord],
) -> Result<(), String> {
    let mut tensors = BTreeMap::new();
    let mut next_offset = 0u64;
    for record in records {
        let tensor_bytes = record
            .descriptor
            .data_offsets
            .1
            .saturating_sub(record.descriptor.data_offsets.0);
        let mut descriptor = record.descriptor.clone();
        descriptor.data_offsets = (next_offset, next_offset.saturating_add(tensor_bytes));
        next_offset = descriptor.data_offsets.1;
        tensors.insert(record.name.clone(), descriptor);
    }

    let metadata = RawBurnpackMetadata {
        tensors,
        metadata: source_metadata.clone(),
    };
    let mut metadata_bytes = Vec::new();
    ciborium::ser::into_writer(&metadata, &mut metadata_bytes)
        .map_err(|err| format!("failed to serialize burnpack part metadata: {err}"))?;
    let metadata_size = u32::try_from(metadata_bytes.len())
        .map_err(|_| "burnpack part metadata size exceeds u32".to_string())?;

    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let mut out = fs::File::create(destination).map_err(|err| {
        format!(
            "failed to create burnpack part {}: {err}",
            destination.display()
        )
    })?;
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(&MAGIC_NUMBER.to_le_bytes());
    header[4..6].copy_from_slice(&version.to_le_bytes());
    header[6..10].copy_from_slice(&metadata_size.to_le_bytes());
    out.write_all(&header).map_err(|err| {
        format!(
            "failed to write part header {}: {err}",
            destination.display()
        )
    })?;
    out.write_all(&metadata_bytes).map_err(|err| {
        format!(
            "failed to write part metadata {}: {err}",
            destination.display()
        )
    })?;

    let mut copy_buffer = vec![0u8; 1024 * 1024];
    for record in records {
        let start = record.descriptor.data_offsets.0;
        let end = record.descriptor.data_offsets.1;
        let mut remaining = end.saturating_sub(start);
        source
            .seek(SeekFrom::Start(data_start.saturating_add(start)))
            .map_err(|err| format!("failed to seek source burnpack: {err}"))?;
        while remaining > 0 {
            let chunk = remaining.min(copy_buffer.len() as u64) as usize;
            source
                .read_exact(&mut copy_buffer[..chunk])
                .map_err(|err| format!("failed to read source burnpack tensor bytes: {err}"))?;
            out.write_all(&copy_buffer[..chunk])
                .map_err(|err| format!("failed to write burnpack part tensor bytes: {err}"))?;
            remaining -= chunk as u64;
        }
    }
    out.flush().map_err(|err| {
        format!(
            "failed to flush burnpack part {}: {err}",
            destination.display()
        )
    })?;
    Ok(())
}

fn ensure_parent_dir(path: &Path) -> Result<(), String> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    fs::create_dir_all(parent).map_err(|err| {
        format!(
            "failed to create parent directory {}: {err}",
            parent.display()
        )
    })
}

fn manifest_has_all_parts(path: &Path, source_burnpack_path: Option<&Path>) -> bool {
    let Ok(manifest) = read_parts_manifest(path) else {
        return false;
    };
    if manifest.parts.is_empty() {
        return false;
    }
    if let Some(source_burnpack_path) = source_burnpack_path {
        if source_burnpack_path.exists()
            && !manifest_matches_source_file(&manifest, source_burnpack_path)
        {
            return false;
        }
    }
    manifest.parts.iter().all(|entry| {
        resolve_part_entry_path(path, &entry.path).is_ok_and(|part| {
            if !part.exists() {
                return false;
            }
            if entry.bytes == 0 {
                return true;
            }
            fs::metadata(&part)
                .map(|metadata| metadata.len() == entry.bytes)
                .unwrap_or(false)
        })
    })
}

fn manifest_matches_source_file(manifest: &BurnpackPartsManifest, source_path: &Path) -> bool {
    if !source_path.exists() {
        return true;
    }

    let Some(source_file_name) = source_path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    if !manifest.source_file.is_empty() && manifest.source_file != source_file_name {
        return false;
    }

    let actual_bytes = match fs::metadata(source_path) {
        Ok(metadata) => metadata.len(),
        Err(_) => return false,
    };
    if manifest.total_bytes > 0 && manifest.total_bytes != actual_bytes {
        return false;
    }

    if manifest.source_modified_unix_ms == 0 {
        return false;
    }
    let Some(actual_modified_unix_ms) = file_modified_unix_ms(source_path) else {
        return false;
    };
    manifest.source_modified_unix_ms == actual_modified_unix_ms
}

fn file_modified_unix_ms(path: &Path) -> Option<u64> {
    let metadata = fs::metadata(path).ok()?;
    let modified = metadata.modified().ok()?;
    let duration = modified.duration_since(UNIX_EPOCH).ok()?;
    Some(duration.as_millis().min(u64::MAX as u128) as u64)
}

fn cleanup_existing_parts(manifest_path: &Path) -> Result<(), String> {
    let manifest = match read_parts_manifest(manifest_path) {
        Ok(manifest) => manifest,
        Err(_) => return Ok(()),
    };
    for entry in &manifest.parts {
        let path = resolve_part_entry_path(manifest_path, &entry.path)?;
        if path.exists() {
            fs::remove_file(&path).map_err(|err| {
                format!(
                    "failed to remove old burnpack part {}: {err}",
                    path.display()
                )
            })?;
        }
    }
    Ok(())
}

fn part_matches_cache(path: &Path, part: &BurnpackPartEntry) -> Result<bool, String> {
    if !path.exists() {
        return Ok(false);
    }
    if part.bytes == 0 {
        return Ok(true);
    }
    let bytes = fs::metadata(path)
        .map_err(|err| format!("failed to read part metadata {}: {err}", path.display()))?
        .len();
    Ok(bytes == part.bytes)
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path)
        .map_err(|err| format!("failed to read {} for sha256: {err}", path.display()))?;
    Ok(sha256_bytes(bytes.as_slice()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    format!("{digest:x}")
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use burn_store::ApplyResult;

    use super::{
        burnpack_parts_manifest_path, load_model_from_burnpack_parts, manifest_is_complete,
        BurnpackPartEntry, BurnpackPartsManifest,
    };

    fn unique_tmp_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("burn_yono_parts_test_{nanos}"))
    }

    #[test]
    fn manifest_complete_requires_all_parts() {
        let root = unique_tmp_dir();
        fs::create_dir_all(&root).expect("create root");
        let burnpack = root.join("model.bpk");
        let manifest_path = burnpack_parts_manifest_path(&burnpack);

        let part0 = root.join("model.bpk.part-00000.bpk");
        let part1 = root.join("model.bpk.part-00001.bpk");
        fs::write(&part0, vec![1_u8, 2, 3]).expect("write part0");
        fs::write(&part1, vec![4_u8, 5]).expect("write part1");

        let manifest = BurnpackPartsManifest {
            version: 1,
            source_file: "model.bpk".to_string(),
            source_modified_unix_ms: 0,
            total_bytes: 0,
            max_part_bytes: 0,
            parts: vec![
                BurnpackPartEntry {
                    path: "model.bpk.part-00000.bpk".to_string(),
                    bytes: 3,
                    sha256: String::new(),
                    tensors: 1,
                },
                BurnpackPartEntry {
                    path: "model.bpk.part-00001.bpk".to_string(),
                    bytes: 2,
                    sha256: String::new(),
                    tensors: 1,
                },
            ],
        };
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");

        assert!(
            manifest_is_complete(&manifest_path).expect("complete check"),
            "expected manifest to be complete"
        );
        fs::remove_file(&part1).expect("remove part1");
        assert!(
            !manifest_is_complete(&manifest_path).expect("incomplete check"),
            "expected manifest to be incomplete"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn loads_model_from_parts_manifest() {
        let root = unique_tmp_dir();
        fs::create_dir_all(&root).expect("create root");
        let burnpack = root.join("weights.bpk");
        let manifest_path = burnpack_parts_manifest_path(&burnpack);

        let part0 = root.join("weights.bpk.part-00000.bpk");
        let part1 = root.join("weights.bpk.part-00001.bpk");
        fs::write(&part0, b"abc").expect("write part0");
        fs::write(&part1, b"de").expect("write part1");

        let manifest = BurnpackPartsManifest {
            version: 1,
            source_file: "weights.bpk".to_string(),
            source_modified_unix_ms: 0,
            total_bytes: 0,
            max_part_bytes: 0,
            parts: vec![
                BurnpackPartEntry {
                    path: "weights.bpk.part-00000.bpk".to_string(),
                    bytes: 3,
                    sha256: String::new(),
                    tensors: 1,
                },
                BurnpackPartEntry {
                    path: "weights.bpk.part-00001.bpk".to_string(),
                    bytes: 2,
                    sha256: String::new(),
                    tensors: 1,
                },
            ],
        };
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");

        let loaded = load_model_from_burnpack_parts(
            &[burnpack],
            "test-model",
            false,
            Vec::<u8>::new,
            |model, part| {
                model.extend_from_slice(part.as_slice());
                Ok(ApplyResult {
                    applied: vec![format!("part_{}", part.len())],
                    skipped: Vec::new(),
                    missing: Vec::new(),
                    unused: Vec::new(),
                    errors: Vec::new(),
                })
            },
        )
        .expect("load should succeed")
        .expect("parts loader should return a model");

        assert_eq!(loaded.0, b"abcde".to_vec());
        assert_eq!(loaded.1.applied.len(), 2);

        let _ = fs::remove_dir_all(root);
    }
}
