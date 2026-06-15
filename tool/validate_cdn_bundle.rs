#![recursion_limit = "512"]

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use burn::prelude::*;
use burn_reconstruction::{
    backend::default_device, ComponentLoadReport, ImageToGaussianPipeline, PipelineConfig,
    PipelineInputImage, PipelineModel, PipelineQuality, PipelineRunWithCameras, PipelineWeights,
    YonoWeightFormat, YonoWeightPrecision, YonoWeights, ZipSplatWeightFormat,
    ZipSplatWeightPrecision, ZipSplatWeights,
};
use burn_yono::parts::BurnpackPartsManifest;
use clap::{Parser, ValueEnum};
use sha2::{Digest, Sha256};

type BackendImpl = burn_reconstruction::backend::BackendImpl;

#[derive(Debug, Clone)]
struct LoadedImage {
    name: String,
    bytes: Vec<u8>,
}

#[derive(Debug)]
struct FieldSnapshot {
    name: &'static str,
    shape: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Debug)]
struct RunSnapshot {
    fields: Vec<FieldSnapshot>,
}

#[derive(Debug, Clone, Copy)]
struct DiffStats {
    mean_abs: f32,
    max_abs: f32,
    mse: f32,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModelArg {
    Yono,
    Zipsplat,
}

#[derive(Debug, Parser)]
#[command(about = "Validate hosted CDN model shards and compare CDN-loaded inference numerically")]
struct Args {
    #[arg(long, value_enum, default_value_t = ModelArg::Zipsplat)]
    model: ModelArg,

    #[arg(
        long,
        default_value = "https://aberration.technology/model",
        help = "Base URL that contains model-specific folders"
    )]
    model_base_url: String,

    #[arg(long, default_value = "yono")]
    yono_remote_root: String,

    #[arg(long, default_value = "yono_backbone_f16.bpk")]
    yono_backbone_burnpack: String,

    #[arg(long, default_value = "yono_head_f16.bpk")]
    yono_head_burnpack: String,

    #[arg(long)]
    local_yono_backbone_weights: Option<PathBuf>,

    #[arg(long)]
    local_yono_head_weights: Option<PathBuf>,

    #[arg(long, default_value = "zipsplat")]
    zipsplat_remote_root: String,

    #[arg(long, default_value = "zipsplat_f16.bpk")]
    zipsplat_burnpack: String,

    #[arg(long)]
    local_zipsplat_weights: Option<PathBuf>,

    #[arg(long, num_args = 1..)]
    image: Vec<PathBuf>,

    #[arg(long)]
    image_size: Option<usize>,

    #[arg(long, default_value_t = PipelineQuality::Compact.default_zipsplat_r())]
    zipsplat_r: usize,

    #[arg(long)]
    shard_cache_dir: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    clear_shard_cache: bool,

    #[arg(long, default_value_t = true)]
    compare_local: bool,

    #[arg(long, default_value_t = 1e-5)]
    max_mean_abs: f32,

    #[arg(long, default_value_t = 5e-4)]
    max_max_abs: f32,

    #[arg(long, default_value_t = 1e-7)]
    max_mse: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let model = match args.model {
        ModelArg::Yono => PipelineModel::Yono,
        ModelArg::Zipsplat => PipelineModel::ZipSplat,
    };
    let capabilities = model.capabilities();
    let image_size = args
        .image_size
        .unwrap_or_else(|| capabilities.default_image_size);
    if image_size % capabilities.image_size_multiple != 0 {
        return Err(format!(
            "--image-size must be divisible by {}, got {}",
            capabilities.image_size_multiple, image_size
        )
        .into());
    }

    let images = load_images(resolve_images(args.image.as_slice()).as_slice())?;
    if images.len() < capabilities.min_views {
        return Err(format!(
            "{} expects at least {} images, got {}",
            model.display_name(),
            capabilities.min_views,
            images.len()
        )
        .into());
    }

    let cache_dir = args
        .shard_cache_dir
        .clone()
        .unwrap_or_else(|| default_shard_cache_dir(model));
    if args.clear_shard_cache && cache_dir.exists() {
        fs::remove_dir_all(&cache_dir)?;
        println!("[cdn] cleared shard cache {}", cache_dir.display());
    }
    fs::create_dir_all(&cache_dir)?;

    let cfg = PipelineConfig {
        model,
        quality: PipelineQuality::Balanced,
        image_size,
        zipsplat_r: args.zipsplat_r,
    };
    let cdn_snapshot = match model {
        PipelineModel::Yono => {
            let backbone_parts = load_cdn_parts_bundle(
                args.model_base_url.as_str(),
                args.yono_remote_root.as_str(),
                args.yono_backbone_burnpack.as_str(),
                cache_dir.join("backbone").as_path(),
            )?;
            let head_parts = load_cdn_parts_bundle(
                args.model_base_url.as_str(),
                args.yono_remote_root.as_str(),
                args.yono_head_burnpack.as_str(),
                cache_dir.join("head").as_path(),
            )?;
            run_yono_cdn_pipeline(cfg.clone(), backbone_parts, head_parts, images.as_slice())?
        }
        PipelineModel::ZipSplat => {
            let parts = load_cdn_parts_bundle(
                args.model_base_url.as_str(),
                args.zipsplat_remote_root.as_str(),
                args.zipsplat_burnpack.as_str(),
                cache_dir.as_path(),
            )?;
            run_zipsplat_cdn_pipeline(cfg.clone(), parts, images.as_slice())?
        }
    };

    if args.compare_local {
        let local_snapshot = match model {
            PipelineModel::Yono => {
                let backbone = args
                    .local_yono_backbone_weights
                    .clone()
                    .unwrap_or_else(default_local_yono_backbone_weights);
                let head = args
                    .local_yono_head_weights
                    .clone()
                    .unwrap_or_else(default_local_yono_head_weights);
                run_yono_local_pipeline(cfg, backbone.as_path(), head.as_path(), images.as_slice())?
            }
            PipelineModel::ZipSplat => {
                let local_path = args
                    .local_zipsplat_weights
                    .clone()
                    .unwrap_or_else(default_local_zipsplat_weights);
                run_zipsplat_local_pipeline(cfg, local_path.as_path(), images.as_slice())?
            }
        };
        compare_snapshots(
            &cdn_snapshot,
            &local_snapshot,
            args.max_mean_abs,
            args.max_max_abs,
            args.max_mse,
        )?;
    }

    println!("[cdn] PASS");
    Ok(())
}

fn run_yono_cdn_pipeline(
    cfg: PipelineConfig,
    backbone_parts: Vec<Vec<u8>>,
    head_parts: Vec<Vec<u8>>,
    images: &[LoadedImage],
) -> Result<RunSnapshot, Box<dyn std::error::Error>> {
    let load_start = Instant::now();
    let (pipeline, load_report) = ImageToGaussianPipeline::load_from_yono_parts_with_progress(
        default_device(),
        cfg,
        backbone_parts.as_slice(),
        head_parts.as_slice(),
        |message| println!("[cdn] {message}"),
    )?;
    drop(backbone_parts);
    drop(head_parts);
    report_apply_summary("cdn.yono.backbone", &load_report.backbone);
    report_apply_summary("cdn.yono.head", &load_report.head);
    println!(
        "[cdn] model_load_ms={:.3}",
        load_start.elapsed().as_secs_f64() * 1000.0
    );
    run_pipeline("cdn", &pipeline, images)
}

fn run_zipsplat_cdn_pipeline(
    cfg: PipelineConfig,
    parts: Vec<Vec<u8>>,
    images: &[LoadedImage],
) -> Result<RunSnapshot, Box<dyn std::error::Error>> {
    let load_start = Instant::now();
    let (pipeline, load_report) = ImageToGaussianPipeline::load_from_zipsplat_parts_with_progress(
        default_device(),
        cfg,
        parts.as_slice(),
        |message| println!("[cdn] {message}"),
    )?;
    drop(parts);
    report_apply_summary("cdn.zipsplat", &load_report.backbone);
    println!(
        "[cdn] model_load_ms={:.3}",
        load_start.elapsed().as_secs_f64() * 1000.0
    );
    run_pipeline("cdn", &pipeline, images)
}

fn run_yono_local_pipeline(
    cfg: PipelineConfig,
    backbone_path: &Path,
    head_path: &Path,
    images: &[LoadedImage],
) -> Result<RunSnapshot, Box<dyn std::error::Error>> {
    let weights = PipelineWeights::from_yono(
        YonoWeights::new(backbone_path, head_path)
            .with_format(YonoWeightFormat::Burnpack)
            .with_precision(YonoWeightPrecision::F16),
    );
    let load_start = Instant::now();
    let (pipeline, load_report) = ImageToGaussianPipeline::load(default_device(), cfg, weights)?;
    report_apply_summary("local.yono.backbone", &load_report.backbone);
    report_apply_summary("local.yono.head", &load_report.head);
    println!(
        "[local] backbone={} head={} model_load_ms={:.3}",
        backbone_path.display(),
        head_path.display(),
        load_start.elapsed().as_secs_f64() * 1000.0
    );
    run_pipeline("local", &pipeline, images)
}

fn run_zipsplat_local_pipeline(
    cfg: PipelineConfig,
    weights_path: &Path,
    images: &[LoadedImage],
) -> Result<RunSnapshot, Box<dyn std::error::Error>> {
    let weights = PipelineWeights::from_zipsplat(
        ZipSplatWeights::new(weights_path)
            .with_format(ZipSplatWeightFormat::Burnpack)
            .with_precision(ZipSplatWeightPrecision::F16),
    );
    let load_start = Instant::now();
    let (pipeline, load_report) = ImageToGaussianPipeline::load(default_device(), cfg, weights)?;
    report_apply_summary("local.zipsplat", &load_report.backbone);
    println!(
        "[local] weights={} model_load_ms={:.3}",
        weights_path.display(),
        load_start.elapsed().as_secs_f64() * 1000.0
    );
    run_pipeline("local", &pipeline, images)
}

fn run_pipeline(
    label: &str,
    pipeline: &ImageToGaussianPipeline,
    images: &[LoadedImage],
) -> Result<RunSnapshot, Box<dyn std::error::Error>> {
    let inputs = to_input_images(images);
    let wall_start = Instant::now();
    let output = pipeline.run_image_bytes_timed_with_cameras(inputs.as_slice(), true)?;
    println!(
        "[{label}] inference_ms={:.3} forward_ms={:.3} image_load_ms={:.3} gaussians={}",
        wall_start.elapsed().as_secs_f64() * 1000.0,
        output.timings.total.as_secs_f64() * 1000.0,
        output.timings.image_load.as_secs_f64() * 1000.0,
        output.gaussians.means.shape().dims::<3>()[1]
    );
    snapshot_output(output)
}

fn snapshot_output(
    output: PipelineRunWithCameras,
) -> Result<RunSnapshot, Box<dyn std::error::Error>> {
    let means_shape = output.gaussians.means.shape().dims::<3>().to_vec();
    let covariances_shape = output.gaussians.covariances.shape().dims::<4>().to_vec();
    let harmonics_shape = output.gaussians.harmonics.shape().dims::<4>().to_vec();
    let opacities_shape = output.gaussians.opacities.shape().dims::<2>().to_vec();
    let rotations_shape = output.gaussians.rotations.shape().dims::<3>().to_vec();
    let scales_shape = output.gaussians.scales.shape().dims::<3>().to_vec();
    let camera_shape = vec![output.camera_poses.len(), 4, 4];

    Ok(RunSnapshot {
        fields: vec![
            FieldSnapshot {
                name: "means",
                shape: means_shape,
                values: tensor_to_vec(output.gaussians.means)?,
            },
            FieldSnapshot {
                name: "covariances",
                shape: covariances_shape,
                values: tensor_to_vec(output.gaussians.covariances)?,
            },
            FieldSnapshot {
                name: "harmonics",
                shape: harmonics_shape,
                values: tensor_to_vec(output.gaussians.harmonics)?,
            },
            FieldSnapshot {
                name: "opacities",
                shape: opacities_shape,
                values: tensor_to_vec(output.gaussians.opacities)?,
            },
            FieldSnapshot {
                name: "rotations",
                shape: rotations_shape,
                values: tensor_to_vec(output.gaussians.rotations)?,
            },
            FieldSnapshot {
                name: "scales",
                shape: scales_shape,
                values: tensor_to_vec(output.gaussians.scales)?,
            },
            FieldSnapshot {
                name: "camera_poses",
                shape: camera_shape,
                values: flatten_camera_poses(output.camera_poses.as_slice()),
            },
        ],
    })
}

fn compare_snapshots(
    actual: &RunSnapshot,
    expected: &RunSnapshot,
    max_mean_abs: f32,
    max_max_abs: f32,
    max_mse: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if actual.fields.len() != expected.fields.len() {
        return Err(format!(
            "field count mismatch: actual={} expected={}",
            actual.fields.len(),
            expected.fields.len()
        )
        .into());
    }
    for (actual_field, expected_field) in actual.fields.iter().zip(expected.fields.iter()) {
        if actual_field.name != expected_field.name {
            return Err(format!(
                "field name mismatch: actual={} expected={}",
                actual_field.name, expected_field.name
            )
            .into());
        }
        if actual_field.shape != expected_field.shape {
            return Err(format!(
                "{} shape mismatch: actual={:?} expected={:?}",
                actual_field.name, actual_field.shape, expected_field.shape
            )
            .into());
        }
        let stats = diff_stats(
            actual_field.values.as_slice(),
            expected_field.values.as_slice(),
        )?;
        println!(
            "[cdn] compare.{} shape={:?} mean_abs={:.9} max_abs={:.9} mse={:.9}",
            actual_field.name, actual_field.shape, stats.mean_abs, stats.max_abs, stats.mse
        );
        if stats.mean_abs > max_mean_abs || stats.max_abs > max_max_abs || stats.mse > max_mse {
            return Err(format!(
                "{} mismatch: mean_abs={:.9} max_abs={:.9} mse={:.9} bounds(mean_abs<={} max_abs<={} mse<={})",
                actual_field.name,
                stats.mean_abs,
                stats.max_abs,
                stats.mse,
                max_mean_abs,
                max_max_abs,
                max_mse
            )
            .into());
        }
    }
    Ok(())
}

fn load_cdn_parts_bundle(
    model_base_url: &str,
    remote_root: &str,
    burnpack_file: &str,
    cache_dir: &Path,
) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    fs::create_dir_all(cache_dir)?;
    let manifest_url = join_url(
        join_url(model_base_url, remote_root).as_str(),
        format!("{burnpack_file}.parts.json").as_str(),
    );
    let manifest_bytes = fetch_bytes(manifest_url.as_str())?;
    let manifest: BurnpackPartsManifest = serde_json::from_slice(manifest_bytes.as_slice())?;
    if manifest.parts.is_empty() {
        return Err(format!("parts manifest is empty: {manifest_url}").into());
    }
    println!(
        "[cdn] manifest={} parts={} total_bytes={} source_file={}",
        manifest_url,
        manifest.parts.len(),
        manifest.total_bytes,
        manifest.source_file
    );

    load_cdn_parts(manifest_url.as_str(), &manifest, cache_dir, true)
}

fn load_cdn_parts(
    manifest_url: &str,
    manifest: &BurnpackPartsManifest,
    cache_dir: &Path,
    verify_sha: bool,
) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let mut parts = Vec::with_capacity(manifest.parts.len());
    let mut cached = 0usize;
    let mut downloaded = 0usize;
    let mut total_bytes = 0u64;

    for (index, part) in manifest.parts.iter().enumerate() {
        let url = resolve_manifest_entry_url(manifest_url, part.path.as_str());
        let cache_path = cache_dir.join(cache_file_name(part.path.as_str())?);
        let mut bytes = match read_valid_cached_part(cache_path.as_path(), part.bytes, &part.sha256)
        {
            Ok(Some(value)) => {
                cached += 1;
                value
            }
            Ok(None) => {
                let value = fetch_bytes(url.as_str())?;
                fs::write(&cache_path, value.as_slice())?;
                downloaded += 1;
                value
            }
            Err(err) => {
                eprintln!(
                    "[cdn] invalid cached part {}; re-downloading ({err})",
                    cache_path.display()
                );
                let value = fetch_bytes(url.as_str())?;
                fs::write(&cache_path, value.as_slice())?;
                downloaded += 1;
                value
            }
        };

        if part.bytes > 0 && bytes.len() as u64 != part.bytes {
            return Err(format!(
                "{} expected {} bytes, got {}",
                part.path,
                part.bytes,
                bytes.len()
            )
            .into());
        }
        if verify_sha && !part.sha256.trim().is_empty() {
            let actual = sha256_hex(bytes.as_slice());
            if !actual.eq_ignore_ascii_case(part.sha256.trim()) {
                return Err(format!(
                    "{} sha256 mismatch: expected {}, got {}",
                    part.path,
                    part.sha256.trim(),
                    actual
                )
                .into());
            }
        }
        total_bytes = total_bytes.saturating_add(bytes.len() as u64);
        println!(
            "[cdn] part {}/{} {} bytes={} sha256={}",
            index + 1,
            manifest.parts.len(),
            if cache_path.exists() { "ok" } else { "missing" },
            bytes.len(),
            if part.sha256.trim().is_empty() {
                "<absent>"
            } else {
                part.sha256.trim()
            }
        );
        parts.push(std::mem::take(&mut bytes));
    }

    println!(
        "[cdn] shard_cache={} cached={} downloaded={} bytes={}",
        cache_dir.display(),
        cached,
        downloaded,
        total_bytes
    );
    Ok(parts)
}

fn read_valid_cached_part(
    path: &Path,
    expected_bytes: u64,
    expected_sha: &str,
) -> Result<Option<Vec<u8>>, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(path)?;
    if expected_bytes > 0 && bytes.len() as u64 != expected_bytes {
        return Err(format!(
            "byte count mismatch for {}: expected {}, got {}",
            path.display(),
            expected_bytes,
            bytes.len()
        )
        .into());
    }
    if !expected_sha.trim().is_empty() {
        let actual = sha256_hex(bytes.as_slice());
        if !actual.eq_ignore_ascii_case(expected_sha.trim()) {
            return Err(format!(
                "sha256 mismatch for {}: expected {}, got {}",
                path.display(),
                expected_sha.trim(),
                actual
            )
            .into());
        }
    }
    Ok(Some(bytes))
}

fn fetch_bytes(url: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(30))
        .timeout_read(Duration::from_secs(120))
        .timeout_write(Duration::from_secs(30))
        .build();
    let response = agent.get(url).call().map_err(|err| match err {
        ureq::Error::Status(code, response) => {
            format!("{url}: HTTP {code} ({})", response.status_text())
        }
        ureq::Error::Transport(transport) => format!("{url}: transport error: {transport}"),
    })?;
    let expected_len = response
        .header("Content-Length")
        .and_then(|value| value.parse::<usize>().ok());
    let mut reader = response.into_reader();
    let mut out = Vec::with_capacity(expected_len.unwrap_or(0));
    reader.read_to_end(&mut out)?;
    if out.is_empty() {
        return Err(format!("{url}: empty response body").into());
    }
    if let Some(expected) = expected_len {
        if expected != out.len() {
            return Err(format!(
                "{url}: content-length mismatch: expected {expected}, got {}",
                out.len()
            )
            .into());
        }
    }
    Ok(out)
}

fn load_images(paths: &[PathBuf]) -> Result<Vec<LoadedImage>, Box<dyn std::error::Error>> {
    let mut out = Vec::with_capacity(paths.len());
    for path in paths {
        out.push(LoadedImage {
            name: image_name(path),
            bytes: fs::read(path)?,
        });
    }
    Ok(out)
}

fn resolve_images(paths: &[PathBuf]) -> Vec<PathBuf> {
    if !paths.is_empty() {
        return paths.to_vec();
    }
    vec![
        workspace_path("assets/images/re10k/0.png"),
        workspace_path("assets/images/re10k/1.png"),
        workspace_path("assets/images/re10k/2.png"),
    ]
}

fn to_input_images(images: &[LoadedImage]) -> Vec<PipelineInputImage<'_>> {
    images
        .iter()
        .map(|image| PipelineInputImage {
            name: image.name.as_str(),
            bytes: image.bytes.as_slice(),
        })
        .collect()
}

fn image_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| path.display().to_string())
}

fn workspace_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn default_shard_cache_dir(model: PipelineModel) -> PathBuf {
    workspace_path(format!("target/cdn-validate/{}-shards", model.id()).as_str())
}

fn default_local_yono_backbone_weights() -> PathBuf {
    workspace_path("assets/models/yono_backbone.bpk")
}

fn default_local_yono_head_weights() -> PathBuf {
    workspace_path("assets/models/yono_head.bpk")
}

fn default_local_zipsplat_weights() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".burn_reconstruction/models/zipsplat/zipsplat.bpk")
}

fn cache_file_name(path: &str) -> Result<&str, Box<dyn std::error::Error>> {
    Path::new(path)
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("invalid part path in manifest: {path}").into())
}

fn join_url(root: &str, rel: &str) -> String {
    let mut out = root.trim_end_matches('/').to_string();
    out.push('/');
    out.push_str(rel.trim_start_matches('/'));
    out
}

fn resolve_manifest_entry_url(manifest_url: &str, entry_url: &str) -> String {
    if entry_url.contains("://") || entry_url.starts_with('/') {
        return entry_url.to_string();
    }
    let normalized = entry_url.replace('\\', "/");
    if let Some((parent, _)) = manifest_url.rsplit_once('/') {
        return format!("{}/{}", parent.trim_end_matches('/'), normalized);
    }
    normalized
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn tensor_to_vec<const D: usize>(
    tensor: Tensor<BackendImpl, D>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| format!("failed tensor readback: {err:?}").into())
}

fn flatten_camera_poses(poses: &[[[f32; 4]; 4]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(poses.len() * 16);
    for pose in poses {
        for row in pose {
            out.extend_from_slice(row);
        }
    }
    out
}

fn diff_stats(lhs: &[f32], rhs: &[f32]) -> Result<DiffStats, Box<dyn std::error::Error>> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "length mismatch in diff stats: lhs={} rhs={}",
            lhs.len(),
            rhs.len()
        )
        .into());
    }
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&left, &right) in lhs.iter().zip(rhs.iter()) {
        let abs = (left - right).abs();
        sum_abs += abs as f64;
        sum_sq += (abs as f64) * (abs as f64);
        max_abs = max_abs.max(abs);
    }
    let len = lhs.len().max(1) as f64;
    Ok(DiffStats {
        mean_abs: (sum_abs / len) as f32,
        max_abs,
        mse: (sum_sq / len) as f32,
    })
}

fn report_apply_summary(name: &str, summary: &ComponentLoadReport) {
    println!(
        "[cdn] load.{name}: applied={} missing={} unused={} skipped={}",
        summary.applied,
        summary.missing.len(),
        summary.unused.len(),
        summary.skipped.len()
    );
}
