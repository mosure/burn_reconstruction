#![recursion_limit = "512"]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use burn::prelude::*;
use burn_reconstruction::{
    backend::default_device, ComponentLoadReport, ForwardTimings, ImageToGaussianPipeline,
    PipelineConfig, PipelineInputImage, PipelineModel, PipelineQuality, PipelineWeights,
    YonoWeightFormat, YonoWeightPrecision,
};
use clap::{ArgAction, Parser};

type BackendImpl = burn_reconstruction::backend::BackendImpl;

#[derive(Debug, Clone)]
struct LoadedImage {
    name: String,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
struct TimingAverages {
    image_load: Duration,
    backbone: Duration,
    head: Duration,
    forward_total: Duration,
    wall: Duration,
}

#[derive(Debug, Clone, Copy)]
struct DiffStats {
    mean_abs: f32,
    max_abs: f32,
}

#[derive(Debug, Parser)]
#[command(
    about = "Benchmark burn_reconstruction model init + inference scaling and validate determinism"
)]
struct Args {
    #[arg(long, required = true, num_args = 2..)]
    image: Vec<PathBuf>,

    #[arg(long, default_value_t = 224)]
    image_size: usize,

    #[arg(long, default_value_t = 1)]
    warmup_iters: usize,

    #[arg(long, default_value_t = 3)]
    bench_iters: usize,

    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    synchronize: bool,

    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    check_determinism: bool,

    #[arg(long, default_value_t = 1e-5)]
    max_determinism_mean_abs: f32,

    #[arg(long, default_value_t = 5e-4)]
    max_determinism_max_abs: f32,

    #[arg(long, value_delimiter = ',', num_args = 1..)]
    view_counts: Option<Vec<usize>>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.bench_iters == 0 {
        return Err("--bench-iters must be > 0".into());
    }
    if args.image_size % 14 != 0 {
        return Err(format!(
            "--image-size must be divisible by 14, got {}",
            args.image_size
        )
        .into());
    }

    let images = load_images(args.image.as_slice())?;
    let selected_view_counts = resolve_view_counts(images.len(), args.view_counts.as_deref())?;
    if selected_view_counts.is_empty() {
        return Err("no valid view counts selected".into());
    }

    println!(
        "[bench] inputs={} warmup_iters={} bench_iters={} synchronize={}",
        images.len(),
        args.warmup_iters,
        args.bench_iters,
        args.synchronize
    );

    let bootstrap_start = Instant::now();
    let weights = PipelineWeights::resolve_or_bootstrap_yono_with_precision(
        YonoWeightFormat::Burnpack,
        YonoWeightPrecision::F16,
    )?;
    let bootstrap_elapsed = bootstrap_start.elapsed();
    println!(
        "[bench] bootstrap_ms={:.2} backbone={} head={}",
        bootstrap_elapsed.as_secs_f64() * 1000.0,
        weights.yono.backbone.display(),
        weights.yono.head.display()
    );

    let load_start = Instant::now();
    let (pipeline, load_report) = ImageToGaussianPipeline::load(
        default_device(),
        PipelineConfig {
            model: PipelineModel::Yono,
            quality: PipelineQuality::Balanced,
            image_size: args.image_size,
        },
        weights,
    )?;
    let load_elapsed = load_start.elapsed();
    report_apply_summary("backbone", &load_report.backbone);
    report_apply_summary("head", &load_report.head);
    println!(
        "[bench] model_init_ms={:.2}",
        load_elapsed.as_secs_f64() * 1000.0
    );

    println!(
        "[bench] view_count,image_load_ms,backbone_ms,head_ms,forward_ms,wall_ms,wall_per_view_ms,wall_per_view_sq_ms,total_gaussians"
    );
    let mut last_wall_ms: Option<f64> = None;
    let mut last_views: Option<usize> = None;
    let max_views = *selected_view_counts.iter().max().unwrap_or(&2);
    for &views in &selected_view_counts {
        let inputs = to_input_images(&images[..views]);
        for _ in 0..args.warmup_iters {
            let _ =
                pipeline.run_image_bytes_timed_with_cameras(inputs.as_slice(), args.synchronize)?;
        }

        let mut timing_samples = Vec::with_capacity(args.bench_iters);
        let mut wall_samples = Vec::with_capacity(args.bench_iters);
        let mut total_gaussians = 0usize;
        for _ in 0..args.bench_iters {
            let wall_start = Instant::now();
            let run =
                pipeline.run_image_bytes_timed_with_cameras(inputs.as_slice(), args.synchronize)?;
            wall_samples.push(wall_start.elapsed());
            timing_samples.push(run.timings);
            let [batch, gaussians_per_batch, _] = run.gaussians.means.shape().dims::<3>();
            total_gaussians = batch * gaussians_per_batch;
        }

        let avg = average_timing_samples(timing_samples.as_slice(), wall_samples.as_slice());

        let wall_ms = avg.wall.as_secs_f64() * 1000.0;
        let wall_per_view_ms = wall_ms / views as f64;
        let wall_per_view_sq_ms = wall_ms / (views * views) as f64;
        println!(
            "[bench] {views},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{}",
            avg.image_load.as_secs_f64() * 1000.0,
            avg.backbone.as_secs_f64() * 1000.0,
            avg.head.as_secs_f64() * 1000.0,
            avg.forward_total.as_secs_f64() * 1000.0,
            wall_ms,
            wall_per_view_ms,
            wall_per_view_sq_ms,
            total_gaussians
        );
        if let (Some(previous_wall_ms), Some(previous_views)) = (last_wall_ms, last_views) {
            println!(
                "[bench]   growth_vs_prev={:.3}x ({} -> {} views)",
                wall_ms / previous_wall_ms,
                previous_views,
                views
            );
        }
        last_wall_ms = Some(wall_ms);
        last_views = Some(views);

        if args.check_determinism && views == max_views {
            let first = pipeline.run_image_bytes_timed_with_cameras(inputs.as_slice(), true)?;
            let second = pipeline.run_image_bytes_timed_with_cameras(inputs.as_slice(), true)?;
            let means_diff = diff_stats(
                tensor_to_vec(first.gaussians.means.clone())?.as_slice(),
                tensor_to_vec(second.gaussians.means.clone())?.as_slice(),
            )?;
            let opacities_diff = diff_stats(
                tensor_to_vec(first.gaussians.opacities.clone())?.as_slice(),
                tensor_to_vec(second.gaussians.opacities.clone())?.as_slice(),
            )?;
            let camera_diff = diff_stats(
                flatten_camera_poses(first.camera_poses.as_slice()).as_slice(),
                flatten_camera_poses(second.camera_poses.as_slice()).as_slice(),
            )?;

            println!(
                "[bench] determinism(max_views={}): means(mean_abs={:.6}, max_abs={:.6}) opacities(mean_abs={:.6}, max_abs={:.6}) cameras(mean_abs={:.6}, max_abs={:.6})",
                views,
                means_diff.mean_abs,
                means_diff.max_abs,
                opacities_diff.mean_abs,
                opacities_diff.max_abs,
                camera_diff.mean_abs,
                camera_diff.max_abs
            );

            for (name, stats) in [
                ("means", means_diff),
                ("opacities", opacities_diff),
                ("camera_poses", camera_diff),
            ] {
                if stats.mean_abs > args.max_determinism_mean_abs
                    || stats.max_abs > args.max_determinism_max_abs
                {
                    return Err(format!(
                        "determinism check failed for {name}: mean_abs={:.6} max_abs={:.6} (thresholds mean_abs<={} max_abs<={})",
                        stats.mean_abs,
                        stats.max_abs,
                        args.max_determinism_mean_abs,
                        args.max_determinism_max_abs
                    )
                    .into());
                }
            }
        }
    }

    Ok(())
}

fn load_images(paths: &[PathBuf]) -> Result<Vec<LoadedImage>, String> {
    let mut out = Vec::with_capacity(paths.len());
    for path in paths {
        let bytes =
            fs::read(path).map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        out.push(LoadedImage {
            name: image_name(path),
            bytes,
        });
    }
    Ok(out)
}

fn image_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| path.display().to_string())
}

fn resolve_view_counts(
    total_images: usize,
    requested: Option<&[usize]>,
) -> Result<Vec<usize>, String> {
    if total_images < 2 {
        return Err(format!("expected at least 2 images, got {total_images}"));
    }
    let mut counts = match requested {
        Some(values) => values.to_vec(),
        None => (2..=total_images).collect(),
    };
    counts.sort_unstable();
    counts.dedup();
    counts.retain(|count| *count >= 2 && *count <= total_images);
    Ok(counts)
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

fn report_apply_summary(name: &str, summary: &ComponentLoadReport) {
    println!(
        "[bench] load.{name}: applied={} missing={} unused={} skipped={}",
        summary.applied,
        summary.missing.len(),
        summary.unused.len(),
        summary.skipped.len()
    );
}

fn average_duration(samples: &[Duration]) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    let sum_secs = samples.iter().map(Duration::as_secs_f64).sum::<f64>();
    Duration::from_secs_f64(sum_secs / samples.len() as f64)
}

fn average_timing_samples(samples: &[ForwardTimings], wall_samples: &[Duration]) -> TimingAverages {
    if samples.is_empty() {
        return TimingAverages {
            image_load: Duration::ZERO,
            backbone: Duration::ZERO,
            head: Duration::ZERO,
            forward_total: Duration::ZERO,
            wall: average_duration(wall_samples),
        };
    }

    let count = samples.len() as f64;
    let image_load_secs = samples
        .iter()
        .map(|timings| timings.image_load.as_secs_f64())
        .sum::<f64>()
        / count;
    let backbone_secs = samples
        .iter()
        .map(|timings| timings.backbone.as_secs_f64())
        .sum::<f64>()
        / count;
    let head_secs = samples
        .iter()
        .map(|timings| timings.head.as_secs_f64())
        .sum::<f64>()
        / count;
    let forward_total_secs = samples
        .iter()
        .map(|timings| timings.total.as_secs_f64())
        .sum::<f64>()
        / count;

    TimingAverages {
        image_load: Duration::from_secs_f64(image_load_secs),
        backbone: Duration::from_secs_f64(backbone_secs),
        head: Duration::from_secs_f64(head_secs),
        forward_total: Duration::from_secs_f64(forward_total_secs),
        wall: average_duration(wall_samples),
    }
}

fn tensor_to_vec<const D: usize>(tensor: Tensor<BackendImpl, D>) -> Result<Vec<f32>, String> {
    tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| format!("failed tensor readback: {err:?}"))
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

fn diff_stats(lhs: &[f32], rhs: &[f32]) -> Result<DiffStats, String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "length mismatch in diff stats: lhs={} rhs={}",
            lhs.len(),
            rhs.len()
        ));
    }
    let mut sum_abs = 0.0f32;
    let mut max_abs = 0.0f32;
    for (&l, &r) in lhs.iter().zip(rhs.iter()) {
        let abs = (l - r).abs();
        sum_abs += abs;
        max_abs = max_abs.max(abs);
    }
    Ok(DiffStats {
        mean_abs: sum_abs / lhs.len().max(1) as f32,
        max_abs,
    })
}
