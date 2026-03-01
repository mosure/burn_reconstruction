#![recursion_limit = "512"]

use std::path::PathBuf;

use burn::prelude::*;
use burn_reconstruction::{
    ImageToGaussianPipeline, PipelineConfig, PipelineModel, PipelineQuality, PipelineWeights,
    YonoWeightFormat, YonoWeightPrecision, YonoWeights,
};
use clap::{ArgAction, Parser};

#[derive(Clone, Copy, Debug)]
struct MetricStats {
    mean_abs: f32,
    max_abs: f32,
    max_rel: f32,
    mse: f32,
}

#[derive(Debug, Parser)]
#[command(about = "Evaluate f32 vs f16 burnpack numerical parity")]
struct Args {
    #[arg(long, required = true, num_args = 2..)]
    image: Vec<PathBuf>,

    #[arg(long, default_value_t = 224)]
    image_size: usize,

    #[arg(long, default_value = "assets/models/yono_backbone.bpk")]
    backbone: PathBuf,

    #[arg(long, default_value = "assets/models/yono_head.bpk")]
    head: PathBuf,

    #[arg(long, default_value_t = 5e-3)]
    max_mean_abs: f32,

    #[arg(long, default_value_t = 1e-1)]
    max_max_abs: f32,

    #[arg(long, default_value_t = 5e-4)]
    max_mse: f32,

    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    assert_thresholds: bool,
}

fn compute_stats(lhs: &[f32], rhs: &[f32]) -> Result<MetricStats, String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "length mismatch: lhs={} rhs={}",
            lhs.len(),
            rhs.len()
        ));
    }

    let mut sum_abs = 0.0f32;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut sum_sq = 0.0f32;

    for (&l, &r) in lhs.iter().zip(rhs.iter()) {
        let diff = l - r;
        let abs = diff.abs();
        sum_abs += abs;
        max_abs = max_abs.max(abs);
        if r.abs() > f32::EPSILON {
            max_rel = max_rel.max(abs / r.abs());
        }
        sum_sq += diff * diff;
    }

    let n = lhs.len() as f32;
    Ok(MetricStats {
        mean_abs: sum_abs / n,
        max_abs,
        max_rel,
        mse: sum_sq / n,
    })
}

fn tensor_to_vec<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Result<Vec<f32>, String> {
    tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| format!("failed tensor readback: {err:?}"))
}

fn flatten_camera_poses(poses: &[[[f32; 4]; 4]]) -> Vec<f32> {
    let mut flat = Vec::with_capacity(poses.len() * 16);
    for pose in poses {
        for row in pose {
            flat.extend_from_slice(row.as_slice());
        }
    }
    flat
}

fn report(name: &str, stats: MetricStats) {
    println!(
        "{name}: mean_abs={:.6} max_abs={:.6} max_rel={:.6} mse={:.6}",
        stats.mean_abs, stats.max_abs, stats.max_rel, stats.mse
    );
}

fn threshold_ok(stats: MetricStats, max_mean_abs: f32, max_max_abs: f32, max_mse: f32) -> bool {
    stats.mean_abs <= max_mean_abs && stats.max_abs <= max_max_abs && stats.mse <= max_mse
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.image.len() < 2 {
        return Err("expected at least two --image inputs".into());
    }
    for path in &args.image {
        if !path.exists() {
            return Err(format!("missing input image: {}", path.display()).into());
        }
    }
    if !args.backbone.exists() {
        return Err(format!("missing backbone burnpack: {}", args.backbone.display()).into());
    }
    if !args.head.exists() {
        return Err(format!("missing head burnpack: {}", args.head.display()).into());
    }

    let cfg = PipelineConfig {
        model: PipelineModel::Yono,
        quality: PipelineQuality::Balanced,
        image_size: args.image_size,
    };

    let f32_weights = PipelineWeights::from_yono(
        YonoWeights::burnpack(args.backbone.clone(), args.head.clone())
            .with_format(YonoWeightFormat::Burnpack)
            .with_precision(YonoWeightPrecision::F32),
    );
    let f16_weights = PipelineWeights::from_yono(
        YonoWeights::burnpack(args.backbone.clone(), args.head.clone())
            .with_format(YonoWeightFormat::Burnpack)
            .with_precision(YonoWeightPrecision::F16),
    );

    let (pipeline_f32, _) = ImageToGaussianPipeline::load_default(cfg.clone(), f32_weights)?;
    let (pipeline_f16, _) = ImageToGaussianPipeline::load_default(cfg, f16_weights)?;

    let out_f32 = pipeline_f32.run_images_timed_with_cameras(args.image.as_slice(), true)?;
    let out_f16 = pipeline_f16.run_images_timed_with_cameras(args.image.as_slice(), true)?;

    let means_stats = compute_stats(
        tensor_to_vec(out_f32.gaussians.means.clone())?.as_slice(),
        tensor_to_vec(out_f16.gaussians.means.clone())?.as_slice(),
    )?;
    let covariances_stats = compute_stats(
        tensor_to_vec(out_f32.gaussians.covariances.clone())?.as_slice(),
        tensor_to_vec(out_f16.gaussians.covariances.clone())?.as_slice(),
    )?;
    let harmonics_stats = compute_stats(
        tensor_to_vec(out_f32.gaussians.harmonics.clone())?.as_slice(),
        tensor_to_vec(out_f16.gaussians.harmonics.clone())?.as_slice(),
    )?;
    let opacities_stats = compute_stats(
        tensor_to_vec(out_f32.gaussians.opacities.clone())?.as_slice(),
        tensor_to_vec(out_f16.gaussians.opacities.clone())?.as_slice(),
    )?;
    let rotations_stats = compute_stats(
        tensor_to_vec(out_f32.gaussians.rotations.clone())?.as_slice(),
        tensor_to_vec(out_f16.gaussians.rotations.clone())?.as_slice(),
    )?;
    let scales_stats = compute_stats(
        tensor_to_vec(out_f32.gaussians.scales.clone())?.as_slice(),
        tensor_to_vec(out_f16.gaussians.scales.clone())?.as_slice(),
    )?;
    let camera_stats = compute_stats(
        flatten_camera_poses(out_f32.camera_poses.as_slice()).as_slice(),
        flatten_camera_poses(out_f16.camera_poses.as_slice()).as_slice(),
    )?;

    report("means", means_stats);
    report("covariances", covariances_stats);
    report("harmonics", harmonics_stats);
    report("opacities", opacities_stats);
    report("rotations", rotations_stats);
    report("scales", scales_stats);
    report("camera_poses", camera_stats);

    println!(
        "f32 timings: image_load={:.3}ms backbone={:.3}ms head={:.3}ms total={:.3}ms",
        out_f32.timings.image_load.as_secs_f64() * 1000.0,
        out_f32.timings.backbone.as_secs_f64() * 1000.0,
        out_f32.timings.head.as_secs_f64() * 1000.0,
        out_f32.timings.total.as_secs_f64() * 1000.0
    );
    println!(
        "f16 timings: image_load={:.3}ms backbone={:.3}ms head={:.3}ms total={:.3}ms",
        out_f16.timings.image_load.as_secs_f64() * 1000.0,
        out_f16.timings.backbone.as_secs_f64() * 1000.0,
        out_f16.timings.head.as_secs_f64() * 1000.0,
        out_f16.timings.total.as_secs_f64() * 1000.0
    );

    if args.assert_thresholds {
        let metrics = [
            ("means", means_stats),
            ("covariances", covariances_stats),
            ("harmonics", harmonics_stats),
            ("opacities", opacities_stats),
            ("rotations", rotations_stats),
            ("scales", scales_stats),
            ("camera_poses", camera_stats),
        ];
        let mut failures = Vec::new();
        for (name, stats) in metrics {
            if !threshold_ok(stats, args.max_mean_abs, args.max_max_abs, args.max_mse) {
                failures.push(format!(
                    "{name}: mean_abs={:.6} max_abs={:.6} mse={:.6}",
                    stats.mean_abs, stats.max_abs, stats.mse
                ));
            }
        }
        if !failures.is_empty() {
            return Err(format!(
                "precision parity failed thresholds (mean_abs <= {}, max_abs <= {}, mse <= {}):\n{}",
                args.max_mean_abs,
                args.max_max_abs,
                args.max_mse,
                failures.join("\n")
            )
            .into());
        }
    }

    Ok(())
}
