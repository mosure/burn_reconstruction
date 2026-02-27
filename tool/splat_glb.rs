#![recursion_limit = "512"]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use burn_gaussian_splatting::{
    backend::default_device, ComponentLoadReport, ForwardTimings, GlbExportOptions,
    GlbExportReport, GlbSortMode, ImageToGaussianPipeline, PipelineConfig, PipelineGaussians,
    PipelineModel, PipelineQuality, PipelineWeights, YonoWeightFormat, YonoWeights,
};
use clap::{Parser, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum QualityArg {
    Fast,
    Balanced,
    High,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SortModeArg {
    Opacity,
    Index,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum WeightFormatArg {
    Safetensors,
    Bpk,
}

#[derive(Clone, Debug, Parser)]
#[command(
    about = "Run multi-image inference and export KHR_gaussian_splatting GLB",
    version,
    long_about = None
)]
struct CliConfig {
    #[arg(long, required = true, num_args = 2..)]
    images: Vec<PathBuf>,

    #[arg(long, default_value = "outputs/gaussians.glb")]
    output: PathBuf,

    #[arg(long, default_value_t = 224)]
    image_size: usize,

    #[arg(long, value_enum, default_value_t = QualityArg::Balanced)]
    quality: QualityArg,

    #[arg(long)]
    max_gaussians: Option<usize>,

    #[arg(long)]
    opacity_threshold: Option<f32>,

    #[arg(long, value_enum)]
    sort_mode: Option<SortModeArg>,

    #[arg(long, default_value_t = false)]
    profile: bool,

    #[arg(long, default_value_t = 1)]
    warmup_iters: usize,

    #[arg(long, default_value_t = 1)]
    bench_iters: usize,

    #[arg(long, default_value_t = false)]
    single_sync_profile: bool,

    #[arg(long, value_enum, default_value_t = WeightFormatArg::Safetensors)]
    weights_format: WeightFormatArg,

    #[arg(long)]
    backbone_weights: Option<PathBuf>,

    #[arg(long)]
    head_weights: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = CliConfig::parse();
    let device = default_device();

    if args.image_size % 14 != 0 {
        return Err(format!(
            "--image-size must be divisible by 14, got {}",
            args.image_size
        )
        .into());
    }
    if args.image_size != 224 {
        return Err(
            "the provided pretrained backbone weights are calibrated for --image-size 224".into(),
        );
    }
    if args.bench_iters == 0 {
        return Err("--bench-iters must be > 0".into());
    }

    let quality = match args.quality {
        QualityArg::Fast => PipelineQuality::Fast,
        QualityArg::Balanced => PipelineQuality::Balanced,
        QualityArg::High => PipelineQuality::High,
    };

    let weight_format = match args.weights_format {
        WeightFormatArg::Safetensors => YonoWeightFormat::Safetensors,
        WeightFormatArg::Bpk => YonoWeightFormat::Burnpack,
    };

    let weights = match (&args.backbone_weights, &args.head_weights) {
        (Some(backbone), Some(head)) => PipelineWeights::from_yono(
            YonoWeights::new(backbone.clone(), head.clone()).with_format(weight_format),
        ),
        (None, None) => {
            let weights = PipelineWeights::resolve_or_bootstrap_yono(weight_format)?;
            println!(
                "[BOOTSTRAP] using cached YoNo weights:\n  backbone={}\n  head={}",
                weights.yono.backbone.display(),
                weights.yono.head.display()
            );
            weights
        }
        _ => {
            return Err(
                "`--backbone-weights` and `--head-weights` must be provided together".into(),
            );
        }
    };

    let (pipeline, load_report) = ImageToGaussianPipeline::load(
        device,
        PipelineConfig {
            model: PipelineModel::Yono,
            quality,
            image_size: args.image_size,
        },
        weights,
    )?;

    report_apply_summary("backbone", &load_report.backbone);
    report_apply_summary("head", &load_report.head);

    let (gaussians, timed) = if args.profile {
        for _ in 0..args.warmup_iters {
            let run =
                pipeline.run_images_timed(args.images.as_slice(), !args.single_sync_profile)?;
            if args.single_sync_profile {
                sync_flat_gaussians(&run.gaussians);
            }
        }

        let mut timings_samples = Vec::with_capacity(args.bench_iters);
        let mut wall_samples = Vec::with_capacity(args.bench_iters);
        let mut output = None;
        for _ in 0..args.bench_iters {
            let wall_start = Instant::now();
            let run =
                pipeline.run_images_timed(args.images.as_slice(), !args.single_sync_profile)?;
            if args.single_sync_profile {
                sync_flat_gaussians(&run.gaussians);
            }
            wall_samples.push(wall_start.elapsed());
            timings_samples.push(run.timings);
            output = Some(run.gaussians);
        }

        let gaussians = output.expect("bench_iters > 0 guarantees at least one output");
        (
            gaussians,
            Some(ProfileSummary {
                timings: average_timings(timings_samples.as_slice()),
                wall: average_duration(wall_samples.as_slice()),
                warmup_iters: args.warmup_iters,
                bench_iters: args.bench_iters,
            }),
        )
    } else {
        (pipeline.run_images(args.images.as_slice())?, None)
    };

    let mut export = quality.export_options();
    if let Some(max_gaussians) = args.max_gaussians {
        export.max_gaussians = max_gaussians;
    }
    if let Some(opacity_threshold) = args.opacity_threshold {
        export.opacity_threshold = opacity_threshold;
    }
    if let Some(sort_mode) = args.sort_mode {
        export.sort_mode = match sort_mode {
            SortModeArg::Opacity => GlbSortMode::Opacity,
            SortModeArg::Index => GlbSortMode::Index,
        };
    }

    let export_report = pipeline.save_glb(&args.output, &gaussians, &export)?;

    println!(
        "Wrote GLB with {} gaussians to {}",
        export_report.selected_gaussians,
        args.output.display()
    );

    if let Some(profile) = timed.as_ref() {
        print_profile(profile, &export_report, &export);
    }

    Ok(())
}

fn report_apply_summary(name: &str, summary: &ComponentLoadReport) {
    println!(
        "[LOAD] {name}: applied={} missing={} unused={} skipped={}",
        summary.applied,
        summary.missing.len(),
        summary.unused.len(),
        summary.skipped.len()
    );

    if !summary.missing.is_empty() {
        println!("[LOAD] {name} missing:");
        for key in &summary.missing {
            println!("  - {key}");
        }
    }

    if !summary.unused.is_empty() {
        println!("[LOAD] {name} unused:");
        for key in &summary.unused {
            println!("  - {key}");
        }
    }
}

#[derive(Debug, Clone)]
struct ProfileSummary {
    timings: ForwardTimings,
    wall: Duration,
    warmup_iters: usize,
    bench_iters: usize,
}

fn average_timings(samples: &[ForwardTimings]) -> ForwardTimings {
    if samples.is_empty() {
        return ForwardTimings::default();
    }

    let mut image_load_sum = 0.0f64;
    let mut backbone_sum = 0.0f64;
    let mut head_sum = 0.0f64;
    let mut total_sum = 0.0f64;

    for timings in samples {
        image_load_sum += timings.image_load.as_secs_f64();
        backbone_sum += timings.backbone.as_secs_f64();
        head_sum += timings.head.as_secs_f64();
        total_sum += timings.total.as_secs_f64();
    }

    let n = samples.len() as f64;
    let image_load = Duration::from_secs_f64(image_load_sum / n);
    let backbone = Duration::from_secs_f64(backbone_sum / n);
    let head = Duration::from_secs_f64(head_sum / n);
    let total = Duration::from_secs_f64(total_sum / n);

    ForwardTimings {
        image_load,
        backbone,
        head,
        total,
    }
}

fn average_duration(samples: &[Duration]) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }

    let mut total = 0.0f64;
    for sample in samples {
        total += sample.as_secs_f64();
    }
    Duration::from_secs_f64(total / samples.len() as f64)
}

fn print_profile(profile: &ProfileSummary, export: &GlbExportReport, options: &GlbExportOptions) {
    let forward = &profile.timings;
    println!(
        "[PROFILE] warmup_iters={} bench_iters={}",
        profile.warmup_iters, profile.bench_iters
    );
    println!(
        "[PROFILE] image_load_ms={:.3}",
        forward.image_load.as_secs_f64() * 1000.0
    );
    println!(
        "[PROFILE] backbone_ms={:.3}",
        forward.backbone.as_secs_f64() * 1000.0
    );
    println!(
        "[PROFILE] head_ms={:.3}",
        forward.head.as_secs_f64() * 1000.0
    );
    println!(
        "[PROFILE] forward_total_ms={:.3}",
        forward.total.as_secs_f64() * 1000.0
    );
    println!(
        "[PROFILE] forward_wall_ms={:.3}",
        profile.wall.as_secs_f64() * 1000.0
    );
    println!("[PROFILE] export_select_ms={:.3}", export.select_millis);
    println!("[PROFILE] export_write_ms={:.3}", export.write_millis);
    println!(
        "[PROFILE] export_config=max_gaussians:{} opacity_threshold:{:.6} sort_mode:{:?}",
        options.max_gaussians, options.opacity_threshold, options.sort_mode
    );
}

fn sync_flat_gaussians(gaussians: &PipelineGaussians) {
    let [batch, count] = gaussians.opacities.shape().dims::<2>();
    if batch == 0 || count == 0 {
        return;
    }
    let _ = gaussians
        .opacities
        .clone()
        .slice([0..1, 0..1])
        .into_data()
        .to_vec::<f32>();
}
