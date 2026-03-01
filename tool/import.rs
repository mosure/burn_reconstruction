#![recursion_limit = "512"]

use std::path::{Path, PathBuf};

use burn::prelude::Backend;
use clap::{ArgAction, Parser, ValueEnum};

use burn_yono::{
    full_backbone_config,
    import::{
        convert_burnpack_to_f16, ensure_burnpack_parts, load_yono_backbone_from_safetensors,
        load_yono_head_from_safetensors, report_apply_result, save_yono_backbone_record,
        save_yono_backbone_record_bpk, save_yono_head_record, save_yono_head_record_bpk,
        CheckpointFormat, DEFAULT_PART_SIZE_MIB,
    },
    model::YonoHeadConfig,
};

type ImportBackendF32 = burn::backend::NdArray<f32>;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OutputFormat {
    Bpk,
    Mpk,
}

impl From<OutputFormat> for CheckpointFormat {
    fn from(value: OutputFormat) -> Self {
        match value {
            OutputFormat::Bpk => Self::Bpk,
            OutputFormat::Mpk => Self::Mpk,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ComponentArg {
    Both,
    Backbone,
    Head,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PrecisionArg {
    F16,
    F32,
    Both,
}

impl PrecisionArg {
    fn include_f16(self) -> bool {
        matches!(self, Self::F16 | Self::Both)
    }

    fn include_f32(self) -> bool {
        matches!(self, Self::F32 | Self::Both)
    }
}

#[derive(Clone, Debug, Parser)]
#[command(about = "burn_reconstruction import", version, long_about = None)]
struct ImportConfig {
    #[arg(long, value_enum, default_value_t = ComponentArg::Both)]
    component: ComponentArg,

    #[arg(
        long,
        default_value = "assets/models/yono_backbone_weights.safetensors"
    )]
    backbone_weights: PathBuf,

    #[arg(long, default_value = "assets/models/yono_head_weights.safetensors")]
    head_weights: PathBuf,

    #[arg(long, default_value = "assets/models/yono_backbone")]
    backbone_output: PathBuf,

    #[arg(long, default_value = "assets/models/yono_head")]
    head_output: PathBuf,

    #[arg(long, value_enum, default_value_t = OutputFormat::Bpk)]
    format: OutputFormat,

    #[arg(long, value_enum, default_value_t = PrecisionArg::Both)]
    precision: PrecisionArg,

    #[arg(long, default_value_t = 14)]
    patch_size: usize,

    #[arg(long, default_value_t = 14)]
    gaussians_per_axis: usize,

    #[arg(long, default_value_t = 2)]
    upscale_token_ratio: usize,

    #[arg(long, default_value_t = 1)]
    num_surfaces: usize,

    #[arg(long, default_value_t = true)]
    pose_free: bool,

    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    parts: bool,

    #[arg(long, default_value_t = DEFAULT_PART_SIZE_MIB)]
    parts_max_mib: u64,

    #[arg(long, default_value_t = false)]
    parts_overwrite: bool,
}

fn maybe_write_parts(
    path: &Path,
    args: &ImportConfig,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if !args.parts || !matches!(args.format, OutputFormat::Bpk) {
        return Ok(());
    }
    if let Some(parts_report) =
        ensure_burnpack_parts(path, args.parts_max_mib, args.parts_overwrite)?
    {
        println!(
            "Wrote {label} parts manifest {} ({} parts, {:.1} MiB source)",
            parts_report.manifest_path.display(),
            parts_report.part_paths.len(),
            parts_report.total_bytes as f64 / (1024.0 * 1024.0)
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = ImportConfig::parse();

    if matches!(args.format, OutputFormat::Mpk) && args.precision.include_f16() {
        println!("[IMPORT] --precision includes f16, but .mpk export stores full precision only");
    }

    if matches!(args.component, ComponentArg::Both | ComponentArg::Backbone) {
        let device = <ImportBackendF32 as Backend>::Device::default();
        let (model, result) = load_yono_backbone_from_safetensors::<ImportBackendF32>(
            &device,
            full_backbone_config(),
            args.backbone_weights.as_path(),
        )?;
        report_apply_result("yono_backbone/f32", &result);

        if matches!(args.format, OutputFormat::Mpk) {
            let checkpoint_path = save_yono_backbone_record(
                &model,
                args.format.into(),
                args.backbone_output.as_path(),
            )?;
            println!("Saved backbone checkpoint to {}", checkpoint_path.display());
        } else {
            let checkpoint_path =
                save_yono_backbone_record_bpk(&model, args.backbone_output.as_path())?;
            println!(
                "Saved backbone f32 checkpoint to {}",
                checkpoint_path.display()
            );
            if args.precision.include_f32() {
                maybe_write_parts(checkpoint_path.as_path(), &args, "backbone f32")?;
            }
            if args.precision.include_f16() {
                let f16_path = convert_burnpack_to_f16(
                    checkpoint_path.as_path(),
                    args.backbone_output.as_path(),
                )?;
                println!("Saved backbone f16 checkpoint to {}", f16_path.display());
                maybe_write_parts(f16_path.as_path(), &args, "backbone f16")?;
            }
        }
    }

    if matches!(args.component, ComponentArg::Both | ComponentArg::Head) {
        let config = YonoHeadConfig::new()
            .with_patch_size(args.patch_size)
            .with_gaussians_per_axis(args.gaussians_per_axis)
            .with_upscale_token_ratio(args.upscale_token_ratio)
            .with_num_surfaces(args.num_surfaces)
            .with_pose_free(args.pose_free);

        let device = <ImportBackendF32 as Backend>::Device::default();
        let (model, result) = load_yono_head_from_safetensors::<ImportBackendF32>(
            &device,
            config,
            args.head_weights.as_path(),
        )?;
        report_apply_result("yono_head/f32", &result);

        if matches!(args.format, OutputFormat::Mpk) {
            let checkpoint_path =
                save_yono_head_record(&model, args.format.into(), args.head_output.as_path())?;
            println!("Saved head checkpoint to {}", checkpoint_path.display());
        } else {
            let checkpoint_path = save_yono_head_record_bpk(&model, args.head_output.as_path())?;
            println!("Saved head f32 checkpoint to {}", checkpoint_path.display());
            if args.precision.include_f32() {
                maybe_write_parts(checkpoint_path.as_path(), &args, "head f32")?;
            }
            if args.precision.include_f16() {
                let f16_path =
                    convert_burnpack_to_f16(checkpoint_path.as_path(), args.head_output.as_path())?;
                println!("Saved head f16 checkpoint to {}", f16_path.display());
                maybe_write_parts(f16_path.as_path(), &args, "head f16")?;
            }
        }
    }

    Ok(())
}
