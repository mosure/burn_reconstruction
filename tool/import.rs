use std::path::PathBuf;

use burn::prelude::Backend;
use clap::{Parser, ValueEnum};

use burn_gaussian_splatting::{
    burn_yono::full_backbone_config,
    import::{
        load_yono_backbone_from_safetensors, load_yono_head_from_safetensors, report_apply_result,
        save_yono_backbone_record, save_yono_head_record, CheckpointFormat,
    },
    model::YonoHeadConfig,
};

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

#[derive(Clone, Debug, Parser)]
#[command(about = "burn_gaussian_splatting import", version, long_about = None)]
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
}

type BackendImpl = burn_gaussian_splatting::backend::BackendImpl;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = ImportConfig::parse();
    let device = <BackendImpl as Backend>::Device::default();

    if matches!(args.component, ComponentArg::Both | ComponentArg::Backbone) {
        let (model, result) = load_yono_backbone_from_safetensors::<BackendImpl>(
            &device,
            full_backbone_config(),
            args.backbone_weights.as_path(),
        )?;
        report_apply_result("yono_backbone", &result);

        let checkpoint_path =
            save_yono_backbone_record(&model, args.format.into(), args.backbone_output.as_path())?;
        println!("Saved backbone checkpoint to {}", checkpoint_path.display());
    }

    if matches!(args.component, ComponentArg::Both | ComponentArg::Head) {
        let config = YonoHeadConfig::new()
            .with_patch_size(args.patch_size)
            .with_gaussians_per_axis(args.gaussians_per_axis)
            .with_upscale_token_ratio(args.upscale_token_ratio)
            .with_num_surfaces(args.num_surfaces)
            .with_pose_free(args.pose_free);

        let (model, result) = load_yono_head_from_safetensors::<BackendImpl>(
            &device,
            config,
            args.head_weights.as_path(),
        )?;
        report_apply_result("yono_head", &result);

        let checkpoint_path =
            save_yono_head_record(&model, args.format.into(), args.head_output.as_path())?;
        println!("Saved head checkpoint to {}", checkpoint_path.display());
    }

    Ok(())
}
