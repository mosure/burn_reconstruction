#![recursion_limit = "512"]

use std::path::PathBuf;

use burn::prelude::*;
use burn_reconstruction::correctness::{
    compute_stats, load_safetensors, read_tensor, tensor_to_vec,
};
use burn_yono::{
    full_backbone_config, full_head_config,
    import::{load_yono_backbone_from_safetensors, load_yono_head_from_safetensors},
    model::YonoHeadInput,
};
use clap::Parser;

type BackendImpl = burn_reconstruction::backend::BackendImpl;

const REFERENCE_PATH: &str = "tmp/cli_test/python_multiview_reference.safetensors";
const BACKBONE_WEIGHTS: &str = "assets/models/yono_backbone_weights.safetensors";
const HEAD_WEIGHTS: &str = "assets/models/yono_head_weights.safetensors";

#[derive(Debug, Parser)]
#[command(about = "Evaluate Burn vs Python multiview parity using safetensors references")]
struct CliArgs {
    #[arg(long, default_value = REFERENCE_PATH)]
    reference: PathBuf,
    #[arg(long, default_value = BACKBONE_WEIGHTS)]
    backbone_weights: PathBuf,
    #[arg(long, default_value = HEAD_WEIGHTS)]
    head_weights: PathBuf,
}

fn tensor_from_f32<const D: usize>(
    values: Vec<f32>,
    shape: [usize; D],
    device: &<BackendImpl as Backend>::Device,
) -> Tensor<BackendImpl, D> {
    let mut shape_i32 = [0_i32; D];
    for (dst, src) in shape_i32.iter_mut().zip(shape.iter()) {
        *dst = *src as i32;
    }
    Tensor::<BackendImpl, 1>::from_floats(values.as_slice(), device).reshape(shape_i32)
}

fn report(name: &str, actual: Vec<f32>, expected: Vec<f32>) {
    let stats = compute_stats(&actual, &expected).expect("stats should compute");
    println!(
        "{name}: mean_abs={:.6} max_abs={:.6} max_rel={:.6} mse={:.6}",
        stats.mean_abs, stats.max_abs, stats.max_rel, stats.mse
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = CliArgs::parse();

    if !args.reference.exists() {
        return Err(format!("reference missing at {}", args.reference.display()).into());
    }
    if !args.backbone_weights.exists() {
        return Err(format!(
            "backbone weights missing at {}",
            args.backbone_weights.display()
        )
        .into());
    }
    if !args.head_weights.exists() {
        return Err(format!("head weights missing at {}", args.head_weights.display()).into());
    }

    let device = <BackendImpl as Backend>::Device::default();
    let tensors = load_safetensors(args.reference.as_path())?;

    let (image_v, image_s) = read_tensor(&tensors, "image")?;
    let (intr_v, intr_s) = read_tensor(&tensors, "intrinsics")?;
    let (hidden_ref_v, hidden_ref_s) = read_tensor(&tensors, "backbone_hidden")?;
    let (pos_ref_v, pos_ref_s) = read_tensor(&tensors, "backbone_pos")?;
    let (means_ref_v, _) = read_tensor(&tensors, "gaussians_means")?;
    let (harm_ref_v, _) = read_tensor(&tensors, "gaussians_harmonics")?;
    let (opa_ref_v, _) = read_tensor(&tensors, "gaussians_opacities")?;
    let (scl_ref_v, _) = read_tensor(&tensors, "gaussians_scales")?;
    let (rot_ref_v, _) = read_tensor(&tensors, "gaussians_rotations")?;

    let image = tensor_from_f32::<5>(
        image_v,
        [image_s[0], image_s[1], image_s[2], image_s[3], image_s[4]],
        &device,
    );
    let intrinsics = tensor_from_f32::<4>(
        intr_v,
        [intr_s[0], intr_s[1], intr_s[2], intr_s[3]],
        &device,
    );
    let hidden_ref = tensor_from_f32::<3>(
        hidden_ref_v.clone(),
        [hidden_ref_s[0], hidden_ref_s[1], hidden_ref_s[2]],
        &device,
    );
    let pos_ref = tensor_from_f32::<3>(
        pos_ref_v.clone(),
        [pos_ref_s[0], pos_ref_s[1], pos_ref_s[2]],
        &device,
    );

    let (backbone, backbone_apply) = load_yono_backbone_from_safetensors::<BackendImpl>(
        &device,
        full_backbone_config(),
        args.backbone_weights.as_path(),
    )?;
    println!(
        "backbone apply: applied={} missing={} unused={}",
        backbone_apply.applied.len(),
        backbone_apply.missing.len(),
        backbone_apply.unused.len()
    );
    if !backbone_apply.missing.is_empty() {
        println!("backbone missing: {:?}", backbone_apply.missing);
    }

    let (head, head_apply) = load_yono_head_from_safetensors::<BackendImpl>(
        &device,
        full_head_config(),
        args.head_weights.as_path(),
    )?;
    println!(
        "head apply: applied={} missing={} unused={}",
        head_apply.applied.len(),
        head_apply.missing.len(),
        head_apply.unused.len()
    );

    let back = backbone.forward(image.clone(), Some(intrinsics.clone()));
    let back_auto = backbone.forward_with_normalized_intrinsics(image.clone());
    report(
        "backbone_hidden",
        tensor_to_vec(back.hidden.clone())?,
        hidden_ref_v.clone(),
    );
    report(
        "backbone_hidden_auto_intrinsics",
        tensor_to_vec(back_auto.hidden.clone())?,
        hidden_ref_v.clone(),
    );
    report(
        "backbone_pos",
        tensor_to_vec(back.pos.clone())?,
        pos_ref_v.clone(),
    );

    let head_full = head.forward(YonoHeadInput {
        image: image.clone(),
        hidden: back.hidden,
        pos: back.pos,
        hidden_upsampled: None,
        pos_upsampled: None,
        patch_start_idx: back.patch_start_idx,
        global_step: 0,
        training: false,
        extrinsics: None,
        use_predicted_pose: true,
        scheduled_sampling_draw: None,
    });
    report(
        "full_gaussians_means",
        tensor_to_vec(head_full.gaussians_flat.means.clone())?,
        means_ref_v.clone(),
    );
    report(
        "full_gaussians_harmonics",
        tensor_to_vec(head_full.gaussians_flat.harmonics.clone())?,
        harm_ref_v.clone(),
    );
    report(
        "full_gaussians_opacities",
        tensor_to_vec(head_full.gaussians_flat.opacities.clone())?,
        opa_ref_v.clone(),
    );
    report(
        "full_gaussians_scales",
        tensor_to_vec(head_full.gaussians_flat.scales.clone())?,
        scl_ref_v.clone(),
    );
    report(
        "full_gaussians_rotations",
        tensor_to_vec(head_full.gaussians_flat.rotations.clone())?,
        rot_ref_v.clone(),
    );

    let head_ref_hidden = head.forward(YonoHeadInput {
        image,
        hidden: hidden_ref,
        pos: pos_ref,
        hidden_upsampled: None,
        pos_upsampled: None,
        patch_start_idx: 5,
        global_step: 0,
        training: false,
        extrinsics: None,
        use_predicted_pose: true,
        scheduled_sampling_draw: None,
    });
    report(
        "head_only_gaussians_means",
        tensor_to_vec(head_ref_hidden.gaussians_flat.means.clone())?,
        means_ref_v,
    );
    report(
        "head_only_gaussians_harmonics",
        tensor_to_vec(head_ref_hidden.gaussians_flat.harmonics.clone())?,
        harm_ref_v,
    );
    report(
        "head_only_gaussians_opacities",
        tensor_to_vec(head_ref_hidden.gaussians_flat.opacities.clone())?,
        opa_ref_v,
    );
    report(
        "head_only_gaussians_scales",
        tensor_to_vec(head_ref_hidden.gaussians_flat.scales.clone())?,
        scl_ref_v,
    );
    report(
        "head_only_gaussians_rotations",
        tensor_to_vec(head_ref_hidden.gaussians_flat.rotations.clone())?,
        rot_ref_v,
    );

    Ok(())
}
