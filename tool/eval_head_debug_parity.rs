#![recursion_limit = "512"]

use std::path::Path;

use burn::prelude::*;
use burn::tensor::activation::sigmoid;
use burn_gaussian_splatting::correctness::{
    compute_stats, load_safetensors, read_tensor, tensor_to_vec,
};
use burn_yono::{
    full_head_config,
    import::load_yono_head_from_safetensors,
    model::{
        ops::interpolate_bilinear_align_corners_false, ops::position_getter, ops::se3_inverse_flat,
    },
};

type BackendImpl = burn_gaussian_splatting::backend::BackendImpl;

const DEBUG_REFERENCE_PATH: &str = "tmp/cli_test/python_head_only_debug_fp32.safetensors";
const HEAD_WEIGHTS: &str = "assets/models/yono_head_weights.safetensors";

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
    if !Path::new(DEBUG_REFERENCE_PATH).exists() {
        return Err(format!("reference missing at {DEBUG_REFERENCE_PATH}").into());
    }
    if !Path::new(HEAD_WEIGHTS).exists() {
        return Err(format!("head weights missing at {HEAD_WEIGHTS}").into());
    }

    let device = <BackendImpl as Backend>::Device::default();
    let tensors = load_safetensors(DEBUG_REFERENCE_PATH)?;

    let (image_v, image_s) = read_tensor(&tensors, "image")?;
    let (hidden_v, hidden_s) = read_tensor(&tensors, "hidden")?;
    let (pos_v, pos_s) = read_tensor(&tensors, "pos")?;

    let (hidden_up_ref_v, _) = read_tensor(&tensors, "hidden_upsampled")?;
    let (pos_up_ref_v, _) = read_tensor(&tensors, "pos_upsampled")?;
    let (rgb_feat_ref_v, _) = read_tensor(&tensors, "rgb_feat")?;
    let (hidden_gaussian_ref_v, _) = read_tensor(&tensors, "hidden_gaussian")?;
    let (point_hidden_ref_v, _) = read_tensor(&tensors, "point_hidden")?;
    let (gaussian_hidden_ref_v, _) = read_tensor(&tensors, "gaussian_hidden")?;
    let (camera_hidden_ref_v, _) = read_tensor(&tensors, "camera_hidden")?;
    let (point_raw_ref_v, _) = read_tensor(&tensors, "point_raw")?;
    let (local_points_ref_v, _) = read_tensor(&tensors, "local_points")?;
    let (gaussian_params_ref_v, _) = read_tensor(&tensors, "gaussian_params")?;
    let (camera_poses_ref_v, _) = read_tensor(&tensors, "camera_poses")?;
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
    let hidden = tensor_from_f32::<3>(hidden_v, [hidden_s[0], hidden_s[1], hidden_s[2]], &device);
    let pos = tensor_from_f32::<3>(pos_v, [pos_s[0], pos_s[1], pos_s[2]], &device);

    let (head, head_apply) = load_yono_head_from_safetensors::<BackendImpl>(
        &device,
        full_head_config(),
        Path::new(HEAD_WEIGHTS),
    )?;
    println!(
        "head apply: applied={} missing={} unused={}",
        head_apply.applied.len(),
        head_apply.missing.len(),
        head_apply.unused.len()
    );

    let cfg = head.config();
    let [b, v, _, h, w] = image.shape().dims::<5>();
    let patch_h = h / cfg.patch_size;
    let patch_w = w / cfg.patch_size;
    let bv = b * v;
    let patch_start_idx = 5usize;

    let hidden_dims = hidden.shape().dims::<3>();
    let img_tokens = hidden_dims[1] - patch_start_idx;
    let hidden_aux = hidden.clone().slice([
        0..bv as i32,
        0..patch_start_idx as i32,
        0..hidden_dims[2] as i32,
    ]);
    let hidden_img = hidden
        .clone()
        .slice([
            0..bv as i32,
            patch_start_idx as i32..(patch_start_idx + img_tokens) as i32,
            0..hidden_dims[2] as i32,
        ])
        .reshape([
            bv as i32,
            patch_h as i32,
            patch_w as i32,
            hidden_dims[2] as i32,
        ])
        .permute([0, 3, 1, 2]);
    let hidden_img = interpolate_bilinear_align_corners_false(
        hidden_img,
        [
            patch_h * cfg.upscale_token_ratio,
            patch_w * cfg.upscale_token_ratio,
        ],
    )
    .permute([0, 2, 3, 1])
    .reshape([
        bv as i32,
        (patch_h * patch_w * cfg.upscale_token_ratio * cfg.upscale_token_ratio) as i32,
        hidden_dims[2] as i32,
    ]);
    let hidden_upsampled = Tensor::cat(vec![hidden_aux, hidden_img], 1);

    let pos_aux = pos
        .clone()
        .slice([0..bv as i32, 0..patch_start_idx as i32, 0..2]);
    let mut pos_img = position_getter(
        bv,
        patch_h * cfg.upscale_token_ratio,
        patch_w * cfg.upscale_token_ratio,
        &device,
    );
    pos_img = pos_img + 1.0;
    let pos_upsampled = Tensor::cat(vec![pos_aux, pos_img], 1);

    report(
        "hidden_upsampled",
        tensor_to_vec(hidden_upsampled.clone())?,
        hidden_up_ref_v,
    );
    report(
        "pos_upsampled",
        tensor_to_vec(pos_upsampled.clone())?,
        pos_up_ref_v,
    );

    let rgb = image.reshape([bv as i32, 3, h as i32, w as i32]);
    let rgb_feat = head.rgb_embed.forward(rgb);
    report("rgb_feat", tensor_to_vec(rgb_feat.clone())?, rgb_feat_ref_v);

    let up_dims = hidden_upsampled.shape().dims::<3>();
    let hidden_aux = hidden_upsampled.clone().slice([
        0..bv as i32,
        0..patch_start_idx as i32,
        0..up_dims[2] as i32,
    ]);
    let hidden_img = hidden_upsampled.clone().slice([
        0..bv as i32,
        patch_start_idx as i32..up_dims[1] as i32,
        0..up_dims[2] as i32,
    ]) + rgb_feat;
    let hidden_gaussian = Tensor::cat(vec![hidden_aux, hidden_img], 1);
    report(
        "hidden_gaussian",
        tensor_to_vec(hidden_gaussian.clone())?,
        hidden_gaussian_ref_v,
    );

    let point_hidden = head
        .point_decoder
        .forward(hidden_upsampled.clone(), Some(&pos_upsampled));
    let gaussian_hidden = head
        .gaussian_decoder
        .forward(hidden_gaussian.clone(), Some(&pos_upsampled));
    let camera_hidden = head.camera_decoder.forward(hidden.clone(), Some(&pos));

    report(
        "point_hidden",
        tensor_to_vec(point_hidden.clone())?,
        point_hidden_ref_v,
    );
    report(
        "gaussian_hidden",
        tensor_to_vec(gaussian_hidden.clone())?,
        gaussian_hidden_ref_v,
    );
    report(
        "camera_hidden",
        tensor_to_vec(camera_hidden.clone())?,
        camera_hidden_ref_v,
    );

    let gaussians_per_axis = cfg.effective_gaussians_per_axis();
    let out_h = patch_h * gaussians_per_axis;
    let out_w = patch_w * gaussians_per_axis;

    let point_tokens = point_hidden.clone().slice([
        0..bv as i32,
        patch_start_idx as i32..point_hidden.shape().dims::<3>()[1] as i32,
        0..point_hidden.shape().dims::<3>()[2] as i32,
    ]);
    let point_raw = head.point_head.forward(point_tokens, (h, w)).reshape([
        b as i32,
        v as i32,
        out_h as i32,
        out_w as i32,
        3,
    ]);
    report(
        "point_raw",
        tensor_to_vec(point_raw.clone())?,
        point_raw_ref_v,
    );

    let xy = point_raw.clone().slice([
        0..b as i32,
        0..v as i32,
        0..out_h as i32,
        0..out_w as i32,
        0..2,
    ]);
    let z = point_raw
        .slice([
            0..b as i32,
            0..v as i32,
            0..out_h as i32,
            0..out_w as i32,
            2..3,
        ])
        .exp();
    let local_points = Tensor::cat(vec![xy * z.clone().repeat_dim(4, 2), z], 4);
    report(
        "local_points",
        tensor_to_vec(local_points.clone())?,
        local_points_ref_v,
    );

    let gaussian_tokens = gaussian_hidden.clone().slice([
        0..bv as i32,
        patch_start_idx as i32..gaussian_hidden.shape().dims::<3>()[1] as i32,
        0..gaussian_hidden.shape().dims::<3>()[2] as i32,
    ]);
    let gaussian_params = head
        .gaussian_head
        .forward(gaussian_tokens, (h, w))
        .reshape([
            b as i32,
            v as i32,
            (out_h * out_w) as i32,
            cfg.num_surfaces as i32,
            cfg.raw_gaussian_dim() as i32,
        ]);
    report(
        "gaussian_params",
        tensor_to_vec(gaussian_params.clone())?,
        gaussian_params_ref_v,
    );

    let camera_tokens = camera_hidden.clone().slice([
        0..bv as i32,
        patch_start_idx as i32..camera_hidden.shape().dims::<3>()[1] as i32,
        0..camera_hidden.shape().dims::<3>()[2] as i32,
    ]);
    let mut camera_poses = head
        .camera_head
        .forward_pose(camera_tokens, patch_h, patch_w)
        .reshape([b as i32, v as i32, 4, 4]);

    let first_pose = camera_poses.clone().slice([0..b as i32, 0..1, 0..4, 0..4]);
    let first_pose = first_pose.reshape([b as i32, 4, 4]);
    let first_inv = se3_inverse_flat(first_pose)
        .reshape([b as i32, 1, 4, 4])
        .repeat_dim(1, v)
        .reshape([bv as i32, 4, 4]);

    camera_poses = first_inv
        .matmul(camera_poses.clone().reshape([bv as i32, 4, 4]))
        .reshape([b as i32, v as i32, 4, 4]);

    report(
        "camera_poses",
        tensor_to_vec(camera_poses.clone())?,
        camera_poses_ref_v,
    );

    let points = local_points
        .clone()
        .reshape([b as i32, v as i32, (out_h * out_w) as i32, 3])
        .reshape([b as i32, v as i32, (out_h * out_w) as i32, 1, 3])
        .repeat_dim(3, cfg.num_surfaces)
        .unsqueeze_dim(4);
    let depths = points
        .clone()
        .slice([
            0..b as i32,
            0..v as i32,
            0..(out_h * out_w) as i32,
            0..cfg.num_surfaces as i32,
            0..1,
            2..3,
        ])
        .reshape([
            b as i32,
            v as i32,
            (out_h * out_w) as i32,
            cfg.num_surfaces as i32,
            1,
            1,
        ]);
    let densities = sigmoid(
        gaussian_params
            .clone()
            .slice([
                0..b as i32,
                0..v as i32,
                0..(out_h * out_w) as i32,
                0..cfg.num_surfaces as i32,
                0..1,
            ])
            .reshape([
                b as i32,
                v as i32,
                (out_h * out_w) as i32,
                cfg.num_surfaces as i32,
            ]),
    )
    .unsqueeze_dim(4);
    let opacity = head.map_pdf_to_opacity(densities, 0);
    let raw_gaussians = gaussian_params
        .clone()
        .slice([
            0..b as i32,
            0..v as i32,
            0..(out_h * out_w) as i32,
            0..cfg.num_surfaces as i32,
            1..cfg.raw_gaussian_dim() as i32,
        ])
        .unsqueeze_dim(4);
    let extrinsics = camera_poses
        .reshape([b as i32, v as i32, 1, 1, 1, 16])
        .repeat_dim(2, out_h * out_w)
        .repeat_dim(3, cfg.num_surfaces);

    let gauss =
        head.gaussian_adapter
            .forward_spp(points, depths, opacity, raw_gaussians, Some(extrinsics));
    report("gaussians_means", tensor_to_vec(gauss.means)?, means_ref_v);
    report(
        "gaussians_harmonics",
        tensor_to_vec(gauss.harmonics)?,
        harm_ref_v,
    );
    report(
        "gaussians_opacities",
        tensor_to_vec(gauss.opacities)?,
        opa_ref_v,
    );
    report("gaussians_scales", tensor_to_vec(gauss.scales)?, scl_ref_v);
    report(
        "gaussians_rotations",
        tensor_to_vec(gauss.rotations)?,
        rot_ref_v,
    );

    Ok(())
}
