#![cfg(feature = "import")]
#![cfg(feature = "backend_ndarray")]

use std::path::Path;

use burn::prelude::*;
use burn::tensor::activation::sigmoid;
use burn_gaussian_splatting::{
    correctness::{compute_stats, load_safetensors, read_tensor, tensor_to_vec},
    import::load_yono_head_from_safetensors,
    model::{YonoHeadConfig, YonoHeadInput},
};

type BackendImpl = burn::backend::NdArray<f32>;

const WEIGHTS_PATH: &str = "assets/models/yono_head_weights.safetensors";
const REFERENCE_PATH: &str = "assets/fixtures/yono_head_reference.safetensors";

fn should_skip() -> bool {
    !(Path::new(WEIGHTS_PATH).exists() && Path::new(REFERENCE_PATH).exists())
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

fn assert_tensor_close(
    name: &'static str,
    actual: Vec<f32>,
    expected: Vec<f32>,
    mean_abs: f32,
    max_abs: f32,
    mse: f32,
) {
    let stats = compute_stats(&actual, &expected)
        .unwrap_or_else(|err| panic!("failed to compute stats for {name}: {err}"));

    assert!(
        stats.within(mean_abs, max_abs, mse),
        "{name} mismatch: mean_abs={}, max_abs={}, max_rel={}, mse={}",
        stats.mean_abs,
        stats.max_abs,
        stats.max_rel,
        stats.mse
    );
}

fn rel_rmse(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: {} vs {}",
        actual.len(),
        expected.len()
    );

    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&a, &e) in actual.iter().zip(expected.iter()) {
        let d = (a - e) as f64;
        num += d * d;
        den += (e as f64) * (e as f64);
    }

    if den <= f64::EPSILON {
        return 0.0;
    }

    (num / den).sqrt() as f32
}

fn read_tensor_optional(
    tensors: &safetensors::tensor::SafeTensors<'_>,
    name: &'static str,
) -> Option<(Vec<f32>, Vec<usize>)> {
    read_tensor(tensors, name).ok()
}

#[test]
fn yono_head_pipeline_matches_python_reference() -> Result<(), Box<dyn std::error::Error>> {
    if should_skip() {
        eprintln!(
            "Skipping parity test because fixtures are missing. Run `python tool/export_yono_head_reference.py` first."
        );
        return Ok(());
    }

    let device = <BackendImpl as Backend>::Device::default();

    let (model, apply_result) = load_yono_head_from_safetensors::<BackendImpl>(
        &device,
        YonoHeadConfig::new(),
        Path::new(WEIGHTS_PATH),
    )?;
    eprintln!(
        "apply: applied={}, missing={}, unused={}, skipped={}",
        apply_result.applied.len(),
        apply_result.missing.len(),
        apply_result.unused.len(),
        apply_result.skipped.len()
    );
    if !apply_result.missing.is_empty() {
        eprintln!("missing: {:?}", apply_result.missing);
    }
    if !apply_result.unused.is_empty() {
        eprintln!("unused: {:?}", apply_result.unused);
    }

    let tensors = load_safetensors(REFERENCE_PATH)?;

    let (hidden_v, hidden_s) = read_tensor(&tensors, "hidden")?;
    let (pos_v, pos_s) = read_tensor(&tensors, "pos")?;
    let (hidden_up_v, hidden_up_s) = read_tensor(&tensors, "hidden_upsampled")?;
    let (pos_up_v, pos_up_s) = read_tensor(&tensors, "pos_upsampled")?;
    let (image_v, image_s) = read_tensor(&tensors, "image")?;
    let (extrinsics_v, extrinsics_s) = read_tensor(&tensors, "extrinsics")?;

    let hidden = tensor_from_f32::<3>(hidden_v, [hidden_s[0], hidden_s[1], hidden_s[2]], &device);
    let pos = tensor_from_f32::<3>(pos_v, [pos_s[0], pos_s[1], pos_s[2]], &device);
    let hidden_upsampled = tensor_from_f32::<3>(
        hidden_up_v,
        [hidden_up_s[0], hidden_up_s[1], hidden_up_s[2]],
        &device,
    );
    let pos_upsampled =
        tensor_from_f32::<3>(pos_up_v, [pos_up_s[0], pos_up_s[1], pos_up_s[2]], &device);
    let image = tensor_from_f32::<5>(
        image_v,
        [image_s[0], image_s[1], image_s[2], image_s[3], image_s[4]],
        &device,
    );
    let extrinsics = tensor_from_f32::<4>(
        extrinsics_v,
        [
            extrinsics_s[0],
            extrinsics_s[1],
            extrinsics_s[2],
            extrinsics_s[3],
        ],
        &device,
    );

    let output = model.forward(YonoHeadInput {
        image,
        hidden,
        pos,
        hidden_upsampled: Some(hidden_upsampled),
        pos_upsampled: Some(pos_upsampled),
        patch_start_idx: 5,
        global_step: 0,
        training: false,
        extrinsics: Some(extrinsics),
        use_predicted_pose: false,
        scheduled_sampling_draw: None,
    });

    let (expected_point_hidden, _) = read_tensor(&tensors, "point_hidden")?;
    let (expected_point_out_raw, _) = read_tensor(&tensors, "point_out_raw")?;
    let (expected_gaussian_hidden, _) = read_tensor(&tensors, "gaussian_hidden")?;
    let (expected_camera_hidden, _) = read_tensor(&tensors, "camera_hidden")?;
    let (expected_local_points, expected_local_points_s) = read_tensor(&tensors, "local_points")?;
    let (expected_gaussian_params, expected_gaussian_params_s) =
        read_tensor(&tensors, "gaussian_params")?;
    let (expected_camera_poses, expected_camera_poses_s) = read_tensor(&tensors, "camera_poses")?;

    let (expected_means, _) = read_tensor(&tensors, "gaussians_means")?;
    let (expected_covs, _) = read_tensor(&tensors, "gaussians_covariances")?;
    let (expected_harmonics, _) = read_tensor(&tensors, "gaussians_harmonics")?;
    let (expected_opacities, _) = read_tensor(&tensors, "gaussians_opacities")?;
    let (expected_rotations, _) = read_tensor(&tensors, "gaussians_rotations")?;
    let (expected_scales, _) = read_tensor(&tensors, "gaussians_scales")?;
    let expected_means_spp = read_tensor_optional(&tensors, "gaussians_means_spp");
    let expected_covs_spp = read_tensor_optional(&tensors, "gaussians_covariances_spp");
    let expected_harmonics_spp = read_tensor_optional(&tensors, "gaussians_harmonics_spp");
    let expected_opacities_spp = read_tensor_optional(&tensors, "gaussians_opacities_spp");
    let expected_rotations_spp = read_tensor_optional(&tensors, "gaussians_rotations_spp");
    let expected_scales_spp = read_tensor_optional(&tensors, "gaussians_scales_spp");

    assert_tensor_close(
        "point_hidden",
        tensor_to_vec(output.point_hidden.clone())?,
        expected_point_hidden,
        2e-3,
        2e-2,
        2e-4,
    );
    assert_tensor_close(
        "gaussian_hidden",
        tensor_to_vec(output.gaussian_hidden)?,
        expected_gaussian_hidden,
        2e-3,
        2e-2,
        2e-4,
    );
    assert_tensor_close(
        "camera_hidden",
        tensor_to_vec(output.camera_hidden)?,
        expected_camera_hidden,
        2e-3,
        2e-2,
        2e-4,
    );

    let [b, v, h, w, _] = output.local_points.shape().dims::<5>();
    let point_tokens = output.point_hidden.clone().slice([
        0..(b * v) as i32,
        5..output.point_hidden.shape().dims::<3>()[1] as i32,
        0..output.point_hidden.shape().dims::<3>()[2] as i32,
    ]);
    let point_out_raw = model
        .point_head
        .forward(point_tokens, (h, w))
        .reshape([b as i32, v as i32, h as i32, w as i32, 3]);
    let point_out_raw_vec = tensor_to_vec(point_out_raw.clone())?;
    assert_tensor_close(
        "point_out_raw",
        point_out_raw_vec,
        expected_point_out_raw,
        2e-3,
        2e-2,
        2e-4,
    );

    let xy_from_raw =
        point_out_raw
            .clone()
            .slice([0..b as i32, 0..v as i32, 0..h as i32, 0..w as i32, 0..2]);
    let z_from_raw = point_out_raw
        .slice([0..b as i32, 0..v as i32, 0..h as i32, 0..w as i32, 2..3])
        .exp();
    let local_points_from_raw = Tensor::cat(
        vec![
            xy_from_raw * z_from_raw.clone().repeat_dim(4, 2),
            z_from_raw,
        ],
        4,
    );
    assert_tensor_close(
        "local_points_internal",
        tensor_to_vec(output.local_points.clone())?,
        tensor_to_vec(local_points_from_raw)?,
        1e-6,
        1e-5,
        1e-7,
    );
    assert_tensor_close(
        "gaussian_params",
        tensor_to_vec(output.gaussian_params)?,
        expected_gaussian_params.clone(),
        2e-3,
        2e-2,
        2e-4,
    );
    assert_tensor_close(
        "camera_poses",
        tensor_to_vec(output.camera_poses)?,
        expected_camera_poses.clone(),
        3e-3,
        3e-2,
        4e-4,
    );

    let expected_local_points_t = tensor_from_f32::<5>(
        expected_local_points.clone(),
        [
            expected_local_points_s[0],
            expected_local_points_s[1],
            expected_local_points_s[2],
            expected_local_points_s[3],
            expected_local_points_s[4],
        ],
        &device,
    );
    let [b_ref, v_ref, h_ref, w_ref, _] = expected_local_points_s
        .as_slice()
        .try_into()
        .expect("expected local_points shape must be rank-5");
    let r_ref = h_ref * w_ref;

    let expected_gaussian_params_t = tensor_from_f32::<5>(
        expected_gaussian_params.clone(),
        [
            expected_gaussian_params_s[0],
            expected_gaussian_params_s[1],
            expected_gaussian_params_s[2],
            expected_gaussian_params_s[3],
            expected_gaussian_params_s[4],
        ],
        &device,
    );
    let expected_camera_poses_t = tensor_from_f32::<4>(
        expected_camera_poses.clone(),
        [
            expected_camera_poses_s[0],
            expected_camera_poses_s[1],
            expected_camera_poses_s[2],
            expected_camera_poses_s[3],
        ],
        &device,
    );

    let points_ref = expected_local_points_t
        .reshape([b_ref as i32, v_ref as i32, r_ref as i32, 3])
        .reshape([b_ref as i32, v_ref as i32, r_ref as i32, 1, 3]);
    let depths_ref = points_ref
        .clone()
        .slice([
            0..b_ref as i32,
            0..v_ref as i32,
            0..r_ref as i32,
            0..1,
            2..3,
        ])
        .reshape([b_ref as i32, v_ref as i32, r_ref as i32, 1, 1]);
    let densities_ref = sigmoid(
        expected_gaussian_params_t
            .clone()
            .slice([
                0..expected_gaussian_params_s[0] as i32,
                0..expected_gaussian_params_s[1] as i32,
                0..r_ref as i32,
                0..expected_gaussian_params_s[3] as i32,
                0..1,
            ])
            .reshape([
                expected_gaussian_params_s[0] as i32,
                expected_gaussian_params_s[1] as i32,
                r_ref as i32,
                expected_gaussian_params_s[3] as i32,
            ]),
    );
    let opacity_ref = model.map_pdf_to_opacity(densities_ref, 0);
    let raw_gaussians_ref = expected_gaussian_params_t.clone().slice([
        0..expected_gaussian_params_s[0] as i32,
        0..expected_gaussian_params_s[1] as i32,
        0..r_ref as i32,
        0..expected_gaussian_params_s[3] as i32,
        1..expected_gaussian_params_s[4] as i32,
    ]);
    let extrinsics_ref = expected_camera_poses_t
        .reshape([
            expected_camera_poses_s[0] as i32,
            expected_camera_poses_s[1] as i32,
            1,
            1,
            expected_camera_poses_s[2] as i32,
            expected_camera_poses_s[3] as i32,
        ])
        .repeat_dim(2, r_ref)
        .repeat_dim(3, expected_gaussian_params_s[3]);

    let gaussians_ref = model.gaussian_adapter.forward(
        points_ref,
        depths_ref,
        opacity_ref,
        raw_gaussians_ref,
        Some(extrinsics_ref),
    );

    assert_tensor_close(
        "adapter_only_gaussians_means",
        tensor_to_vec(gaussians_ref.means)?,
        expected_means.clone(),
        5e-4,
        5e-1,
        1e-4,
    );
    assert_tensor_close(
        "adapter_only_gaussians_covariances",
        tensor_to_vec(gaussians_ref.covariances)?,
        expected_covs.clone(),
        3e-3,
        1e-1,
        1e-3,
    );
    assert_tensor_close(
        "adapter_only_gaussians_harmonics",
        tensor_to_vec(gaussians_ref.harmonics)?,
        expected_harmonics.clone(),
        3e-3,
        3e-2,
        4e-4,
    );
    assert_tensor_close(
        "adapter_only_gaussians_opacities",
        tensor_to_vec(gaussians_ref.opacities)?,
        expected_opacities.clone(),
        3e-3,
        3e-2,
        4e-4,
    );
    assert_tensor_close(
        "adapter_only_gaussians_rotations",
        tensor_to_vec(gaussians_ref.rotations)?,
        expected_rotations.clone(),
        3e-3,
        3e-2,
        4e-4,
    );
    assert_tensor_close(
        "adapter_only_gaussians_scales",
        tensor_to_vec(gaussians_ref.scales)?,
        expected_scales.clone(),
        3e-3,
        3e-2,
        4e-4,
    );

    let gaussians_means_vec = tensor_to_vec(output.gaussians_structured.means)?;
    let gaussians_means_rel = rel_rmse(&gaussians_means_vec, &expected_means);
    assert!(
        gaussians_means_rel <= 5e-2,
        "gaussians_means rel_rmse too high: {gaussians_means_rel}"
    );
    assert_tensor_close(
        "gaussians_covariances",
        tensor_to_vec(output.gaussians_structured.covariances)?,
        expected_covs,
        3e-3,
        3e-2,
        4e-4,
    );
    assert_tensor_close(
        "gaussians_harmonics",
        tensor_to_vec(output.gaussians_structured.harmonics)?,
        expected_harmonics,
        3e-3,
        3e-2,
        4e-4,
    );
    assert_tensor_close(
        "gaussians_opacities",
        tensor_to_vec(output.gaussians_structured.opacities)?,
        expected_opacities,
        3e-3,
        3e-2,
        4e-4,
    );
    assert_tensor_close(
        "gaussians_rotations",
        tensor_to_vec(output.gaussians_structured.rotations)?,
        expected_rotations,
        3e-3,
        3e-2,
        4e-4,
    );
    assert_tensor_close(
        "gaussians_scales",
        tensor_to_vec(output.gaussians_structured.scales)?,
        expected_scales,
        3e-3,
        3e-2,
        4e-4,
    );

    if let Some((expected, _)) = expected_means_spp {
        assert_tensor_close(
            "gaussians_means_spp",
            tensor_to_vec(output.gaussians_structured_spp.means)?,
            expected,
            5e-4,
            5e-1,
            1e-4,
        );
    }
    if let Some((expected, _)) = expected_covs_spp {
        assert_tensor_close(
            "gaussians_covariances_spp",
            tensor_to_vec(output.gaussians_structured_spp.covariances)?,
            expected,
            3e-3,
            3e-2,
            4e-4,
        );
    }
    if let Some((expected, _)) = expected_harmonics_spp {
        assert_tensor_close(
            "gaussians_harmonics_spp",
            tensor_to_vec(output.gaussians_structured_spp.harmonics)?,
            expected,
            3e-3,
            3e-2,
            4e-4,
        );
    }
    if let Some((expected, _)) = expected_opacities_spp {
        assert_tensor_close(
            "gaussians_opacities_spp",
            tensor_to_vec(output.gaussians_structured_spp.opacities)?,
            expected,
            3e-3,
            3e-2,
            4e-4,
        );
    }
    if let Some((expected, _)) = expected_rotations_spp {
        assert_tensor_close(
            "gaussians_rotations_spp",
            tensor_to_vec(output.gaussians_structured_spp.rotations)?,
            expected,
            3e-3,
            3e-2,
            4e-4,
        );
    }
    if let Some((expected, _)) = expected_scales_spp {
        assert_tensor_close(
            "gaussians_scales_spp",
            tensor_to_vec(output.gaussians_structured_spp.scales)?,
            expected,
            3e-3,
            3e-2,
            4e-4,
        );
    }

    Ok(())
}
