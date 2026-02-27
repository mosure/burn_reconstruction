use burn::{module::Ignored, prelude::*, tensor::activation::sigmoid};

use super::{
    camera_head::{CameraHead, CameraHeadConfig},
    config::YonoHeadConfig,
    gaussian::{
        flatten_structured_gaussians_spp, squeeze_spp_structured_gaussians, FlatGaussians,
        StructuredGaussians, StructuredGaussiansSpp, UnifiedGaussianAdapter,
    },
    ops::{interpolate_bilinear_align_corners_false, position_getter, se3_inverse_flat},
    patch_embed::{PatchEmbedNorm, PatchEmbedNormConfig},
    transformer_head::{LinearPts3d, LinearPts3dConfig, TransformerDecoder},
};

#[derive(Debug)]
pub struct YonoHeadInput<B: Backend> {
    pub image: Tensor<B, 5>,
    pub hidden: Tensor<B, 3>,
    pub pos: Tensor<B, 3>,
    pub hidden_upsampled: Option<Tensor<B, 3>>,
    pub pos_upsampled: Option<Tensor<B, 3>>,
    pub patch_start_idx: usize,
    pub global_step: usize,
    pub training: bool,
    pub extrinsics: Option<Tensor<B, 4>>,
    pub use_predicted_pose: bool,
    pub scheduled_sampling_draw: Option<f32>,
}

#[derive(Debug)]
pub struct YonoHeadOutput<B: Backend> {
    pub point_hidden: Tensor<B, 3>,
    pub gaussian_hidden: Tensor<B, 3>,
    pub camera_hidden: Tensor<B, 3>,
    pub local_points: Tensor<B, 5>,
    pub gaussian_params: Tensor<B, 5>,
    pub camera_poses: Tensor<B, 4>,
    pub gaussians_structured_spp: StructuredGaussiansSpp<B>,
    pub gaussians_structured: StructuredGaussians<B>,
    pub gaussians_flat: FlatGaussians<B>,
}

#[derive(Module, Debug)]
pub struct YonoHeadPipeline<B: Backend> {
    pub point_decoder: TransformerDecoder<B>,
    pub gaussian_decoder: TransformerDecoder<B>,
    pub camera_decoder: TransformerDecoder<B>,
    pub point_head: LinearPts3d<B>,
    pub gaussian_head: LinearPts3d<B>,
    pub camera_head: CameraHead<B>,
    pub rgb_embed: PatchEmbedNorm<B>,
    pub gaussian_adapter: UnifiedGaussianAdapter<B>,
    cfg: Ignored<YonoHeadConfig>,
}

impl<B: Backend> YonoHeadPipeline<B> {
    pub fn new(device: &B::Device, cfg: YonoHeadConfig) -> Self {
        let point_decoder = TransformerDecoder::new(device, &cfg.point_decoder);
        let gaussian_decoder = if cfg.share_point_decoder_init {
            point_decoder.clone()
        } else {
            TransformerDecoder::new(device, &cfg.gaussian_decoder)
        };
        let camera_decoder = TransformerDecoder::new(device, &cfg.camera_decoder);

        let point_head = LinearPts3d::new(
            device,
            &LinearPts3dConfig::new(cfg.point_patch_size(), cfg.point_decoder.out_dim)
                .with_output_dim(3)
                .with_downsample_ratio(cfg.gaussian_downsample_ratio)
                .with_points_per_axis(Some(cfg.point_per_axis())),
        );

        let gaussian_head = LinearPts3d::new(
            device,
            &LinearPts3dConfig::new(cfg.point_patch_size(), cfg.gaussian_decoder.out_dim)
                .with_output_dim(cfg.raw_gaussian_dim())
                .with_downsample_ratio(cfg.gaussian_downsample_ratio)
                .with_points_per_axis(Some(cfg.point_per_axis())),
        );

        let camera_head = CameraHead::new(
            device,
            &CameraHeadConfig::new().with_dim(cfg.camera_decoder.out_dim),
        );
        let rgb_embed = PatchEmbedNormConfig::new(
            cfg.patch_size / cfg.upscale_token_ratio,
            3,
            cfg.point_decoder.in_dim,
        )
        .init(device);

        let gaussian_adapter = UnifiedGaussianAdapter::new(device, &cfg.gaussian_adapter);

        Self {
            point_decoder,
            gaussian_decoder,
            camera_decoder,
            point_head,
            gaussian_head,
            camera_head,
            rgb_embed,
            gaussian_adapter,
            cfg: Ignored(cfg),
        }
    }

    pub fn config(&self) -> &YonoHeadConfig {
        &self.cfg.0
    }

    pub fn map_pdf_to_opacity<const D: usize>(
        &self,
        pdf: Tensor<B, D>,
        global_step: usize,
    ) -> Tensor<B, D> {
        let cfg = &self.cfg.0.opacity_mapping;
        let warmup = cfg.warm_up.max(1);
        let progress = (global_step as f32 / warmup as f32).min(1.0);
        let x = cfg.initial + progress * (cfg.final_value - cfg.initial);
        let exponent = 2f32.powf(x);

        let one = Tensor::<B, D>::ones_like(&pdf);
        let a = (one.clone() - pdf.clone()).powf_scalar(exponent);
        let b = pdf.clone().powf_scalar(1.0 / exponent);
        (one - a + b).mul_scalar(0.5)
    }

    #[allow(clippy::too_many_lines)]
    pub fn forward(&self, input: YonoHeadInput<B>) -> YonoHeadOutput<B> {
        let cfg = &self.cfg.0;

        let [b, v, _, h, w] = input.image.shape().dims::<5>();
        let patch_h = h / cfg.patch_size;
        let patch_w = w / cfg.patch_size;
        let bv = b * v;

        let hidden = input.hidden;
        let pos = input.pos;

        let (hidden_upsampled, pos_upsampled) =
            if let (Some(hidden_upsampled), Some(pos_upsampled)) =
                (input.hidden_upsampled, input.pos_upsampled)
            {
                (hidden_upsampled, pos_upsampled)
            } else if cfg.upscale_token_ratio > 1 {
                let hidden_dims = hidden.shape().dims::<3>();
                let img_tokens = hidden_dims[1] - input.patch_start_idx;

                let hidden_aux = hidden.clone().slice([
                    0..bv as i32,
                    0..input.patch_start_idx as i32,
                    0..hidden_dims[2] as i32,
                ]);
                let hidden_img = hidden.clone().slice([
                    0..bv as i32,
                    input.patch_start_idx as i32..(input.patch_start_idx + img_tokens) as i32,
                    0..hidden_dims[2] as i32,
                ]);

                let hidden_img = hidden_img
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

                let pos_aux =
                    pos.clone()
                        .slice([0..bv as i32, 0..input.patch_start_idx as i32, 0..2]);
                let mut pos_img = position_getter(
                    bv,
                    patch_h * cfg.upscale_token_ratio,
                    patch_w * cfg.upscale_token_ratio,
                    &hidden_upsampled.device(),
                );
                if input.patch_start_idx > 0 {
                    pos_img = pos_img + 1.0;
                }
                let pos_upsampled = Tensor::cat(vec![pos_aux, pos_img], 1);

                (hidden_upsampled, pos_upsampled)
            } else {
                (hidden.clone(), pos.clone())
            };

        let rgb = input.image.reshape([bv as i32, 3, h as i32, w as i32]);
        let rgb_feat = self.rgb_embed.forward(rgb);

        let up_dims = hidden_upsampled.shape().dims::<3>();
        let hidden_aux = hidden_upsampled.clone().slice([
            0..bv as i32,
            0..input.patch_start_idx as i32,
            0..up_dims[2] as i32,
        ]);
        let hidden_img = hidden_upsampled.clone().slice([
            0..bv as i32,
            input.patch_start_idx as i32..up_dims[1] as i32,
            0..up_dims[2] as i32,
        ]) + rgb_feat;
        let hidden_gaussian = Tensor::cat(vec![hidden_aux, hidden_img], 1);

        let point_hidden = self
            .point_decoder
            .forward(hidden_upsampled.clone(), Some(&pos_upsampled));
        let gaussian_hidden = self
            .gaussian_decoder
            .forward(hidden_gaussian, Some(&pos_upsampled));
        let camera_hidden = self.camera_decoder.forward(hidden.clone(), Some(&pos));

        let gaussians_per_axis = cfg.effective_gaussians_per_axis();
        let out_h = patch_h * gaussians_per_axis;
        let out_w = patch_w * gaussians_per_axis;

        let point_tokens = point_hidden.clone().slice([
            0..bv as i32,
            input.patch_start_idx as i32..point_hidden.shape().dims::<3>()[1] as i32,
            0..point_hidden.shape().dims::<3>()[2] as i32,
        ]);
        let point_pred = self.point_head.forward(point_tokens, (h, w)).reshape([
            b as i32,
            v as i32,
            out_h as i32,
            out_w as i32,
            3,
        ]);

        let xy = point_pred.clone().slice([
            0..b as i32,
            0..v as i32,
            0..out_h as i32,
            0..out_w as i32,
            0..2,
        ]);
        let z = point_pred
            .slice([
                0..b as i32,
                0..v as i32,
                0..out_h as i32,
                0..out_w as i32,
                2..3,
            ])
            .exp();
        let local_points = Tensor::cat(vec![xy * z.clone().repeat_dim(4, 2), z], 4);

        let gaussian_tokens = gaussian_hidden.clone().slice([
            0..bv as i32,
            input.patch_start_idx as i32..gaussian_hidden.shape().dims::<3>()[1] as i32,
            0..gaussian_hidden.shape().dims::<3>()[2] as i32,
        ]);
        let gaussian_params = self
            .gaussian_head
            .forward(gaussian_tokens, (h, w))
            .reshape([
                b as i32,
                v as i32,
                out_h as i32,
                out_w as i32,
                cfg.raw_gaussian_dim() as i32,
            ]);

        let camera_tokens = camera_hidden.clone().slice([
            0..bv as i32,
            input.patch_start_idx as i32..camera_hidden.shape().dims::<3>()[1] as i32,
            0..camera_hidden.shape().dims::<3>()[2] as i32,
        ]);
        let mut camera_poses = self
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

        let gauss = gaussian_params.reshape([
            b as i32,
            v as i32,
            (out_h * out_w) as i32,
            cfg.num_surfaces as i32,
            cfg.raw_gaussian_dim() as i32,
        ]);

        let densities = sigmoid(
            gauss
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

        let c2w = Self::select_c2w(
            cfg,
            camera_poses.clone(),
            input.extrinsics,
            input.use_predicted_pose,
            input.training,
            input.global_step,
            input.scheduled_sampling_draw,
        );

        let opacity = self.map_pdf_to_opacity(densities, input.global_step);

        let raw_gaussians = gauss
            .clone()
            .slice([
                0..b as i32,
                0..v as i32,
                0..(out_h * out_w) as i32,
                0..cfg.num_surfaces as i32,
                1..cfg.raw_gaussian_dim() as i32,
            ])
            .unsqueeze_dim(4);

        let extrinsics = c2w
            .reshape([b as i32, v as i32, 1, 1, 1, 16])
            .repeat_dim(2, out_h * out_w)
            .repeat_dim(3, cfg.num_surfaces);

        let gaussians_structured_spp = self.gaussian_adapter.forward_spp(
            points,
            depths,
            opacity,
            raw_gaussians,
            Some(extrinsics),
        );

        let gaussians_structured = squeeze_spp_structured_gaussians(StructuredGaussiansSpp {
            means: gaussians_structured_spp.means.clone(),
            covariances: gaussians_structured_spp.covariances.clone(),
            harmonics: gaussians_structured_spp.harmonics.clone(),
            opacities: gaussians_structured_spp.opacities.clone(),
            rotations: gaussians_structured_spp.rotations.clone(),
            scales: gaussians_structured_spp.scales.clone(),
        });

        let gaussians_flat = flatten_structured_gaussians_spp(StructuredGaussiansSpp {
            means: gaussians_structured_spp.means.clone(),
            covariances: gaussians_structured_spp.covariances.clone(),
            harmonics: gaussians_structured_spp.harmonics.clone(),
            opacities: gaussians_structured_spp.opacities.clone(),
            rotations: gaussians_structured_spp.rotations.clone(),
            scales: gaussians_structured_spp.scales.clone(),
        });

        YonoHeadOutput {
            point_hidden,
            gaussian_hidden,
            camera_hidden,
            local_points,
            gaussian_params: gauss,
            camera_poses,
            gaussians_structured_spp,
            gaussians_structured,
            gaussians_flat,
        }
    }

    fn select_c2w(
        cfg: &YonoHeadConfig,
        camera_poses: Tensor<B, 4>,
        extrinsics: Option<Tensor<B, 4>>,
        use_predicted_pose: bool,
        training: bool,
        global_step: usize,
        scheduled_sampling_draw: Option<f32>,
    ) -> Tensor<B, 4> {
        if !cfg.pose_free {
            return extrinsics.unwrap_or(camera_poses);
        }

        let Some(extrinsics) = extrinsics else {
            return camera_poses;
        };

        if !use_predicted_pose {
            return extrinsics;
        }

        if !training {
            return camera_poses;
        }

        let draw = scheduled_sampling_draw.unwrap_or(1.0).clamp(0.0, 1.0);
        let prob_use_gt = cfg.scheduled_pose_epsilon(global_step);
        if draw < prob_use_gt {
            extrinsics
        } else {
            camera_poses
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{YonoHeadConfig, YonoHeadPipeline};
    use burn::{prelude::*, tensor::Tensor};

    type TestBackend = burn::backend::NdArray<f32>;

    fn poses_with_translation(
        device: &<TestBackend as Backend>::Device,
        tx: f32,
    ) -> Tensor<TestBackend, 4> {
        Tensor::<TestBackend, 1>::from_floats(
            [
                1.0f32, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0, //
                1.0, 0.0, 0.0, tx, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0, //
            ]
            .as_slice(),
            device,
        )
        .reshape([1, 2, 4, 4])
    }

    fn as_vec(tensor: Tensor<TestBackend, 4>) -> Vec<f32> {
        tensor
            .into_data()
            .to_vec::<f32>()
            .expect("tensor must be readable")
    }

    #[test]
    fn scheduled_pose_selection_matches_expected_modes() {
        let device = <TestBackend as Backend>::Device::default();
        let predicted = poses_with_translation(&device, 2.0);
        let gt = poses_with_translation(&device, 0.5);

        let pose_free_off = YonoHeadConfig::new().with_pose_free(false);
        let out = YonoHeadPipeline::<TestBackend>::select_c2w(
            &pose_free_off,
            predicted.clone(),
            Some(gt.clone()),
            true,
            true,
            2000,
            Some(0.1),
        );
        assert_eq!(as_vec(out), as_vec(gt.clone()));

        let pose_free_on = YonoHeadConfig::new()
            .with_pose_free(true)
            .with_gt_pose_sampling_decay_start_step(1000)
            .with_gt_pose_sampling_decay_end_step(5000)
            .with_gt_pose_final_sample_ratio(0.9);

        let out = YonoHeadPipeline::<TestBackend>::select_c2w(
            &pose_free_on,
            predicted.clone(),
            Some(gt.clone()),
            false,
            true,
            2000,
            Some(0.1),
        );
        assert_eq!(as_vec(out), as_vec(gt.clone()));

        let out = YonoHeadPipeline::<TestBackend>::select_c2w(
            &pose_free_on,
            predicted.clone(),
            Some(gt.clone()),
            true,
            false,
            2000,
            Some(0.1),
        );
        assert_eq!(as_vec(out), as_vec(predicted.clone()));

        let out = YonoHeadPipeline::<TestBackend>::select_c2w(
            &pose_free_on,
            predicted.clone(),
            Some(gt.clone()),
            true,
            true,
            2000,
            Some(0.1),
        );
        assert_eq!(as_vec(out), as_vec(gt.clone()));

        let out = YonoHeadPipeline::<TestBackend>::select_c2w(
            &pose_free_on,
            predicted.clone(),
            Some(gt),
            true,
            true,
            2000,
            Some(0.99),
        );
        assert_eq!(as_vec(out), as_vec(predicted));
    }
}
