use burn::{module::Ignored, prelude::*};
use burn_dino::model::dino::{DinoVisionTransformer, DinoVisionTransformerConfig};

use super::{
    config::YonoHeadConfig,
    ops::position_getter,
    yono_head::{YonoHeadInput, YonoHeadOutput, YonoHeadPipeline},
};

#[derive(Debug)]
pub struct YonoEncoderInput<B: Backend> {
    pub image: Tensor<B, 5>,
    pub global_step: usize,
    pub training: bool,
    pub extrinsics: Option<Tensor<B, 4>>,
    pub use_predicted_pose: bool,
    pub scheduled_sampling_draw: Option<f32>,
}

#[derive(Debug)]
pub struct YonoEncoderOutput<B: Backend> {
    pub hidden: Tensor<B, 3>,
    pub pos: Tensor<B, 3>,
    pub patch_start_idx: usize,
    pub head_output: YonoHeadOutput<B>,
}

#[derive(Module, Debug)]
pub struct YonoEncoderPipeline<B: Backend> {
    pub encoder: DinoVisionTransformer<B>,
    pub encoder_projection: Option<nn::Linear<B>>,
    pub head: YonoHeadPipeline<B>,
    patch_start_idx: Ignored<usize>,
}

impl<B: Backend> YonoEncoderPipeline<B> {
    pub fn from_configs(
        device: &B::Device,
        dino_cfg: DinoVisionTransformerConfig,
        head_cfg: YonoHeadConfig,
    ) -> Self {
        let patch_start_idx = 1 + if dino_cfg.use_register_tokens {
            dino_cfg.register_token_count
        } else {
            0
        };

        let encoder = dino_cfg.clone().init(device);
        let head = YonoHeadPipeline::new(device, head_cfg.clone());

        let expected_in_dim = head_cfg.point_decoder.in_dim;
        let encoder_projection = if dino_cfg.embedding_dimension == expected_in_dim {
            None
        } else {
            Some(nn::LinearConfig::new(dino_cfg.embedding_dimension, expected_in_dim).init(device))
        };

        Self {
            encoder,
            encoder_projection,
            head,
            patch_start_idx: Ignored(patch_start_idx),
        }
    }

    pub fn patch_start_idx(&self) -> usize {
        self.patch_start_idx.0
    }

    pub fn forward(&self, input: YonoEncoderInput<B>) -> YonoEncoderOutput<B> {
        let [b, v, _, h, w] = input.image.shape().dims::<5>();
        let bv = b * v;
        let patch_h = h / self.head.config().patch_size;
        let patch_w = w / self.head.config().patch_size;

        let image_bv = input
            .image
            .clone()
            .reshape([bv as i32, 3, h as i32, w as i32]);
        let encoder_out = self.encoder.forward(image_bv, None);
        let mut hidden = encoder_out.x_prenorm;

        if let Some(proj) = &self.encoder_projection {
            hidden = proj.forward(hidden);
        }

        let expected_tokens = self.patch_start_idx.0 + patch_h * patch_w;
        let hidden_tokens = hidden.shape().dims::<3>()[1];
        assert_eq!(
            hidden_tokens, expected_tokens,
            "encoder token count mismatch: got {hidden_tokens}, expected {expected_tokens}"
        );

        let device = hidden.device();
        let pos_aux = Tensor::<B, 3>::zeros([bv as i32, self.patch_start_idx.0 as i32, 2], &device);
        let mut pos_img = position_getter(bv, patch_h, patch_w, &device);
        if self.patch_start_idx.0 > 0 {
            pos_img = pos_img + 1.0;
        }
        let pos = Tensor::cat(vec![pos_aux, pos_img], 1);

        let head_output = self.head.forward(YonoHeadInput {
            image: input.image,
            hidden: hidden.clone(),
            pos: pos.clone(),
            hidden_upsampled: None,
            pos_upsampled: None,
            patch_start_idx: self.patch_start_idx.0,
            global_step: input.global_step,
            training: input.training,
            extrinsics: input.extrinsics,
            use_predicted_pose: input.use_predicted_pose,
            scheduled_sampling_draw: input.scheduled_sampling_draw,
        });

        YonoEncoderOutput {
            hidden,
            pos,
            patch_start_idx: self.patch_start_idx.0,
            head_output,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{YonoEncoderInput, YonoEncoderPipeline};
    use crate::model::{
        ops::position_getter, TransformerDecoderSpec, YonoHeadConfig, YonoHeadInput,
    };
    use burn::{prelude::*, tensor::Tensor};
    use burn_dino::model::dino::DinoVisionTransformerConfig;

    type TestBackend = burn::backend::NdArray<f32>;

    fn small_head_config() -> YonoHeadConfig {
        let point_decoder = TransformerDecoderSpec::new()
            .with_in_dim(64)
            .with_embed_dim(32)
            .with_out_dim(32)
            .with_depth(1)
            .with_num_heads(4)
            .with_mlp_ratio(2.0)
            .with_need_project(true);

        let camera_decoder = TransformerDecoderSpec::new()
            .with_in_dim(64)
            .with_embed_dim(32)
            .with_out_dim(16)
            .with_depth(1)
            .with_num_heads(4)
            .with_mlp_ratio(2.0)
            .with_need_project(true);

        YonoHeadConfig::new()
            .with_patch_size(14)
            .with_gaussians_per_axis(2)
            .with_upscale_token_ratio(1)
            .with_num_surfaces(1)
            .with_point_decoder(point_decoder.clone())
            .with_gaussian_decoder(point_decoder)
            .with_camera_decoder(camera_decoder)
            .with_share_point_decoder_init(false)
    }

    #[test]
    fn wrapper_matches_manual_encoder_plus_head_path() {
        let device = <TestBackend as Backend>::Device::default();

        let dino_cfg =
            DinoVisionTransformerConfig::vits(Some(28), Some(14)).with_register_tokens(4);
        let head_cfg = small_head_config();
        let pipeline =
            YonoEncoderPipeline::<TestBackend>::from_configs(&device, dino_cfg, head_cfg);

        let b = 1usize;
        let v = 2usize;
        let h = 28usize;
        let w = 28usize;
        let image = Tensor::<TestBackend, 1>::from_floats(
            (0..b * v * 3 * h * w)
                .map(|idx| ((idx % 97) as f32 - 48.0) / 48.0)
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        )
        .reshape([b as i32, v as i32, 3, h as i32, w as i32]);

        let wrapped = pipeline.forward(YonoEncoderInput {
            image: image.clone(),
            global_step: 0,
            training: false,
            extrinsics: None,
            use_predicted_pose: true,
            scheduled_sampling_draw: Some(1.0),
        });

        let image_bv = image
            .clone()
            .reshape([(b * v) as i32, 3, h as i32, w as i32]);
        let mut hidden = pipeline.encoder.forward(image_bv, None).x_prenorm;
        if let Some(proj) = &pipeline.encoder_projection {
            hidden = proj.forward(hidden);
        }

        let patch_h = h / pipeline.head.config().patch_size;
        let patch_w = w / pipeline.head.config().patch_size;
        let patch_start = pipeline.patch_start_idx();
        let pos_aux = Tensor::<TestBackend, 3>::zeros(
            [(b * v) as i32, patch_start as i32, 2],
            &hidden.device(),
        );
        let mut pos_img = position_getter(b * v, patch_h, patch_w, &hidden.device());
        if patch_start > 0 {
            pos_img = pos_img + 1.0;
        }
        let pos = Tensor::cat(vec![pos_aux, pos_img], 1);

        let manual = pipeline.head.forward(YonoHeadInput {
            image,
            hidden,
            pos,
            hidden_upsampled: None,
            pos_upsampled: None,
            patch_start_idx: patch_start,
            global_step: 0,
            training: false,
            extrinsics: None,
            use_predicted_pose: true,
            scheduled_sampling_draw: Some(1.0),
        });

        let max_abs = |a: Vec<f32>, b: Vec<f32>| {
            a.into_iter()
                .zip(b)
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0f32, f32::max)
        };

        let wrapped_local = wrapped
            .head_output
            .local_points
            .into_data()
            .to_vec::<f32>()
            .expect("local points should be readable");
        let manual_local = manual
            .local_points
            .into_data()
            .to_vec::<f32>()
            .expect("manual local points should be readable");
        assert!(
            max_abs(wrapped_local, manual_local) <= 1e-6,
            "wrapper local point mismatch"
        );

        let wrapped_pose = wrapped
            .head_output
            .camera_poses
            .into_data()
            .to_vec::<f32>()
            .expect("camera poses should be readable");
        let manual_pose = manual
            .camera_poses
            .into_data()
            .to_vec::<f32>()
            .expect("manual camera poses should be readable");
        assert!(
            max_abs(wrapped_pose, manual_pose) <= 1e-6,
            "wrapper camera pose mismatch"
        );
    }
}
