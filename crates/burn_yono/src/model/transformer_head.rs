use burn::{
    prelude::*,
    tensor::activation::{gelu, relu},
};
use burn_dino::layers::{
    attention::AttentionConfig,
    block::{Block, BlockConfig},
    rope::RopeConfig,
};

use super::{config::TransformerDecoderSpec, ops::pixel_shuffle};

#[derive(Module, Debug)]
pub struct TransformerDecoder<B: Backend> {
    pub projects: Option<nn::Linear<B>>,
    pub blocks: Vec<Block<B>>,
    pub linear_out: nn::Linear<B>,
}

impl<B: Backend> TransformerDecoder<B> {
    pub fn new(device: &B::Device, spec: &TransformerDecoderSpec) -> Self {
        let projects = if spec.need_project {
            Some(nn::LinearConfig::new(spec.in_dim, spec.embed_dim).init(device))
        } else {
            None
        };

        let block_config = BlockConfig {
            attn: AttentionConfig {
                dim: spec.embed_dim,
                num_heads: spec.num_heads,
                qkv_bias: true,
                proj_bias: true,
                attn_drop: 0.0,
                proj_drop: 0.0,
                quiet_softmax: false,
                qk_norm: spec.qk_norm,
                rope: Some(RopeConfig {
                    base_frequency: spec.rope_frequency,
                }),
            },
            layer_scale: None,
            mlp_ratio: spec.mlp_ratio,
        };

        let mut blocks = Vec::with_capacity(spec.depth);
        for _ in 0..spec.depth {
            blocks.push(block_config.init(device));
        }

        let linear_out = nn::LinearConfig::new(spec.embed_dim, spec.out_dim).init(device);

        Self {
            projects,
            blocks,
            linear_out,
        }
    }

    pub fn forward(&self, hidden: Tensor<B, 3>, pos: Option<&Tensor<B, 3>>) -> Tensor<B, 3> {
        let mut hidden = if let Some(projects) = &self.projects {
            projects.forward(hidden)
        } else {
            hidden
        };

        for block in &self.blocks {
            hidden = block.forward(hidden, pos, None);
        }

        self.linear_out.forward(hidden)
    }
}

#[derive(Config, Debug)]
pub struct LinearPts3dConfig {
    pub patch_size: usize,
    pub dec_embed_dim: usize,
    #[config(default = 3)]
    pub output_dim: usize,
    #[config(default = 1)]
    pub downsample_ratio: usize,
    #[config(default = "None")]
    pub points_per_axis: Option<usize>,
}

#[derive(Module, Debug)]
pub struct LinearPts3d<B: Backend> {
    pub patch_size: usize,
    pub downsample_ratio: usize,
    pub points_per_axis: usize,
    pub proj: nn::Linear<B>,
}

impl<B: Backend> LinearPts3d<B> {
    pub fn new(device: &B::Device, config: &LinearPts3dConfig) -> Self {
        let points_per_axis = config
            .points_per_axis
            .unwrap_or(config.patch_size / config.downsample_ratio);
        let points_per_token = points_per_axis * points_per_axis;

        let proj =
            nn::LinearConfig::new(config.dec_embed_dim, config.output_dim * points_per_token)
                .init(device);

        Self {
            patch_size: config.patch_size,
            downsample_ratio: config.downsample_ratio,
            points_per_axis,
            proj,
        }
    }

    pub fn forward(&self, tokens: Tensor<B, 3>, image_hw: (usize, usize)) -> Tensor<B, 4> {
        let (h, w) = image_hw;
        let [batch, _, _] = tokens.shape().dims::<3>();

        let feat = self.proj.forward(tokens);
        let h_patches = h / self.patch_size;
        let w_patches = w / self.patch_size;

        let feat = feat.reshape([batch as i32, h_patches as i32, w_patches as i32, -1]);
        let feat = feat.permute([0, 3, 1, 2]);

        let feat = pixel_shuffle(feat, self.points_per_axis);
        feat.permute([0, 2, 3, 1])
    }
}

#[derive(Config, Debug)]
pub struct MlpHeadConfig {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub out_dim: usize,
}

#[derive(Module, Debug)]
pub struct MlpHead<B: Backend> {
    pub fc1: nn::Linear<B>,
    pub fc2: nn::Linear<B>,
}

impl<B: Backend> MlpHead<B> {
    pub fn new(device: &B::Device, cfg: &MlpHeadConfig) -> Self {
        Self {
            fc1: nn::LinearConfig::new(cfg.in_dim, cfg.hidden_dim).init(device),
            fc2: nn::LinearConfig::new(cfg.hidden_dim, cfg.out_dim).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = gelu(self.fc1.forward(x));
        self.fc2.forward(x)
    }

    pub fn forward_relu(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = relu(self.fc1.forward(x));
        self.fc2.forward(x)
    }
}
