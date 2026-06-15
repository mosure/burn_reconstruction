#![allow(clippy::too_many_arguments)]

#[cfg(feature = "io")]
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf as StdPathBuf},
    time::Duration,
};

use burn::{
    module::{Ignored, Module, Param},
    nn::Initializer,
    prelude::*,
    tensor::{
        activation::{sigmoid, silu, softmax},
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
        Bytes,
    },
};
use burn_dino::layers::{
    attention::{Attention, AttentionConfig},
    layer_norm::{LayerNorm, LayerNormConfig},
    mlp::{Mlp, MlpConfig},
    patch_embed::{PatchEmbed, PatchEmbedConfig},
    rope::RopeConfig,
};
use burn_yono::model::gaussian::{build_covariance_flat, FlatGaussians};

#[cfg(feature = "import")]
use burn_store::{ApplyResult, BurnpackStore, ModuleSnapshot, SafetensorsStore};

pub const MODEL_ID: &str = "zipsplat";
pub const MODEL_DISPLAY_NAME: &str = "ZipSplat";
pub const DEFAULT_CHECKPOINT_PATH: &str = "assets/models/zipsplat.bpk";
pub const DEFAULT_REMOTE_ROOT: &str = "zipsplat";
pub const DEFAULT_IMAGE_SIZE: usize = 252;
pub const DEFAULT_POS_EMBED_IMAGE_SIZE: usize = 518;
pub const IMAGE_SIZE_MULTIPLE: usize = 14;
pub const MIN_VIEWS: usize = 1;
pub const MAX_VIEWS: usize = 24;
pub const MIN_COMPRESSION_R: usize = 1;
pub const MAX_COMPRESSION_R: usize = 16;

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum ZipSplatWeightFormat {
    Safetensors,
    #[default]
    Burnpack,
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum ZipSplatWeightPrecision {
    F32,
    #[default]
    F16,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ZipSplatWeights {
    pub checkpoint: StdPathBuf,
    pub format: ZipSplatWeightFormat,
    pub precision: ZipSplatWeightPrecision,
}

impl ZipSplatWeights {
    pub fn new(checkpoint: impl Into<StdPathBuf>) -> Self {
        Self::burnpack(checkpoint)
    }

    pub fn burnpack(checkpoint: impl Into<StdPathBuf>) -> Self {
        Self {
            checkpoint: checkpoint.into(),
            format: ZipSplatWeightFormat::Burnpack,
            precision: ZipSplatWeightPrecision::default(),
        }
    }

    pub fn safetensors(checkpoint: impl Into<StdPathBuf>) -> Self {
        Self {
            checkpoint: checkpoint.into(),
            format: ZipSplatWeightFormat::Safetensors,
            precision: ZipSplatWeightPrecision::default(),
        }
    }

    pub fn with_format(mut self, format: ZipSplatWeightFormat) -> Self {
        self.format = format;
        self
    }

    pub fn with_precision(mut self, precision: ZipSplatWeightPrecision) -> Self {
        self.precision = precision;
        self
    }
}

impl Default for ZipSplatWeights {
    fn default() -> Self {
        Self::burnpack(DEFAULT_CHECKPOINT_PATH)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ZipSplatCompression {
    r: usize,
}

impl ZipSplatCompression {
    pub const FULL: Self = Self { r: 1 };
    pub const BALANCED: Self = Self { r: 2 };
    pub const COMPACT: Self = Self { r: 4 };
    pub const PREVIEW: Self = Self { r: 8 };

    pub fn new(r: usize) -> Self {
        Self {
            r: r.clamp(MIN_COMPRESSION_R, MAX_COMPRESSION_R),
        }
    }

    pub fn get(self) -> usize {
        self.r
    }

    pub fn compression_ratio(self) -> f32 {
        1.0 / self.r as f32
    }

    pub fn retained_tokens(self, total_tokens: usize) -> usize {
        retained_token_count(total_tokens, self.compression_ratio())
    }
}

impl Default for ZipSplatCompression {
    fn default() -> Self {
        Self::BALANCED
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ZipSplatQualityPreset {
    pub name: &'static str,
    pub compression: ZipSplatCompression,
}

pub const QUALITY_PRESETS: [ZipSplatQualityPreset; 4] = [
    ZipSplatQualityPreset {
        name: "full",
        compression: ZipSplatCompression::FULL,
    },
    ZipSplatQualityPreset {
        name: "balanced",
        compression: ZipSplatCompression::BALANCED,
    },
    ZipSplatQualityPreset {
        name: "compact",
        compression: ZipSplatCompression::COMPACT,
    },
    ZipSplatQualityPreset {
        name: "preview",
        compression: ZipSplatCompression::PREVIEW,
    },
];

#[derive(Debug, Clone)]
pub struct ZipSplatConfig {
    pub image_size: usize,
    pub pos_embed_image_size: usize,
    pub patch_size: usize,
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub out_layers: Vec<usize>,
    pub alt_start: usize,
    pub qk_norm_start: usize,
    pub rope_start: usize,
    pub gaussians_per_token: usize,
    pub sh_degree: usize,
    pub color_skip_dim: usize,
    pub color_mlp_ratio: f32,
    pub kmeans_iterations: usize,
    pub kmeans_chunk_size: usize,
}

impl Default for ZipSplatConfig {
    fn default() -> Self {
        Self::da3_giant()
    }
}

impl ZipSplatConfig {
    pub fn da3_giant() -> Self {
        Self {
            image_size: DEFAULT_IMAGE_SIZE,
            pos_embed_image_size: DEFAULT_POS_EMBED_IMAGE_SIZE,
            patch_size: 14,
            embed_dim: 1536,
            depth: 40,
            num_heads: 24,
            out_layers: vec![19, 29, 39],
            alt_start: 13,
            qk_norm_start: 13,
            rope_start: 13,
            gaussians_per_token: 32,
            sh_degree: 1,
            color_skip_dim: 128,
            color_mlp_ratio: 1.0,
            kmeans_iterations: 5,
            kmeans_chunk_size: 2048,
        }
    }

    pub fn tiny_for_tests() -> Self {
        Self {
            image_size: 28,
            pos_embed_image_size: 28,
            patch_size: 14,
            embed_dim: 64,
            depth: 2,
            num_heads: 4,
            out_layers: vec![0, 1],
            alt_start: 1,
            qk_norm_start: 1,
            rope_start: 1,
            gaussians_per_token: 2,
            sh_degree: 1,
            color_skip_dim: 16,
            color_mlp_ratio: 1.0,
            kmeans_iterations: 2,
            kmeans_chunk_size: 256,
        }
    }

    fn d_sh(&self) -> usize {
        (self.sh_degree + 1).pow(2)
    }

    fn params_per_gaussian(&self) -> usize {
        3 + 3 + 4 + 1 + self.d_sh() * 3
    }

    /// Number of patch tokens emitted by one square input view.
    pub fn patch_tokens_per_view(&self) -> usize {
        let patches_per_axis = (self.image_size / self.patch_size).max(1);
        patches_per_axis * patches_per_axis
    }

    /// Number of scene tokens before ZipSplat compression for `view_count`.
    pub fn total_tokens_for_views(&self, view_count: usize) -> usize {
        view_count.max(1) * self.patch_tokens_per_view()
    }

    /// Expected raw Gaussian count for an input view count and compression.
    ///
    /// Mirrors upstream ZipSplat: retain `int(V * T * compression)` scene
    /// tokens, then decode `gaussians_per_token` Gaussians from each token.
    pub fn estimated_gaussian_count(
        &self,
        view_count: usize,
        compression: ZipSplatCompression,
    ) -> usize {
        let retained = compression.retained_tokens(self.total_tokens_for_views(view_count));
        retained * self.gaussians_per_token
    }
}

#[derive(Debug, Clone, Default)]
pub struct ZipSplatForwardTimings {
    pub image_load: Duration,
    pub backbone: Duration,
    pub head: Duration,
    pub total: Duration,
}

#[derive(Debug, Clone)]
pub struct ZipSplatApplySummary {
    pub applied: usize,
    pub missing: Vec<String>,
    pub unused: Vec<String>,
    pub skipped: Vec<String>,
}

impl ZipSplatApplySummary {
    pub fn synthetic_success(applied: usize) -> Self {
        Self {
            applied,
            missing: Vec::new(),
            unused: Vec::new(),
            skipped: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ZipSplatLoadReport {
    pub model: ZipSplatApplySummary,
}

#[cfg(not(target_arch = "wasm32"))]
struct ForwardTimer(Instant);

#[cfg(not(target_arch = "wasm32"))]
impl ForwardTimer {
    fn now() -> Self {
        Self(Instant::now())
    }

    fn elapsed(&self) -> Duration {
        self.0.elapsed()
    }
}

#[cfg(target_arch = "wasm32")]
struct ForwardTimer;

#[cfg(target_arch = "wasm32")]
impl ForwardTimer {
    fn now() -> Self {
        Self
    }

    fn elapsed(&self) -> Duration {
        Duration::ZERO
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ZipSplatPipelineError {
    #[error("image-size must be divisible by 14, got {0}")]
    InvalidImageSize(usize),
    #[error("expected between {MIN_VIEWS} and {MAX_VIEWS} input images, got {0}")]
    InvalidViewCount(usize),
    #[error("ZipSplat checkpoint not found at {0}")]
    MissingWeights(String),
    #[cfg(feature = "import")]
    #[error("failed to load ZipSplat weights: {0}")]
    Import(String),
    #[cfg(feature = "io")]
    #[error("failed to load image `{path}`: {source}")]
    ImageLoad {
        path: String,
        source: image::ImageError,
    },
    #[cfg(feature = "io")]
    #[error("failed to decode in-memory image `{name}`: {source}")]
    ImageDecode {
        name: String,
        source: image::ImageError,
    },
    #[error("failed to read debug tensor from backend: {0}")]
    TensorData(String),
}

const DA3_LAYER_NORM_EPS: f64 = 1e-6;
const ZIP_SPLAT_LAYER_NORM_EPS: f64 = 1e-5;
const ZIP_SPLAT_LAYER_SCALE_INIT: f32 = 0.01;

fn layer_norm_config(dim: usize, epsilon: f64) -> LayerNormConfig {
    LayerNormConfig { dim, epsilon }
}

#[derive(Module, Debug)]
pub struct ZipSplatLayerScale<B: Backend> {
    pub gamma: Param<Tensor<B, 1>>,
}

impl<B: Backend> ZipSplatLayerScale<B> {
    pub fn new(device: &B::Device, dim: usize) -> Self {
        Self {
            gamma: Param::from_tensor(Tensor::full([dim], ZIP_SPLAT_LAYER_SCALE_INIT, device)),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        x * self.gamma.val().unsqueeze()
    }
}

#[derive(Module, Debug)]
pub struct SwiGluFfn<B: Backend> {
    pub w12: nn::Linear<B>,
    pub w3: nn::Linear<B>,
}

impl<B: Backend> SwiGluFfn<B> {
    pub fn new(device: &B::Device, dim: usize, mlp_ratio: f32) -> Self {
        let hidden = swiglu_hidden_dim(dim, mlp_ratio);
        Self {
            w12: nn::LinearConfig::new(dim, hidden * 2)
                .with_bias(true)
                .init(device),
            w3: nn::LinearConfig::new(hidden, dim)
                .with_bias(true)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let fused = self.w12.forward(x);
        let [batch, tokens, hidden_twice] = fused.shape().dims::<3>();
        let hidden = hidden_twice / 2;
        let lhs = fused
            .clone()
            .slice([0..batch as i32, 0..tokens as i32, 0..hidden as i32]);
        let rhs = fused.slice([
            0..batch as i32,
            0..tokens as i32,
            hidden as i32..hidden_twice as i32,
        ]);
        self.w3.forward(silu(lhs) * rhs)
    }
}

#[derive(Module, Debug)]
pub struct Da3SelfAttentionBlock<B: Backend> {
    norm1: LayerNorm<B>,
    attn: Attention<B>,
    ls1: ZipSplatLayerScale<B>,
    norm2: LayerNorm<B>,
    mlp: SwiGluFfn<B>,
    ls2: ZipSplatLayerScale<B>,
}

impl<B: Backend> Da3SelfAttentionBlock<B> {
    pub fn new(
        device: &B::Device,
        dim: usize,
        num_heads: usize,
        qk_norm: bool,
        rope: bool,
    ) -> Self {
        Self {
            norm1: LayerNormConfig::new(dim).init(device),
            attn: AttentionConfig {
                dim,
                num_heads,
                qkv_bias: true,
                proj_bias: true,
                qk_norm,
                rope: rope.then_some(RopeConfig::default()),
                quiet_softmax: false,
                ..Default::default()
            }
            .init(device),
            ls1: ZipSplatLayerScale::new(device, dim),
            norm2: LayerNormConfig::new(dim).init(device),
            mlp: SwiGluFfn::new(device, dim, 4.0),
            ls2: ZipSplatLayerScale::new(device, dim),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, pos: Option<&Tensor<B, 3>>) -> Tensor<B, 3> {
        let attn = self.attn.forward(self.norm1.forward(x.clone()), pos, None);
        let x = x + self.ls1.forward(attn);
        let mlp = self.mlp.forward(self.norm2.forward(x.clone()));
        x + self.ls2.forward(mlp)
    }
}

#[derive(Module, Debug)]
pub struct SceneSelfAttentionBlock<B: Backend> {
    norm1: LayerNorm<B>,
    attn: ZipSplatSelfAttention<B>,
    ls1: ZipSplatLayerScale<B>,
    norm2: LayerNorm<B>,
    mlp: Mlp<B, 3>,
    ls2: ZipSplatLayerScale<B>,
}

impl<B: Backend> SceneSelfAttentionBlock<B> {
    pub fn new(device: &B::Device, dim: usize, num_heads: usize, mlp_ratio: f32) -> Self {
        Self {
            norm1: layer_norm_config(dim, ZIP_SPLAT_LAYER_NORM_EPS).init(device),
            attn: ZipSplatSelfAttention::new(device, dim, num_heads),
            ls1: ZipSplatLayerScale::new(device, dim),
            norm2: layer_norm_config(dim, ZIP_SPLAT_LAYER_NORM_EPS).init(device),
            mlp: MlpConfig::new(dim)
                .with_hidden_features(Some((dim as f32 * mlp_ratio) as usize))
                .with_out_features(Some(dim))
                .with_bias(Some(true))
                .init(device),
            ls2: ZipSplatLayerScale::new(device, dim),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let attn = self.attn.forward(self.norm1.forward(x.clone()));
        let x = x + self.ls1.forward(attn);
        let mlp = self.mlp.forward(self.norm2.forward(x.clone()));
        x + self.ls2.forward(mlp)
    }
}

#[derive(Module, Debug)]
pub struct ZipSplatSelfAttention<B: Backend> {
    pub qkv: nn::Linear<B>,
    q_norm: LayerNorm<B>,
    k_norm: LayerNorm<B>,
    pub proj: nn::Linear<B>,
    num_heads: Ignored<usize>,
}

impl<B: Backend> ZipSplatSelfAttention<B> {
    pub fn new(device: &B::Device, dim: usize, num_heads: usize) -> Self {
        let heads = num_heads.max(1);
        Self {
            qkv: nn::LinearConfig::new(dim, dim * 3)
                .with_bias(true)
                .init(device),
            q_norm: layer_norm_config(dim / heads, ZIP_SPLAT_LAYER_NORM_EPS).init(device),
            k_norm: layer_norm_config(dim / heads, ZIP_SPLAT_LAYER_NORM_EPS).init(device),
            proj: nn::LinearConfig::new(dim, dim).with_bias(true).init(device),
            num_heads: Ignored(heads),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, tokens, dim] = x.shape().dims::<3>();
        let heads = self.num_heads.0;
        let head_dim = dim / heads;
        let scale = (head_dim as f32).powf(-0.5);
        let qkv = self
            .qkv
            .forward(x)
            .reshape([
                batch as i32,
                tokens as i32,
                3,
                heads as i32,
                head_dim as i32,
            ])
            .permute([2, 0, 3, 1, 4]);
        let q_raw: Tensor<B, 4> = qkv.clone().slice_dim(0, 0..1).squeeze_dim(0);
        let k_raw: Tensor<B, 4> = qkv.clone().slice_dim(0, 1..2).squeeze_dim(0);
        let v: Tensor<B, 4> = qkv.slice_dim(0, 2..3).squeeze_dim(0);
        let q = self.q_norm.forward(q_raw);
        let k = self.k_norm.forward(k_raw);
        let attn = softmax(q.matmul(k.swap_dims(2, 3)).mul_scalar(scale), 3);
        let out = attn
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch as i32, tokens as i32, dim as i32]);
        self.proj.forward(out)
    }
}

#[derive(Module, Debug)]
pub struct Da3Backbone<B: Backend> {
    pub patch_embed: PatchEmbed<B>,
    pub cls_token: Param<Tensor<B, 3>>,
    pub camera_token: Param<Tensor<B, 3>>,
    pub pos_embed: Param<Tensor<B, 3>>,
    pub blocks: Vec<Da3SelfAttentionBlock<B>>,
    pub norm: LayerNorm<B>,
    out_layers: Ignored<Vec<usize>>,
    embed_dim: Ignored<usize>,
    patch_size: Ignored<usize>,
    alt_start: Ignored<usize>,
    rope_start: Ignored<usize>,
}

impl<B: Backend> Da3Backbone<B> {
    pub fn new(device: &B::Device, cfg: &ZipSplatConfig) -> Self {
        let pos_embed_grid = cfg.pos_embed_image_size / cfg.patch_size;
        let patch_count = pos_embed_grid * pos_embed_grid;
        Self {
            patch_embed: PatchEmbedConfig {
                image_size: cfg.image_size,
                patch_size: cfg.patch_size,
                input_channels: 3,
                embedding_dimension: cfg.embed_dim,
            }
            .init(device),
            cls_token: Initializer::Zeros.init([1, 1, cfg.embed_dim], device),
            camera_token: Initializer::Normal {
                mean: 0.0,
                std: 1.0,
            }
            .init([1, 2, cfg.embed_dim], device),
            pos_embed: Initializer::Zeros.init([1, patch_count + 1, cfg.embed_dim], device),
            blocks: (0..cfg.depth)
                .map(|idx| {
                    Da3SelfAttentionBlock::new(
                        device,
                        cfg.embed_dim,
                        cfg.num_heads,
                        idx >= cfg.qk_norm_start,
                        idx >= cfg.rope_start,
                    )
                })
                .collect(),
            norm: layer_norm_config(cfg.embed_dim, DA3_LAYER_NORM_EPS).init(device),
            out_layers: Ignored(cfg.out_layers.clone()),
            embed_dim: Ignored(cfg.embed_dim),
            patch_size: Ignored(cfg.patch_size),
            alt_start: Ignored(cfg.alt_start),
            rope_start: Ignored(cfg.rope_start),
        }
    }

    pub fn forward(&self, images: Tensor<B, 5>) -> Vec<Tensor<B, 4>> {
        let [batch, views, _channels, height, width] = images.shape().dims::<5>();
        let device = images.device();
        let mut x = self.prepare_tokens(images);
        let (local_pos, global_pos) =
            self.prepare_rope_positions(batch, views, height, width, &device);
        let mut outputs = Vec::new();
        let mut local_x: Option<Tensor<B, 4>> = None;

        for (idx, block) in self.blocks.iter().enumerate() {
            if idx == self.alt_start.0 {
                x = replace_camera_token(x, self.default_camera_token(batch, views));
            }

            let is_global = idx >= self.alt_start.0 && idx % 2 == 1;
            let rope_active = idx >= self.rope_start.0;
            let pos = if rope_active {
                if is_global {
                    Some(global_pos.clone())
                } else {
                    Some(local_pos.clone())
                }
            } else {
                None
            };
            x = self.process_attention(x, block, is_global, pos.as_ref());
            if !is_global {
                local_x = Some(x.clone());
            }
            if self.out_layers.0.contains(&idx) {
                let local = local_x.clone().unwrap_or_else(|| x.clone());
                let full = Tensor::cat(vec![local, x.clone()], 3);
                outputs.push(strip_camera_token(self.apply_global_norm(full)));
            }
        }

        outputs.reverse();
        outputs
    }

    fn prepare_tokens(&self, images: Tensor<B, 5>) -> Tensor<B, 4> {
        let [batch, views, channels, height, width] = images.shape().dims::<5>();
        let flat = images.reshape([
            (batch * views) as i32,
            channels as i32,
            height as i32,
            width as i32,
        ]);
        let patches = self.patch_embed.forward(flat);
        let token_count = patches.shape().dims::<3>()[1];
        let cls = self.cls_token.val().repeat_dim(0, batch * views).reshape([
            (batch * views) as i32,
            1,
            self.embed_dim.0 as i32,
        ]);
        let tokens = Tensor::cat(vec![cls, patches], 1);
        let pos = self
            .interpolate_pos_encoding(token_count, height, width)
            .repeat_dim(0, batch * views);
        (tokens + pos).reshape([
            batch as i32,
            views as i32,
            (token_count + 1) as i32,
            self.embed_dim.0 as i32,
        ])
    }

    fn interpolate_pos_encoding(
        &self,
        token_count: usize,
        height: usize,
        width: usize,
    ) -> Tensor<B, 3> {
        let pos_embed = self.pos_embed.val();
        let [_batch, total_tokens, channels] = pos_embed.shape().dims::<3>();
        let source_tokens = total_tokens - 1;

        if source_tokens == token_count && height == width {
            return pos_embed;
        }

        let source_grid = source_tokens.isqrt();
        assert_eq!(
            source_tokens,
            source_grid * source_grid,
            "ZipSplat DA3 positional embedding must have a square patch grid",
        );

        let target_h = (height / self.patch_size.0).max(1);
        let target_w = (width / self.patch_size.0).max(1);

        let class_pos = pos_embed.clone().slice([0..1, 0..1, 0..channels as i32]);
        let patch_pos = pos_embed.slice([0..1, 1..total_tokens as i32, 0..channels as i32]);

        if source_grid == target_h && source_grid == target_w {
            return Tensor::cat(vec![class_pos, patch_pos], 1);
        }

        let patch_pos = patch_pos.swap_dims(1, 2).reshape([
            1,
            channels as i32,
            source_grid as i32,
            source_grid as i32,
        ]);
        let patch_pos = interpolate(
            patch_pos,
            [target_h, target_w],
            InterpolateOptions::new(InterpolateMode::Bicubic),
        )
        .reshape([1, channels as i32, (target_h * target_w) as i32])
        .swap_dims(1, 2);

        Tensor::cat(vec![class_pos, patch_pos], 1)
    }

    fn prepare_rope_positions(
        &self,
        batch: usize,
        views: usize,
        height: usize,
        width: usize,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let patch_h = height / self.patch_size.0;
        let patch_w = width / self.patch_size.0;
        let per_view = patch_h * patch_w + 1;
        let ys = Tensor::<B, 1, Int>::arange(1..(patch_h + 1) as i64, device)
            .float()
            .reshape([patch_h as i32, 1])
            .repeat_dim(1, patch_w);
        let xs = Tensor::<B, 1, Int>::arange(1..(patch_w + 1) as i64, device)
            .float()
            .reshape([1, patch_w as i32])
            .repeat_dim(0, patch_h);
        let local_patches = Tensor::cat(
            vec![
                ys.reshape([(patch_h * patch_w) as i32, 1]),
                xs.reshape([(patch_h * patch_w) as i32, 1]),
            ],
            1,
        );
        let zero_token = Tensor::<B, 2>::zeros([1, 2], device);
        let local_per_view = Tensor::cat(vec![zero_token.clone(), local_patches], 0);
        let local = local_per_view
            .unsqueeze_dim::<3>(0)
            .repeat_dim(0, batch * views);

        let global_patches = Tensor::<B, 2>::ones([(patch_h * patch_w) as i32, 2], device);
        let global_per_view = Tensor::cat(vec![zero_token, global_patches], 0);
        let global = global_per_view
            .unsqueeze_dim::<3>(0)
            .repeat_dim(0, batch * views)
            .reshape([batch as i32, (views * per_view) as i32, 2]);
        (local, global)
    }

    fn process_attention(
        &self,
        x: Tensor<B, 4>,
        block: &Da3SelfAttentionBlock<B>,
        is_global: bool,
        pos: Option<&Tensor<B, 3>>,
    ) -> Tensor<B, 4> {
        let [batch, views, tokens, dim] = x.shape().dims::<4>();
        if is_global {
            let flat = x.reshape([batch as i32, (views * tokens) as i32, dim as i32]);
            block.forward(flat, pos).reshape([
                batch as i32,
                views as i32,
                tokens as i32,
                dim as i32,
            ])
        } else {
            let flat = x.reshape([(batch * views) as i32, tokens as i32, dim as i32]);
            block.forward(flat, pos).reshape([
                batch as i32,
                views as i32,
                tokens as i32,
                dim as i32,
            ])
        }
    }

    fn default_camera_token(&self, batch: usize, views: usize) -> Tensor<B, 3> {
        let dim = self.embed_dim.0;
        let ref_token = self
            .camera_token
            .val()
            .slice([0..1, 0..1, 0..dim as i32])
            .repeat_dim(0, batch);
        if views == 1 {
            return ref_token;
        }
        let src_token = self
            .camera_token
            .val()
            .slice([0..1, 1..2, 0..dim as i32])
            .repeat_dim(0, batch)
            .repeat_dim(1, views - 1);
        Tensor::cat(vec![ref_token, src_token], 1)
    }

    fn apply_global_norm(&self, tokens: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, views, token_count, dim2] = tokens.shape().dims::<4>();
        let dim = self.embed_dim.0;
        let local = tokens.clone().slice([
            0..batch as i32,
            0..views as i32,
            0..token_count as i32,
            0..dim as i32,
        ]);
        let global = tokens.slice([
            0..batch as i32,
            0..views as i32,
            0..token_count as i32,
            dim as i32..dim2 as i32,
        ]);
        Tensor::cat(vec![local, self.norm.forward(global)], 3)
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim.0
    }
}

#[derive(Module, Debug)]
pub struct CrossAttention<B: Backend> {
    pub q_proj: nn::Linear<B>,
    pub kv_proj: nn::Linear<B>,
    q_norm: LayerNorm<B>,
    k_norm: LayerNorm<B>,
    pub proj: nn::Linear<B>,
    num_heads: Ignored<usize>,
}

impl<B: Backend> CrossAttention<B> {
    pub fn new(device: &B::Device, dim: usize, num_heads: usize) -> Self {
        let heads = num_heads.max(1);
        Self {
            q_proj: nn::LinearConfig::new(dim, dim).with_bias(true).init(device),
            kv_proj: nn::LinearConfig::new(dim, dim * 2)
                .with_bias(true)
                .init(device),
            q_norm: layer_norm_config(dim / heads, ZIP_SPLAT_LAYER_NORM_EPS).init(device),
            k_norm: layer_norm_config(dim / heads, ZIP_SPLAT_LAYER_NORM_EPS).init(device),
            proj: nn::LinearConfig::new(dim, dim).with_bias(true).init(device),
            num_heads: Ignored(heads),
        }
    }

    pub fn forward(&self, queries: Tensor<B, 3>, context: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, query_tokens, dim] = queries.shape().dims::<3>();
        let context_tokens = context.shape().dims::<3>()[1];
        let heads = self.num_heads.0;
        let head_dim = dim / heads;
        let scale = (head_dim as f32).powf(-0.5);
        let q = self
            .q_proj
            .forward(queries)
            .reshape([
                batch as i32,
                query_tokens as i32,
                heads as i32,
                head_dim as i32,
            ])
            .swap_dims(1, 2);
        let q = self.q_norm.forward(q);
        let kv = self
            .kv_proj
            .forward(context)
            .reshape([
                batch as i32,
                context_tokens as i32,
                2,
                heads as i32,
                head_dim as i32,
            ])
            .permute([2, 0, 3, 1, 4]);
        let k = self
            .k_norm
            .forward(kv.clone().slice_dim(0, 0..1).squeeze_dim(0));
        let v = kv.slice_dim(0, 1..2).squeeze_dim(0);
        let attn = softmax(q.matmul(k.swap_dims(2, 3)).mul_scalar(scale), 3);
        let out =
            attn.matmul(v)
                .swap_dims(1, 2)
                .reshape([batch as i32, query_tokens as i32, dim as i32]);
        self.proj.forward(out)
    }
}

#[derive(Module, Debug)]
pub struct CrossAttentionBlock<B: Backend> {
    norm_q: LayerNorm<B>,
    norm_kv: LayerNorm<B>,
    attn: CrossAttention<B>,
    ls1: ZipSplatLayerScale<B>,
    norm2: LayerNorm<B>,
    mlp: Mlp<B, 3>,
    ls2: ZipSplatLayerScale<B>,
}

impl<B: Backend> CrossAttentionBlock<B> {
    pub fn new(device: &B::Device, dim: usize, num_heads: usize, mlp_ratio: f32) -> Self {
        Self {
            norm_q: layer_norm_config(dim, ZIP_SPLAT_LAYER_NORM_EPS).init(device),
            norm_kv: layer_norm_config(dim, ZIP_SPLAT_LAYER_NORM_EPS).init(device),
            attn: CrossAttention::new(device, dim, num_heads),
            ls1: ZipSplatLayerScale::new(device, dim),
            norm2: layer_norm_config(dim, ZIP_SPLAT_LAYER_NORM_EPS).init(device),
            mlp: MlpConfig::new(dim)
                .with_hidden_features(Some((dim as f32 * mlp_ratio) as usize))
                .with_out_features(Some(dim))
                .with_bias(Some(true))
                .init(device),
            ls2: ZipSplatLayerScale::new(device, dim),
        }
    }

    pub fn forward_delta(&self, query: Tensor<B, 3>, keys: Tensor<B, 3>) -> Tensor<B, 3> {
        let attn = self.attn.forward(
            self.norm_q.forward(query.clone()),
            self.norm_kv.forward(keys),
        );
        let attn_delta = self.ls1.forward(attn);
        let post_attn = query + attn_delta.clone();
        let mlp_delta = self
            .ls2
            .forward(self.mlp.forward(self.norm2.forward(post_attn)));
        attn_delta + mlp_delta
    }

    pub fn forward(&self, query: Tensor<B, 3>, keys: Tensor<B, 3>) -> Tensor<B, 3> {
        let attn = self.attn.forward(
            self.norm_q.forward(query.clone()),
            self.norm_kv.forward(keys),
        );
        let attn_delta = self.ls1.forward(attn);
        let post_attn = query + attn_delta;
        let mlp_delta = self
            .ls2
            .forward(self.mlp.forward(self.norm2.forward(post_attn.clone())));
        post_attn + mlp_delta
    }
}

#[derive(Module, Debug)]
pub struct GaussianHead<B: Backend> {
    mlp: Mlp<B, 3>,
    linear: nn::Linear<B>,
    gaussians_per_token: Ignored<usize>,
    d_sh: Ignored<usize>,
    params_per_gaussian: Ignored<usize>,
}

impl<B: Backend> GaussianHead<B> {
    pub fn new(device: &B::Device, head_dim: usize, cfg: &ZipSplatConfig) -> Self {
        Self {
            mlp: MlpConfig::new(head_dim)
                .with_hidden_features(Some(2 * head_dim))
                .with_out_features(Some(head_dim))
                .with_bias(Some(true))
                .init(device),
            linear: nn::LinearConfig::new(
                head_dim,
                cfg.gaussians_per_token * cfg.params_per_gaussian(),
            )
            .with_bias(true)
            .init(device),
            gaussians_per_token: Ignored(cfg.gaussians_per_token),
            d_sh: Ignored(cfg.d_sh()),
            params_per_gaussian: Ignored(cfg.params_per_gaussian()),
        }
    }

    pub fn forward(&self, tokens: Tensor<B, 3>) -> FlatGaussians<B> {
        let [batch, scene_tokens, _] = tokens.shape().dims::<3>();
        let params = self.linear.forward(self.mlp.forward(tokens));
        let count = scene_tokens * self.gaussians_per_token.0;
        let params = params.reshape([
            batch as i32,
            count as i32,
            self.params_per_gaussian.0 as i32,
        ]);

        let means_raw = params
            .clone()
            .slice([0..batch as i32, 0..count as i32, 0..3])
            .clamp_min(-5.0)
            .clamp_max(5.0);
        let means = means_raw.clone().sign() * (means_raw.abs().exp() - 1.0);

        let scales = softplus(
            params
                .clone()
                .slice([0..batch as i32, 0..count as i32, 3..6])
                - 4.0,
        )
        .clamp_min(1e-6)
        .clamp_max(15.0);

        let quats_wxyz = params
            .clone()
            .slice([0..batch as i32, 0..count as i32, 6..10]);
        let quat_norm = quats_wxyz
            .clone()
            .powi_scalar(2)
            .sum_dim(2)
            .sqrt()
            .clamp_min(1e-8);
        let quats_wxyz = quats_wxyz / quat_norm;
        let quats_xyzw = quaternion_wxyz_to_xyzw(quats_wxyz);

        let opacities = sigmoid(
            params
                .clone()
                .slice([0..batch as i32, 0..count as i32, 10..11]),
        )
        .squeeze_dim(2);

        let harmonics = params
            .slice([
                0..batch as i32,
                0..count as i32,
                11..self.params_per_gaussian.0 as i32,
            ])
            .reshape([batch as i32, count as i32, self.d_sh.0 as i32, 3])
            .swap_dims(2, 3);

        let covariances = build_covariance_flat(
            scales.clone().reshape([(batch * count) as i32, 3]),
            quats_xyzw.clone().reshape([(batch * count) as i32, 4]),
        )
        .reshape([batch as i32, count as i32, 3, 3]);

        FlatGaussians {
            means,
            covariances,
            harmonics,
            opacities,
            rotations: quats_xyzw,
            scales,
        }
    }
}

#[derive(Module, Debug)]
pub struct ZipSplatModel<B: Backend> {
    backbone: Da3Backbone<B>,
    pre_norm_local: Vec<LayerNorm<B>>,
    pre_norm_global: Vec<LayerNorm<B>>,
    downscale: Vec<nn::Linear<B>>,
    cross_attention: Vec<CrossAttentionBlock<B>>,
    self_attention: Vec<SceneSelfAttentionBlock<B>>,
    color_embed: PatchEmbed<B>,
    color_cross_attention: CrossAttentionBlock<B>,
    gaussian_head: GaussianHead<B>,
    cfg: Ignored<ZipSplatConfig>,
}

impl<B: Backend> ZipSplatModel<B> {
    pub fn new(device: &B::Device, cfg: ZipSplatConfig) -> Self {
        let layers = cfg.out_layers.len();
        let embed_dim = cfg.embed_dim;
        let heads = (embed_dim / 64).max(1);

        Self {
            backbone: Da3Backbone::new(device, &cfg),
            pre_norm_local: (0..layers)
                .map(|_| layer_norm_config(embed_dim, ZIP_SPLAT_LAYER_NORM_EPS).init(device))
                .collect(),
            pre_norm_global: (0..layers)
                .map(|_| layer_norm_config(embed_dim, ZIP_SPLAT_LAYER_NORM_EPS).init(device))
                .collect(),
            downscale: (0..layers)
                .map(|_| {
                    nn::LinearConfig::new(2 * embed_dim, embed_dim)
                        .with_bias(true)
                        .init(device)
                })
                .collect(),
            cross_attention: (0..layers)
                .map(|_| CrossAttentionBlock::new(device, embed_dim, heads, 4.0))
                .collect(),
            self_attention: (0..layers)
                .map(|_| SceneSelfAttentionBlock::new(device, embed_dim, heads, 4.0))
                .collect(),
            color_embed: PatchEmbedConfig {
                image_size: cfg.image_size,
                patch_size: cfg.patch_size,
                input_channels: 3,
                embedding_dimension: cfg.color_skip_dim,
            }
            .init(device),
            color_cross_attention: CrossAttentionBlock::new(
                device,
                cfg.color_skip_dim,
                (cfg.color_skip_dim / 64).max(1),
                cfg.color_mlp_ratio,
            ),
            gaussian_head: GaussianHead::new(device, embed_dim + cfg.color_skip_dim, &cfg),
            cfg: Ignored(cfg),
        }
    }

    pub fn forward(
        &self,
        images: Tensor<B, 5>,
        compression: ZipSplatCompression,
    ) -> FlatGaussians<B> {
        let features = self.backbone.forward(images.clone());
        let layer_tokens = self.prepare(features);
        let middle = layer_tokens.len() / 2;
        // Keep token selection on the active tensor backend; a host round trip dominates WGPU inference.
        let indices = kmeans_nearest_indices_tensor(
            &layer_tokens[middle],
            compression,
            self.cfg.0.kmeans_iterations,
            self.cfg.0.kmeans_chunk_size,
        );
        let scene = self.fuse(&layer_tokens, indices.clone());
        let color = self.color(images, indices);
        self.gaussian_head
            .forward(Tensor::cat(vec![scene, color], 2))
    }

    fn prepare(&self, features: Vec<Tensor<B, 4>>) -> Vec<Tensor<B, 4>> {
        let embed_dim = self.backbone.embed_dim();
        features
            .into_iter()
            .enumerate()
            .map(|(idx, raw)| {
                let [batch, views, tokens, _] = raw.shape().dims::<4>();
                let local = raw.clone().slice([
                    0..batch as i32,
                    0..views as i32,
                    0..tokens as i32,
                    0..embed_dim as i32,
                ]);
                let global = raw.slice([
                    0..batch as i32,
                    0..views as i32,
                    0..tokens as i32,
                    embed_dim as i32..(2 * embed_dim) as i32,
                ]);
                let local = self.pre_norm_local[idx].forward(local);
                let global = self.pre_norm_global[idx].forward(global);
                let fused = self.downscale[idx]
                    .forward(Tensor::cat(vec![local.clone(), global.clone()], 3));
                fused + (local + global).div_scalar(2.0)
            })
            .collect()
    }

    fn fuse(&self, layer_tokens: &[Tensor<B, 4>], indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut scene = gather_layer_tokens(layer_tokens[0].clone(), indices.clone());
        for (idx, layer) in layer_tokens.iter().enumerate() {
            let keys = flatten_view_tokens(layer.clone());
            let queries = gather_flat_tokens(keys.clone(), indices.clone());
            scene = scene + self.cross_attention[idx].forward_delta(queries, keys);
            scene = self.self_attention[idx].forward(scene);
        }
        scene
    }

    fn color(&self, images: Tensor<B, 5>, indices: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, views, channels, height, width] = images.shape().dims::<5>();
        let flat = images.reshape([
            (batch * views) as i32,
            channels as i32,
            height as i32,
            width as i32,
        ]);
        let color = self.color_embed.forward(flat);
        let dims = color.shape().dims::<3>();
        let color = color.reshape([batch as i32, (views * dims[1]) as i32, dims[2] as i32]);
        let queries = gather_flat_tokens(color.clone(), indices);
        self.color_cross_attention.forward(queries, color)
    }
}

#[derive(Module, Debug)]
pub struct ZipSplatModelBundle<B: Backend> {
    pub model: ZipSplatModel<B>,
}

impl<B: Backend> ZipSplatModelBundle<B> {
    pub fn new(device: &B::Device, cfg: ZipSplatConfig) -> Self {
        Self {
            model: ZipSplatModel::new(device, cfg),
        }
    }

    #[cfg(feature = "import")]
    pub fn load_from_weights(
        device: &B::Device,
        weights: &ZipSplatWeights,
    ) -> Result<(Self, ZipSplatLoadReport), ZipSplatPipelineError> {
        Self::load_from_weights_with_config(device, weights, ZipSplatConfig::default())
    }

    #[cfg(feature = "import")]
    pub fn load_from_weights_with_config(
        device: &B::Device,
        weights: &ZipSplatWeights,
        cfg: ZipSplatConfig,
    ) -> Result<(Self, ZipSplatLoadReport), ZipSplatPipelineError> {
        let checkpoint = match weights.format {
            ZipSplatWeightFormat::Burnpack => {
                let precision_path =
                    burnpack_path_for_precision(&weights.checkpoint, weights.precision);
                if precision_path.exists() {
                    precision_path
                } else {
                    weights.checkpoint.clone()
                }
            }
            ZipSplatWeightFormat::Safetensors => weights.checkpoint.clone(),
        };

        if !checkpoint.exists() {
            return Err(ZipSplatPipelineError::MissingWeights(
                checkpoint.display().to_string(),
            ));
        }

        let mut bundle = Self::new(device, cfg);
        let apply = match weights.format {
            ZipSplatWeightFormat::Safetensors => {
                let mut store = SafetensorsStore::from_file(&checkpoint);
                bundle
                    .load_from(&mut store)
                    .map_err(|err| ZipSplatPipelineError::Import(format!("{err:?}")))?
            }
            ZipSplatWeightFormat::Burnpack => {
                let mut store = BurnpackStore::from_file(&checkpoint)
                    .auto_extension(false)
                    .validate(true);
                bundle
                    .load_from(&mut store)
                    .map_err(|err| ZipSplatPipelineError::Import(format!("{err:?}")))?
            }
        };

        Ok((
            bundle,
            ZipSplatLoadReport {
                model: ZipSplatApplySummary::from_apply_result(&apply),
            },
        ))
    }

    pub fn initialized_for_smoke(
        device: &B::Device,
        cfg: ZipSplatConfig,
    ) -> (Self, ZipSplatLoadReport) {
        (
            Self::new(device, cfg),
            ZipSplatLoadReport {
                model: ZipSplatApplySummary::synthetic_success(0),
            },
        )
    }

    #[cfg(feature = "import")]
    pub fn load_from_burnpack_part_bytes_with_progress<F>(
        device: &B::Device,
        cfg: ZipSplatConfig,
        parts: &[Vec<u8>],
        mut progress: F,
    ) -> Result<(Self, ZipSplatLoadReport), ZipSplatPipelineError>
    where
        F: FnMut(String),
    {
        if parts.is_empty() {
            return Err(ZipSplatPipelineError::Import(
                "no ZipSplat burnpack parts supplied".to_string(),
            ));
        }

        let mut bundle = Self::new(device, cfg);
        let mut applied = BTreeSet::new();
        let total = parts.len();
        for (index, part) in parts.iter().enumerate() {
            progress(format!(
                "loading zipsplat model part {}/{}",
                index + 1,
                total
            ));
            let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(part.clone())))
                .allow_partial(true)
                .validate(true);
            let result = bundle
                .load_from(&mut store)
                .map_err(|err| ZipSplatPipelineError::Import(format!("{err:?}")))?;
            for key in result.applied {
                applied.insert(key);
            }
        }
        Ok((
            bundle,
            ZipSplatLoadReport {
                model: ZipSplatApplySummary {
                    applied: applied.len(),
                    missing: Vec::new(),
                    unused: Vec::new(),
                    skipped: Vec::new(),
                },
            },
        ))
    }

    pub fn forward_from_tensor_timed_with_sync(
        &self,
        images: Tensor<B, 5>,
        compression: ZipSplatCompression,
        synchronize: bool,
    ) -> (FlatGaussians<B>, ZipSplatForwardTimings) {
        let total_start = ForwardTimer::now();
        let backbone_start = ForwardTimer::now();
        let output = self.model.forward(images, compression);
        if synchronize {
            sync_tensor_2d(output.opacities.clone());
        }
        let total = total_start.elapsed();
        (
            output,
            ZipSplatForwardTimings {
                image_load: Duration::ZERO,
                backbone: backbone_start.elapsed(),
                head: Duration::ZERO,
                total,
            },
        )
    }

    #[cfg(feature = "io")]
    pub fn forward_from_image_paths_timed_with_sync(
        &self,
        image_paths: &[PathBuf],
        image_size: usize,
        compression: ZipSplatCompression,
        device: &B::Device,
        synchronize: bool,
    ) -> Result<(FlatGaussians<B>, ZipSplatForwardTimings), ZipSplatPipelineError> {
        let total_start = ForwardTimer::now();
        let load_start = ForwardTimer::now();
        let images = load_multi_image_tensor_from_paths::<B>(image_paths, image_size, device)?;
        let image_load = load_start.elapsed();
        let (output, mut timings) =
            self.forward_from_tensor_timed_with_sync(images, compression, synchronize);
        timings.image_load = image_load;
        timings.total = total_start.elapsed();
        Ok((output, timings))
    }

    #[cfg(feature = "io")]
    pub fn forward_from_image_bytes_timed_with_sync(
        &self,
        named_images: &[(&str, &[u8])],
        image_size: usize,
        compression: ZipSplatCompression,
        device: &B::Device,
        synchronize: bool,
    ) -> Result<(FlatGaussians<B>, ZipSplatForwardTimings), ZipSplatPipelineError> {
        let total_start = ForwardTimer::now();
        let load_start = ForwardTimer::now();
        let images = load_multi_image_tensor_from_bytes::<B>(named_images, image_size, device)?;
        let image_load = load_start.elapsed();
        let (output, mut timings) =
            self.forward_from_tensor_timed_with_sync(images, compression, synchronize);
        timings.image_load = image_load;
        timings.total = total_start.elapsed();
        Ok((output, timings))
    }
}

#[cfg(feature = "import")]
impl ZipSplatApplySummary {
    fn from_apply_result(result: &ApplyResult) -> Self {
        Self {
            applied: result.applied.len(),
            missing: result.missing.clone(),
            unused: result.unused.clone(),
            skipped: result.skipped.clone(),
        }
    }
}

fn softplus<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    (tensor.exp() + 1.0).log()
}

fn quaternion_wxyz_to_xyzw<B: Backend>(quats: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch, count, _] = quats.shape().dims::<3>();
    let xyz = quats
        .clone()
        .slice([0..batch as i32, 0..count as i32, 1..4]);
    let w = quats.slice([0..batch as i32, 0..count as i32, 0..1]);
    Tensor::cat(vec![xyz, w], 2)
}

fn swiglu_hidden_dim(dim: usize, mlp_ratio: f32) -> usize {
    let hidden = (dim as f32 * mlp_ratio) as usize;
    let fused = (hidden * 2).div_ceil(3);
    fused.div_ceil(8) * 8
}

fn replace_camera_token<B: Backend>(tokens: Tensor<B, 4>, camera: Tensor<B, 3>) -> Tensor<B, 4> {
    let [batch, views, token_count, dim] = tokens.shape().dims::<4>();
    if token_count <= 1 {
        return camera.reshape([batch as i32, views as i32, 1, dim as i32]);
    }
    let tail = tokens.slice([
        0..batch as i32,
        0..views as i32,
        1..token_count as i32,
        0..dim as i32,
    ]);
    Tensor::cat(
        vec![
            camera.reshape([batch as i32, views as i32, 1, dim as i32]),
            tail,
        ],
        2,
    )
}

fn strip_camera_token<B: Backend>(tokens: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, views, token_count, dim] = tokens.shape().dims::<4>();
    tokens.slice([
        0..batch as i32,
        0..views as i32,
        1..token_count as i32,
        0..dim as i32,
    ])
}

fn flatten_view_tokens<B: Backend>(tokens: Tensor<B, 4>) -> Tensor<B, 3> {
    let [batch, views, token_count, dim] = tokens.shape().dims::<4>();
    tokens.reshape([batch as i32, (views * token_count) as i32, dim as i32])
}

fn gather_layer_tokens<B: Backend>(
    tokens: Tensor<B, 4>,
    indices: Tensor<B, 2, Int>,
) -> Tensor<B, 3> {
    gather_flat_tokens(flatten_view_tokens(tokens), indices)
}

fn gather_flat_tokens<B: Backend>(
    tokens: Tensor<B, 3>,
    indices: Tensor<B, 2, Int>,
) -> Tensor<B, 3> {
    let dim = tokens.shape().dims::<3>()[2];
    let expanded = indices.unsqueeze_dim(2).repeat_dim(2, dim);
    tokens.gather(1, expanded)
}

pub fn retained_token_count(total_tokens: usize, compression: f32) -> usize {
    let compression = if compression.is_finite() {
        compression.clamp(0.0, 1.0)
    } else {
        1.0
    };
    ((total_tokens as f32 * compression) as usize).clamp(1, total_tokens.max(1))
}

pub fn linspace_token_indices(total_tokens: usize, retained: usize) -> Vec<usize> {
    let retained = retained.clamp(1, total_tokens.max(1));
    if retained >= total_tokens {
        return (0..total_tokens).collect();
    }
    if retained == 1 {
        return vec![0];
    }
    let step = (total_tokens - 1) as f32 / (retained - 1) as f32;
    (0..retained)
        .map(|idx| (idx as f32 * step).round() as usize)
        .collect()
}

#[cfg(test)]
fn kmeans_nearest_indices_host_tensor<B: Backend>(
    tokens: &Tensor<B, 4>,
    compression: ZipSplatCompression,
    n_iters: usize,
) -> Tensor<B, 2, Int> {
    let [batch, views, per_view_tokens, dim] = tokens.shape().dims::<4>();
    let total_tokens = views * per_view_tokens;
    let retained = compression.retained_tokens(total_tokens);
    let device = tokens.device();
    if retained >= total_tokens {
        return Tensor::<B, 1, Int>::arange(0..total_tokens as i64, &device)
            .reshape([1, total_tokens as i32])
            .repeat_dim(0, batch);
    }

    let flat = flatten_view_tokens(tokens.clone());
    let data = flat
        .into_data()
        .to_vec::<f32>()
        .expect("ZipSplat kmeans token tensor must be readable as f32");
    let batch_stride = total_tokens * dim;
    let mut indices = Vec::with_capacity(batch * retained);
    for batch_idx in 0..batch {
        let start = batch_idx * batch_stride;
        let end = start + batch_stride;
        indices.extend(
            kmeans_nearest_indices_host(&data[start..end], total_tokens, dim, retained, n_iters)
                .into_iter()
                .map(|idx| idx as i32),
        );
    }

    Tensor::<B, 1, Int>::from_ints(indices.as_slice(), &device)
        .reshape([batch as i32, retained as i32])
}

pub fn kmeans_nearest_indices_tensor<B: Backend>(
    tokens: &Tensor<B, 4>,
    compression: ZipSplatCompression,
    n_iters: usize,
    _chunk_size: usize,
) -> Tensor<B, 2, Int> {
    let [batch, views, per_view_tokens, dim] = tokens.shape().dims::<4>();
    let total_tokens = views * per_view_tokens;
    let retained = compression.retained_tokens(total_tokens);
    let device = tokens.device();
    if retained >= total_tokens {
        return Tensor::<B, 1, Int>::arange(0..total_tokens as i64, &device)
            .reshape([1, total_tokens as i32])
            .repeat_dim(0, batch);
    }

    let flat = flatten_view_tokens(tokens.clone());
    let init = linspace_token_indices(total_tokens, retained)
        .into_iter()
        .map(|value| value as i32)
        .collect::<Vec<_>>();
    let init = Tensor::<B, 1, Int>::from_ints(init.as_slice(), &device)
        .reshape([1, retained as i32, 1])
        .repeat_dim(0, batch)
        .repeat_dim(2, dim);
    let mut centroids = flat.clone().gather(1, init);

    for _ in 0..n_iters.max(1) {
        let assignments: Tensor<B, 2, Int> =
            pairwise_squared_distances(flat.clone(), centroids.clone())
                .argmin(2)
                .squeeze_dim(2);
        let class_ids = Tensor::<B, 1, Int>::arange(0..retained as i64, &device)
            .reshape([1, 1, retained as i32])
            .repeat_dim(0, batch)
            .repeat_dim(1, total_tokens);
        let weights = assignments
            .unsqueeze_dim(2)
            .repeat_dim(2, retained)
            .equal(class_ids)
            .float();
        let next = weights.clone().swap_dims(1, 2).matmul(flat.clone());
        let counts = weights.sum_dim(1).swap_dims(1, 2).clamp_min(1.0);
        centroids = next / counts;
    }

    pairwise_squared_distances(centroids, flat)
        .argmin(2)
        .squeeze_dim(2)
}

fn pairwise_squared_distances<B: Backend>(lhs: Tensor<B, 3>, rhs: Tensor<B, 3>) -> Tensor<B, 3> {
    let lhs_norm = lhs.clone().powi_scalar(2).sum_dim(2);
    let rhs_norm = rhs.clone().powi_scalar(2).sum_dim(2).swap_dims(1, 2);
    let cross = lhs.matmul(rhs.swap_dims(1, 2));
    (lhs_norm + rhs_norm - cross.mul_scalar(2.0)).clamp_min(0.0)
}

pub fn kmeans_nearest_indices_host(
    tokens: &[f32],
    token_count: usize,
    dim: usize,
    retained: usize,
    n_iters: usize,
) -> Vec<usize> {
    let retained = retained.clamp(1, token_count.max(1));
    if retained >= token_count {
        return (0..token_count).collect();
    }

    let init = linspace_token_indices(token_count, retained);
    let mut centroids = vec![0.0f32; retained * dim];
    for (centroid_idx, token_idx) in init.iter().enumerate() {
        centroids[centroid_idx * dim..(centroid_idx + 1) * dim]
            .copy_from_slice(&tokens[token_idx * dim..(token_idx + 1) * dim]);
    }

    let mut assignments = vec![0usize; token_count];
    for _ in 0..n_iters.max(1) {
        for token_idx in 0..token_count {
            assignments[token_idx] = nearest_centroid(
                &tokens[token_idx * dim..(token_idx + 1) * dim],
                &centroids,
                retained,
                dim,
            );
        }

        let mut next = vec![0.0f32; retained * dim];
        let mut counts = vec![0usize; retained];
        for token_idx in 0..token_count {
            let assignment = assignments[token_idx];
            counts[assignment] += 1;
            for d in 0..dim {
                next[assignment * dim + d] += tokens[token_idx * dim + d];
            }
        }
        for centroid_idx in 0..retained {
            let count = counts[centroid_idx].max(1) as f32;
            for d in 0..dim {
                next[centroid_idx * dim + d] /= count;
            }
        }
        centroids = next;
    }

    (0..retained)
        .map(|centroid_idx| {
            nearest_token(
                &centroids[centroid_idx * dim..(centroid_idx + 1) * dim],
                tokens,
                token_count,
                dim,
            )
        })
        .collect()
}

fn nearest_centroid(token: &[f32], centroids: &[f32], centroid_count: usize, dim: usize) -> usize {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;
    for centroid_idx in 0..centroid_count {
        let start = centroid_idx * dim;
        let dist = squared_distance(token, &centroids[start..start + dim]);
        if dist < best_dist {
            best = centroid_idx;
            best_dist = dist;
        }
    }
    best
}

fn nearest_token(centroid: &[f32], tokens: &[f32], token_count: usize, dim: usize) -> usize {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;
    for token_idx in 0..token_count {
        let start = token_idx * dim;
        let dist = squared_distance(centroid, &tokens[start..start + dim]);
        if dist < best_dist {
            best = token_idx;
            best_dist = dist;
        }
    }
    best
}

fn squared_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| {
            let diff = left - right;
            diff * diff
        })
        .sum()
}

#[cfg(feature = "io")]
pub fn load_multi_image_tensor_from_paths<B: Backend>(
    image_paths: &[PathBuf],
    image_size: usize,
    device: &B::Device,
) -> Result<Tensor<B, 5>, ZipSplatPipelineError> {
    validate_image_request(image_paths.len(), image_size)?;
    let mut data = Vec::with_capacity(image_paths.len() * 3 * image_size * image_size);
    for path in image_paths {
        let image = image::open(path).map_err(|source| ZipSplatPipelineError::ImageLoad {
            path: path.display().to_string(),
            source,
        })?;
        append_image_chw_f32(image, image_size, &mut data);
    }
    Ok(
        Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([
            1,
            image_paths.len() as i32,
            3,
            image_size as i32,
            image_size as i32,
        ]),
    )
}

#[cfg(feature = "io")]
pub fn load_multi_image_tensor_from_bytes<B: Backend>(
    named_images: &[(&str, &[u8])],
    image_size: usize,
    device: &B::Device,
) -> Result<Tensor<B, 5>, ZipSplatPipelineError> {
    validate_image_request(named_images.len(), image_size)?;
    let mut data = Vec::with_capacity(named_images.len() * 3 * image_size * image_size);
    for (name, bytes) in named_images {
        let image = image::load_from_memory(bytes).map_err(|source| {
            ZipSplatPipelineError::ImageDecode {
                name: (*name).to_string(),
                source,
            }
        })?;
        append_image_chw_f32(image, image_size, &mut data);
    }
    Ok(
        Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([
            1,
            named_images.len() as i32,
            3,
            image_size as i32,
            image_size as i32,
        ]),
    )
}

#[cfg(feature = "io")]
fn append_image_chw_f32(image: image::DynamicImage, image_size: usize, out: &mut Vec<f32>) {
    let source = image.to_rgb8();
    let (width, height) = source.dimensions();
    let side = width.min(height);
    let left = (width - side) / 2;
    let top = (height - side) / 2;
    let square = image::imageops::crop_imm(&source, left, top, side, side).to_image();
    let rgb = image::imageops::resize(
        &square,
        image_size as u32,
        image_size as u32,
        image::imageops::FilterType::Lanczos3,
    );
    for channel in 0..3usize {
        for y in 0..image_size {
            for x in 0..image_size {
                out.push(rgb.get_pixel(x as u32, y as u32)[channel] as f32 / 255.0);
            }
        }
    }
}

fn validate_image_request(count: usize, image_size: usize) -> Result<(), ZipSplatPipelineError> {
    if !image_size.is_multiple_of(IMAGE_SIZE_MULTIPLE) {
        return Err(ZipSplatPipelineError::InvalidImageSize(image_size));
    }
    if !(MIN_VIEWS..=MAX_VIEWS).contains(&count) {
        return Err(ZipSplatPipelineError::InvalidViewCount(count));
    }
    Ok(())
}

fn sync_tensor_2d<B: Backend>(tensor: Tensor<B, 2>) {
    let [batch, count] = tensor.shape().dims::<2>();
    if batch == 0 || count == 0 {
        return;
    }
    let _ = tensor.slice([0..1, 0..1]).into_data().to_vec::<f32>();
}

pub fn burnpack_path_for_precision(path: &Path, precision: ZipSplatWeightPrecision) -> StdPathBuf {
    match precision {
        ZipSplatWeightPrecision::F32 => path.to_path_buf(),
        ZipSplatWeightPrecision::F16 => {
            let stem = path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("zipsplat");
            if stem.ends_with("_f16") {
                return path.to_path_buf();
            }
            let extension = path
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or("bpk");
            path.with_file_name(format!("{stem}_f16.{extension}"))
        }
    }
}

#[cfg(feature = "import")]
pub mod import {
    use std::path::{Path, PathBuf};

    use burn::prelude::Backend;
    use burn_store::{ApplyResult, BurnpackStore, KeyRemapper, ModuleSnapshot, PytorchStore};

    use crate::{
        burnpack_path_for_precision, ZipSplatConfig, ZipSplatModelBundle, ZipSplatWeightPrecision,
    };

    #[derive(Debug, thiserror::Error)]
    pub enum ZipSplatImportError {
        #[error("invalid key remap rule `{0}` -> `{1}`: {2}")]
        InvalidRemap(String, String, String),
        #[error("failed to apply checkpoint tensors: {0}")]
        Apply(String),
        #[error("failed to save ZipSplat burnpack: {0}")]
        SaveBurnpack(String),
        #[error("failed to convert ZipSplat burnpack precision: {0}")]
        Precision(String),
    }

    pub fn load_zipsplat_from_pytorch<B: Backend>(
        device: &B::Device,
        cfg: ZipSplatConfig,
        path: &Path,
    ) -> Result<(ZipSplatModelBundle<B>, ApplyResult), ZipSplatImportError> {
        let mut model = ZipSplatModelBundle::new(device, cfg);
        let mut store = build_zipsplat_pytorch_store(path)?;
        let result = model
            .load_from(&mut store)
            .map_err(|err| ZipSplatImportError::Apply(format!("{err:?}")))?;
        Ok((model, result))
    }

    pub fn save_zipsplat_record_bpk<B: Backend>(
        model: &ZipSplatModelBundle<B>,
        output_base: &Path,
    ) -> Result<PathBuf, ZipSplatImportError> {
        save_zipsplat_record_bpk_with_precision(model, output_base, ZipSplatWeightPrecision::F32)
    }

    pub fn save_zipsplat_record_bpk_with_precision<B: Backend>(
        model: &ZipSplatModelBundle<B>,
        output_base: &Path,
        precision: ZipSplatWeightPrecision,
    ) -> Result<PathBuf, ZipSplatImportError> {
        let base = normalize_extension(output_base, "bpk");
        match precision {
            ZipSplatWeightPrecision::F32 => {
                let output =
                    burnpack_path_for_precision(base.as_path(), ZipSplatWeightPrecision::F32);
                let mut store = BurnpackStore::from_file(&output)
                    .auto_extension(false)
                    .overwrite(true);
                model
                    .save_into(&mut store)
                    .map_err(|err| ZipSplatImportError::SaveBurnpack(err.to_string()))?;
                Ok(output)
            }
            ZipSplatWeightPrecision::F16 => {
                let source =
                    burnpack_path_for_precision(base.as_path(), ZipSplatWeightPrecision::F32);
                let mut store = BurnpackStore::from_file(&source)
                    .auto_extension(false)
                    .overwrite(true);
                model
                    .save_into(&mut store)
                    .map_err(|err| ZipSplatImportError::SaveBurnpack(err.to_string()))?;
                burn_yono::import::convert_burnpack_to_f16(source.as_path(), base.as_path())
                    .map_err(|err| ZipSplatImportError::Precision(err.to_string()))
            }
        }
    }

    pub fn zipsplat_key_remap_rules() -> &'static [(&'static str, &'static str)] {
        &[
            (r"^module\.", ""),
            (r"^backbone\.backbone\.(.*)$", "model.backbone.$1"),
            (r"^pre_norm_local\.(.*)$", "model.pre_norm_local.$1"),
            (r"^pre_norm_global\.(.*)$", "model.pre_norm_global.$1"),
            (r"^downscale\.(.*)$", "model.downscale.$1"),
            (r"^cross_attention\.(.*)$", "model.cross_attention.$1"),
            (r"^self_attention\.(.*)$", "model.self_attention.$1"),
            (r"^color_embed\.(.*)$", "model.color_embed.$1"),
            (
                r"^color_cross_attention\.(.*)$",
                "model.color_cross_attention.$1",
            ),
            (
                r"^gaussian_head\.gaussian_head\.0\.(.*)$",
                "model.gaussian_head.mlp.$1",
            ),
            (
                r"^gaussian_head\.gaussian_head\.1\.(.*)$",
                "model.gaussian_head.linear.$1",
            ),
            (
                r"^(model\.pre_norm_(?:local|global)\.\d+)\.weight$",
                "$1.gamma",
            ),
            (
                r"^(model\.pre_norm_(?:local|global)\.\d+)\.bias$",
                "$1.beta",
            ),
            (r"^(model\..*\.norm\d?)\.weight$", "$1.gamma"),
            (r"^(model\..*\.norm\d?)\.bias$", "$1.beta"),
            (r"^(model\..*\.norm_q)\.weight$", "$1.gamma"),
            (r"^(model\..*\.norm_q)\.bias$", "$1.beta"),
            (r"^(model\..*\.norm_kv)\.weight$", "$1.gamma"),
            (r"^(model\..*\.norm_kv)\.bias$", "$1.beta"),
            (r"^(model\..*\.q_norm)\.weight$", "$1.gamma"),
            (r"^(model\..*\.q_norm)\.bias$", "$1.beta"),
            (r"^(model\..*\.k_norm)\.weight$", "$1.gamma"),
            (r"^(model\..*\.k_norm)\.bias$", "$1.beta"),
            (r"^(model\..*\.norm)\.weight$", "$1.gamma"),
            (r"^(model\..*\.norm)\.bias$", "$1.beta"),
        ]
    }

    pub fn build_zipsplat_key_remapper() -> Result<KeyRemapper, ZipSplatImportError> {
        let mut remapper = KeyRemapper::new();
        for &(from, to) in zipsplat_key_remap_rules() {
            remapper = remapper.add_pattern(from, to).map_err(|err| {
                ZipSplatImportError::InvalidRemap(from.to_string(), to.to_string(), err.to_string())
            })?;
        }
        Ok(remapper)
    }

    fn build_zipsplat_pytorch_store(path: &Path) -> Result<PytorchStore, ZipSplatImportError> {
        Ok(PytorchStore::from_file(path)
            .with_top_level_key("model")
            .allow_partial(true)
            .remap(build_zipsplat_key_remapper()?)
            .validate(true))
    }

    fn normalize_extension(path: &Path, extension: &str) -> PathBuf {
        if path
            .extension()
            .map(|ext| ext.eq_ignore_ascii_case(extension))
            .unwrap_or(false)
        {
            path.to_path_buf()
        } else {
            path.with_extension(extension)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(feature = "backend_cuda", not(feature = "backend_ndarray")))]
    type TestBackend = burn::backend::Cuda<f32>;
    #[cfg(feature = "backend_ndarray")]
    type TestBackend = burn::backend::NdArray<f32>;

    #[cfg(all(feature = "backend_cuda", not(feature = "backend_ndarray")))]
    fn int_tensor_to_i64<const D: usize>(tensor: Tensor<TestBackend, D, Int>) -> Vec<i64> {
        tensor
            .into_data()
            .to_vec::<i32>()
            .expect("int tensor data")
            .into_iter()
            .map(i64::from)
            .collect()
    }

    #[cfg(not(all(feature = "backend_cuda", not(feature = "backend_ndarray"))))]
    fn int_tensor_to_i64<const D: usize>(tensor: Tensor<TestBackend, D, Int>) -> Vec<i64> {
        tensor.into_data().to_vec::<i64>().expect("int tensor data")
    }

    #[test]
    fn compression_is_clamped_to_supported_range() {
        assert_eq!(ZipSplatCompression::new(0).get(), MIN_COMPRESSION_R);
        assert_eq!(ZipSplatCompression::new(4).get(), 4);
        assert_eq!(ZipSplatCompression::new(999).get(), MAX_COMPRESSION_R);
    }

    #[test]
    fn preset_rs_match_expected_quality_levels() {
        let rs = QUALITY_PRESETS
            .iter()
            .map(|preset| preset.compression.get())
            .collect::<Vec<_>>();
        assert_eq!(rs, vec![1, 2, 4, 8]);
    }

    #[test]
    fn default_weights_are_native_burnpack() {
        let weights = ZipSplatWeights::default();
        assert_eq!(weights.format, ZipSplatWeightFormat::Burnpack);
        assert_eq!(
            weights.checkpoint,
            StdPathBuf::from(DEFAULT_CHECKPOINT_PATH)
        );
    }

    #[test]
    fn burnpack_precision_path_prefers_single_f16_suffix() {
        assert_eq!(
            burnpack_path_for_precision(
                Path::new("assets/models/zipsplat.bpk"),
                ZipSplatWeightPrecision::F16
            ),
            StdPathBuf::from("assets/models/zipsplat_f16.bpk")
        );
        assert_eq!(
            burnpack_path_for_precision(
                Path::new("assets/models/zipsplat_f16.bpk"),
                ZipSplatWeightPrecision::F16
            ),
            StdPathBuf::from("assets/models/zipsplat_f16.bpk")
        );
    }

    #[cfg(feature = "import")]
    #[test]
    fn pytorch_remapper_maps_pre_norm_layer_norm_keys() {
        let remapper = import::build_zipsplat_key_remapper().expect("valid zipsplat remapper");
        let snapshot = burn_store::TensorSnapshot::from_data(
            burn::tensor::TensorData {
                bytes: Bytes::from_bytes_vec(vec![0, 0, 0, 0]),
                shape: vec![1],
                dtype: burn::tensor::DType::F32,
            },
            "pre_norm_global.1.weight"
                .split('.')
                .map(str::to_string)
                .collect(),
            vec!["ZipSplat".to_string()],
            burn::module::ParamId::new(),
        );
        let (snapshots, names) = remapper.remap(vec![snapshot]);

        assert_eq!(snapshots[0].full_path(), "model.pre_norm_global.1.gamma");
        assert_eq!(
            names,
            vec![(
                "model.pre_norm_global.1.gamma".to_string(),
                "pre_norm_global.1.weight".to_string(),
            )]
        );
    }

    #[test]
    fn compression_maps_r_to_upstream_ratio() {
        assert!((ZipSplatCompression::new(1).compression_ratio() - 1.0).abs() < f32::EPSILON);
        assert!((ZipSplatCompression::new(4).compression_ratio() - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn retained_tokens_follow_upstream_floor_behavior() {
        assert_eq!(retained_token_count(10, 1.0), 10);
        assert_eq!(retained_token_count(10, 0.5), 5);
        assert_eq!(retained_token_count(10, 0.25), 2);
        assert_eq!(retained_token_count(3, 0.01), 1);
    }

    #[test]
    fn estimated_gaussian_count_matches_upstream_token_formula() {
        let cfg = ZipSplatConfig::default();
        assert_eq!(cfg.patch_tokens_per_view(), 18 * 18);
        assert_eq!(
            cfg.estimated_gaussian_count(3, ZipSplatCompression::FULL),
            3 * 18 * 18 * 32
        );
        assert_eq!(
            cfg.estimated_gaussian_count(MAX_VIEWS, ZipSplatCompression::FULL),
            24 * 18 * 18 * 32
        );
        assert_eq!(
            cfg.estimated_gaussian_count(MAX_VIEWS, ZipSplatCompression::COMPACT),
            (24 * 18 * 18 / 4) * 32
        );
    }

    #[test]
    fn linspace_indices_match_upstream_initialization_style() {
        assert_eq!(linspace_token_indices(8, 4), vec![0, 2, 5, 7]);
        assert_eq!(linspace_token_indices(4, 4), vec![0, 1, 2, 3]);
    }

    #[test]
    fn host_kmeans_returns_nearest_original_tokens() {
        let tokens = [
            0.0f32, 0.0, //
            0.1, 0.0, //
            5.0, 4.0, //
            8.2, 8.0, //
        ];
        let indices = kmeans_nearest_indices_host(&tokens, 4, 2, 2, 3);
        assert_eq!(indices.len(), 2);
        assert!(indices.iter().all(|idx| *idx < 4));
    }

    #[test]
    fn tensor_kmeans_matches_host_reference_for_small_fixture() {
        let device = <TestBackend as Backend>::Device::default();
        let tokens = [
            0.0f32, 0.0, //
            0.1, 0.0, //
            5.0, 4.0, //
            8.2, 8.0, //
        ];
        let tensor =
            Tensor::<TestBackend, 1>::from_floats(tokens.as_slice(), &device).reshape([1, 1, 4, 2]);
        let expected = kmeans_nearest_indices_host(&tokens, 4, 2, 2, 3)
            .into_iter()
            .map(|value| value as i64)
            .collect::<Vec<_>>();
        let actual = int_tensor_to_i64(kmeans_nearest_indices_tensor(
            &tensor,
            ZipSplatCompression::BALANCED,
            3,
            2048,
        ));
        assert_eq!(actual, expected);
    }

    #[test]
    fn host_tensor_kmeans_preserves_batch_indices() {
        let device = <TestBackend as Backend>::Device::default();
        let tokens = [
            0.0f32, 0.0, //
            0.1, 0.0, //
            5.0, 4.0, //
            8.2, 8.0, //
            10.0, 10.0, //
            10.2, 10.0, //
            -5.0, -4.0, //
            -8.2, -8.0, //
        ];
        let tensor =
            Tensor::<TestBackend, 1>::from_floats(tokens.as_slice(), &device).reshape([2, 1, 4, 2]);

        let expected = [
            kmeans_nearest_indices_host(&tokens[0..8], 4, 2, 2, 3),
            kmeans_nearest_indices_host(&tokens[8..16], 4, 2, 2, 3),
        ]
        .concat()
        .into_iter()
        .map(|value| value as i64)
        .collect::<Vec<_>>();
        let actual = int_tensor_to_i64(kmeans_nearest_indices_host_tensor(
            &tensor,
            ZipSplatCompression::BALANCED,
            3,
        ));

        assert_eq!(actual, expected);
    }

    #[test]
    fn zipsplat_wxyz_quaternions_convert_to_repo_xyzw_convention() {
        let device = <TestBackend as Backend>::Device::default();
        let quats = Tensor::<TestBackend, 1>::from_floats(
            [
                1.0f32, 0.0, 0.0, 0.0, //
                0.5, 0.5, 0.5, 0.5, //
            ]
            .as_slice(),
            &device,
        )
        .reshape([1, 2, 4]);
        let actual = quaternion_wxyz_to_xyzw(quats)
            .into_data()
            .to_vec::<f32>()
            .expect("float tensor data");
        assert_eq!(actual, vec![0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn tiny_model_smoke_outputs_flat_gaussians() {
        let device = <TestBackend as Backend>::Device::default();
        let cfg = ZipSplatConfig::tiny_for_tests();
        let (bundle, _) =
            ZipSplatModelBundle::<TestBackend>::initialized_for_smoke(&device, cfg.clone());
        let images =
            Tensor::<TestBackend, 5>::zeros([1, 2, 3, cfg.image_size, cfg.image_size], &device);
        let output = bundle.model.forward(images, ZipSplatCompression::BALANCED);
        let [batch, count, xyz] = output.means.shape().dims::<3>();
        assert_eq!(batch, 1);
        assert_eq!(xyz, 3);
        assert!(count > 0);
        assert_eq!(output.harmonics.shape().dims::<4>()[3], cfg.d_sh());
    }
}
