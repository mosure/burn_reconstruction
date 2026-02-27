use burn::{module::Ignored, module::Param, nn::Initializer, prelude::*};
use burn_dino::{
    layers::{block::Block, block::BlockConfig},
    model::dino::{DinoOutput, DinoVisionTransformer, DinoVisionTransformerConfig},
};

use super::{ops::position_getter, patch_embed::PatchEmbedNorm, patch_embed::PatchEmbedNormConfig};

#[derive(Config, Debug)]
pub struct CrocoStyleBackboneConfig {
    #[config(default = "DinoVisionTransformerConfig::vitl(Some(224), Some(14))")]
    pub encoder: DinoVisionTransformerConfig,
    #[config(default = 1024)]
    pub decoder_embed_dim: usize,
    #[config(default = 16)]
    pub decoder_heads: usize,
    #[config(default = 8)]
    pub decoder_depth: usize,
    #[config(default = 4.0)]
    pub decoder_mlp_ratio: f32,
    #[config(default = 100.0)]
    pub rope_frequency: f32,
    #[config(default = true)]
    pub decoder_qk_norm: bool,
    #[config(default = 5)]
    pub register_token_count: usize,
    #[config(default = true)]
    pub alternating_local_global: bool,
    #[config(default = true)]
    pub use_intrinsics_embedding: bool,
    #[config(default = 4)]
    pub intrinsics_embed_degree: usize,
}

#[derive(Debug)]
pub struct CrocoBackboneOutput<B: Backend> {
    pub hidden: Tensor<B, 3>,
    pub pos: Tensor<B, 3>,
    pub patch_start_idx: usize,
    pub encoder: DinoOutput<B>,
}

#[derive(Module, Debug)]
pub struct CrocoStyleBackbone<B: Backend> {
    pub encoder: DinoVisionTransformer<B>,
    pub proj: Option<nn::Linear<B>>,
    pub blocks: Vec<Block<B>>,
    pub register_token: Param<Tensor<B, 3>>,
    pub intrinsics_embed: Option<PatchEmbedNorm<B>>,
    cfg: Ignored<CrocoStyleBackboneConfig>,
}

impl<B: Backend> CrocoStyleBackbone<B> {
    pub fn new(device: &B::Device, cfg: CrocoStyleBackboneConfig) -> Self {
        let encoder = cfg.encoder.clone().init(device);
        let encoder_dim = cfg.encoder.embedding_dimension;
        let dec_dim = cfg.decoder_embed_dim;

        let proj = if encoder_dim == dec_dim {
            None
        } else {
            Some(nn::LinearConfig::new(encoder_dim, dec_dim).init(device))
        };

        let block_cfg = BlockConfig {
            attn: burn_dino::layers::attention::AttentionConfig {
                dim: dec_dim,
                num_heads: cfg.decoder_heads,
                qkv_bias: true,
                proj_bias: true,
                attn_drop: 0.0,
                proj_drop: 0.0,
                quiet_softmax: false,
                qk_norm: cfg.decoder_qk_norm,
                rope: Some(burn_dino::layers::rope::RopeConfig {
                    base_frequency: cfg.rope_frequency,
                }),
            },
            layer_scale: Some(burn_dino::layers::layer_scale::LayerScaleConfig { dim: dec_dim }),
            mlp_ratio: cfg.decoder_mlp_ratio,
        };

        let mut blocks = Vec::with_capacity(cfg.decoder_depth);
        for _ in 0..cfg.decoder_depth {
            blocks.push(block_cfg.clone().init(device));
        }

        let register_token = Initializer::Normal {
            mean: 0.0,
            std: 1e-6,
        }
        .init([1, cfg.register_token_count, dec_dim], device);

        let intrinsics_embed = if cfg.use_intrinsics_embedding {
            Some(
                PatchEmbedNormConfig::new(
                    cfg.encoder.patch_size,
                    if cfg.intrinsics_embed_degree > 0 {
                        (cfg.intrinsics_embed_degree + 1).pow(2)
                    } else {
                        3
                    },
                    dec_dim,
                )
                .with_use_norm(true)
                .init(device),
            )
        } else {
            None
        };

        Self {
            encoder,
            proj,
            blocks,
            register_token,
            intrinsics_embed,
            cfg: Ignored(cfg),
        }
    }

    pub fn patch_start_idx(&self) -> usize {
        self.cfg.0.register_token_count
    }

    pub fn output_dim(&self) -> usize {
        // Same as local_global: concat of the last two decoder outputs.
        self.cfg.0.decoder_embed_dim * 2
    }

    pub fn forward(
        &self,
        image: Tensor<B, 5>,
        intrinsics: Option<Tensor<B, 4>>,
    ) -> CrocoBackboneOutput<B> {
        self.forward_impl(image, intrinsics, None)
    }

    pub fn forward_with_normalized_intrinsics(
        &self,
        image: Tensor<B, 5>,
    ) -> CrocoBackboneOutput<B> {
        let [b, v, _, _, _] = image.shape().dims::<5>();
        let intrinsics = normalized_intrinsics_for_dims::<B>(b, v, &image.device());
        self.forward_impl(image, Some(intrinsics), None)
    }

    fn forward_impl(
        &self,
        image: Tensor<B, 5>,
        intrinsics: Option<Tensor<B, 4>>,
        intrinsics_token_override: Option<Tensor<B, 3>>,
    ) -> CrocoBackboneOutput<B> {
        let cfg = &self.cfg.0;
        let [b, v, _, h, w] = image.shape().dims::<5>();
        let bv = b * v;
        let patch_h = h / cfg.encoder.patch_size;
        let patch_w = w / cfg.encoder.patch_size;

        let image_bv = normalize_imagenet(image.reshape([bv as i32, 3, h as i32, w as i32]));
        let encoder_out = self.encoder.forward(image_bv, None);
        let mut hidden = encoder_out.x_norm_patchtokens.clone();

        if let Some(proj) = &self.proj {
            hidden = proj.forward(hidden);
        }

        if let Some(tokens) = intrinsics_token_override {
            hidden = hidden + tokens;
        } else if let (Some(intrinsics), Some(embed)) = (intrinsics, &self.intrinsics_embed) {
            let intr_chw = intrinsic_embedding_image(
                intrinsics,
                h,
                w,
                cfg.intrinsics_embed_degree,
                cfg.encoder.patch_size,
                &hidden.device(),
            );
            hidden = hidden + embed.forward(intr_chw);
        }

        let reg = self
            .register_token
            .val()
            .clone()
            .repeat_dim(0, bv)
            .reshape([
                bv as i32,
                cfg.register_token_count as i32,
                cfg.decoder_embed_dim as i32,
            ]);
        let mut hidden = Tensor::cat(vec![reg, hidden], 1);

        let pos_img = position_getter(bv, patch_h, patch_w, &hidden.device()) + 1.0;
        let pos_special = Tensor::<B, 3>::zeros(
            [bv as i32, cfg.register_token_count as i32, 2],
            &hidden.device(),
        );
        let mut pos = Tensor::cat(vec![pos_special, pos_img], 1);

        let mut outputs = Vec::with_capacity(2);
        for (idx, block) in self.blocks.iter().enumerate() {
            let token_count = hidden.shape().dims::<3>()[1];
            if cfg.alternating_local_global && idx % 2 == 1 {
                hidden = hidden.reshape([
                    b as i32,
                    (v * token_count) as i32,
                    cfg.decoder_embed_dim as i32,
                ]);
                pos = pos.reshape([b as i32, (v * token_count) as i32, 2]);
                hidden = block.forward(hidden, Some(&pos), None);
                hidden =
                    hidden.reshape([bv as i32, token_count as i32, cfg.decoder_embed_dim as i32]);
                pos = pos.reshape([bv as i32, token_count as i32, 2]);
            } else {
                hidden = block.forward(hidden, Some(&pos), None);
            }
            if idx + 2 >= self.blocks.len() {
                outputs.push(hidden.clone());
            }
        }

        let hidden = if outputs.len() == 2 {
            Tensor::cat(vec![outputs[0].clone(), outputs[1].clone()], 2)
        } else {
            let last = hidden.clone();
            Tensor::cat(vec![last.clone(), last], 2)
        };

        CrocoBackboneOutput {
            hidden,
            pos,
            patch_start_idx: cfg.register_token_count,
            encoder: encoder_out,
        }
    }
}

fn normalized_intrinsics_for_dims<B: Backend>(
    batch: usize,
    views: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    Tensor::<B, 1>::from_floats(
        [
            1.0f32, 0.0, 0.5, //
            0.0, 1.0, 0.5, //
            0.0, 0.0, 1.0, //
        ]
        .repeat(batch * views)
        .as_slice(),
        device,
    )
    .reshape([batch as i32, views as i32, 3, 3])
}

fn intrinsic_embedding_image<B: Backend>(
    intrinsics: Tensor<B, 4>,
    h: usize,
    w: usize,
    degree: usize,
    _patch_size: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    if degree == 0 {
        return intrinsic_embedding_image_deg0(intrinsics, h, w, device);
    }
    assert!(
        degree == 4,
        "only degree=0 or degree=4 intrinsics embedding is supported, got degree={degree}"
    );
    intrinsic_embedding_image_deg4(intrinsics, h, w, device)
}

fn intrinsic_embedding_image_deg0<B: Backend>(
    intrinsics: Tensor<B, 4>,
    h: usize,
    w: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let [b, v, _, _] = intrinsics.shape().dims::<4>();
    let fx = intrinsics
        .clone()
        .slice([0..b as i32, 0..v as i32, 0..1, 0..1])
        .reshape([(b * v) as i32, 1, 1, 1])
        .repeat_dim(2, h)
        .repeat_dim(3, w);
    let fy = intrinsics
        .clone()
        .slice([0..b as i32, 0..v as i32, 1..2, 1..2])
        .reshape([(b * v) as i32, 1, 1, 1])
        .repeat_dim(2, h)
        .repeat_dim(3, w);
    let cx = intrinsics
        .slice([0..b as i32, 0..v as i32, 0..1, 2..3])
        .reshape([(b * v) as i32, 1, 1, 1])
        .repeat_dim(2, h)
        .repeat_dim(3, w);

    Tensor::cat(vec![fx, fy, cx], 1).to_device(device)
}

fn intrinsic_embedding_image_deg4<B: Backend>(
    intrinsics: Tensor<B, 4>,
    h: usize,
    w: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let [b, v, _, _] = intrinsics.shape().dims::<4>();
    let bv = b * v;

    let intr_flat = intrinsics.reshape([bv as i32, 9]);
    let fx = intr_flat
        .clone()
        .slice([0..bv as i32, 0..1])
        .reshape([bv as i32, 1, 1]);
    let fy = intr_flat
        .clone()
        .slice([0..bv as i32, 4..5])
        .reshape([bv as i32, 1, 1]);
    let cx = intr_flat
        .clone()
        .slice([0..bv as i32, 2..3])
        .reshape([bv as i32, 1, 1]);
    let cy = intr_flat
        .slice([0..bv as i32, 5..6])
        .reshape([bv as i32, 1, 1]);

    let mut coords = Vec::with_capacity(h * w * 2);
    for y in 0..h {
        for x in 0..w {
            coords.push((x as f32 + 0.5) / w as f32);
            coords.push((y as f32 + 0.5) / h as f32);
        }
    }

    let coords = Tensor::<B, 1>::from_floats(coords.as_slice(), device)
        .reshape([1, h as i32, w as i32, 2])
        .repeat_dim(0, bv);
    let u = coords
        .clone()
        .slice([0..bv as i32, 0..h as i32, 0..w as i32, 0..1])
        .reshape([bv as i32, h as i32, w as i32]);
    let v_coord = coords
        .slice([0..bv as i32, 0..h as i32, 0..w as i32, 1..2])
        .reshape([bv as i32, h as i32, w as i32]);

    let x = (u - cx.clone()) / fx.clone();
    let y = (v_coord - cy) / fy;
    let z = Tensor::<B, 3>::ones([bv as i32, h as i32, w as i32], device);
    let norm = (x.clone().powi_scalar(2) + y.clone().powi_scalar(2) + z.clone().powi_scalar(2))
        .sqrt()
        .clamp_min(1e-8);
    let x = x / norm.clone();
    let y = y / norm.clone();
    let z = z / norm;

    let x2 = x.clone().powi_scalar(2);
    let y2 = y.clone().powi_scalar(2);
    let z2 = z.clone().powi_scalar(2);
    let xy = x.clone() * y.clone();
    let xz = x.clone() * z.clone();
    let yz = y.clone() * z.clone();
    let x4 = x2.clone().powi_scalar(2);
    let y4 = y2.clone().powi_scalar(2);

    let c = |value: f32| Tensor::<B, 3>::ones_like(&x).mul_scalar(value);

    let terms = vec![
        c(0.282_094_8),
        y.clone().mul_scalar(-0.488_602_52),
        z.clone().mul_scalar(0.488_602_52),
        x.clone().mul_scalar(-0.488_602_52),
        xy.clone().mul_scalar(1.092_548_5),
        yz.clone().mul_scalar(-1.092_548_5),
        z2.clone().mul_scalar(0.946_174_7).add_scalar(-0.315_391_57),
        xz.clone().mul_scalar(-1.092_548_5),
        x2.clone()
            .mul_scalar(0.546_274_24)
            .sub(y2.clone().mul_scalar(0.546_274_24)),
        y.clone()
            .mul_scalar(-0.590_043_6)
            .mul(x2.clone().mul_scalar(3.0).sub(y2.clone())),
        xy.clone().mul(z.clone()).mul_scalar(2.890_611_4),
        y.clone()
            .mul(z2.clone().mul_scalar(-7.5).add_scalar(1.5))
            .mul_scalar(0.304_697_2),
        z.clone()
            .mul(z2.clone().mul_scalar(1.5).add_scalar(-0.5))
            .mul_scalar(1.243_921_2)
            .sub(z.clone().mul_scalar(0.497_568_46)),
        x.clone()
            .mul(z2.clone().mul_scalar(-7.5).add_scalar(1.5))
            .mul_scalar(0.304_697_2),
        z.clone()
            .mul(x2.clone().sub(y2.clone()))
            .mul_scalar(1.445_305_7),
        x.clone()
            .mul(x2.clone().sub(y2.clone().mul_scalar(3.0)))
            .mul_scalar(-0.590_043_6),
        xy.clone()
            .mul(x2.clone().sub(y2.clone()))
            .mul_scalar(2.503_343),
        yz.clone()
            .mul(x2.clone().mul_scalar(3.0).sub(y2.clone()))
            .mul_scalar(-1.770_130_8),
        xy.clone()
            .mul(z2.clone().mul_scalar(52.5).add_scalar(-7.5))
            .mul_scalar(0.126_156_63),
        y.clone()
            .mul(
                z.clone()
                    .mul(z2.clone().mul_scalar(-7.5).add_scalar(1.5))
                    .mul_scalar(2.333_333_3)
                    .add(z.clone().mul_scalar(4.0)),
            )
            .mul_scalar(0.267_618_63),
        z.clone()
            .mul(
                z.clone()
                    .mul(z2.clone().mul_scalar(1.5).add_scalar(-0.5))
                    .mul_scalar(1.666_666_6)
                    .sub(z.clone().mul_scalar(0.666_666_7)),
            )
            .mul_scalar(1.480_997_7)
            .sub(z2.clone().mul_scalar(0.952_069_94))
            .add_scalar(0.317_356_65),
        x.clone()
            .mul(
                z.clone()
                    .mul(z2.clone().mul_scalar(-7.5).add_scalar(1.5))
                    .mul_scalar(2.333_333_3)
                    .add(z.clone().mul_scalar(4.0)),
            )
            .mul_scalar(0.267_618_63),
        x2.clone()
            .sub(y2.clone())
            .mul(z2.clone().mul_scalar(52.5).add_scalar(-7.5))
            .mul_scalar(0.063_078_314),
        xz.clone()
            .mul(x2.clone().sub(y2.clone().mul_scalar(3.0)))
            .mul_scalar(-1.770_130_8),
        x2.clone()
            .mul(y2.clone())
            .mul_scalar(-3.755_014_4)
            .add(x4.mul_scalar(0.625_835_7))
            .add(y4.mul_scalar(0.625_835_7)),
    ];

    let mut terms4 = Vec::with_capacity(terms.len());
    for term in terms {
        terms4.push(term.unsqueeze_dim(3));
    }
    Tensor::cat(terms4, 3).permute([0, 3, 1, 2])
}

fn normalize_imagenet<B: Backend>(image: Tensor<B, 4>) -> Tensor<B, 4> {
    let mean = Tensor::<B, 1>::from_floats([0.485f32, 0.456, 0.406].as_slice(), &image.device())
        .reshape([1, 3, 1, 1]);
    let std = Tensor::<B, 1>::from_floats([0.229f32, 0.224, 0.225].as_slice(), &image.device())
        .reshape([1, 3, 1, 1]);
    (image - mean) / std
}

#[cfg(test)]
mod tests {
    use super::{
        intrinsic_embedding_image, normalize_imagenet, CrocoStyleBackbone, CrocoStyleBackboneConfig,
    };
    use burn::{prelude::*, tensor::Tensor};

    type TestBackend = burn::backend::NdArray<f32>;

    fn max_abs_diff<const D: usize>(
        lhs: Tensor<TestBackend, D>,
        rhs: Tensor<TestBackend, D>,
    ) -> f32 {
        lhs.into_data()
            .to_vec::<f32>()
            .expect("lhs should be readable")
            .into_iter()
            .zip(
                rhs.into_data()
                    .to_vec::<f32>()
                    .expect("rhs should be readable"),
            )
            .map(|(l, r)| (l - r).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn croco_style_backbone_forward_has_expected_shapes() {
        let device = <TestBackend as Backend>::Device::default();
        let cfg = CrocoStyleBackboneConfig::new()
            .with_encoder(
                burn_dino::model::dino::DinoVisionTransformerConfig::vits(Some(28), Some(14))
                    .without_register_tokens(),
            )
            .with_decoder_embed_dim(64)
            .with_decoder_heads(4)
            .with_decoder_depth(2)
            .with_register_token_count(5);
        let model = CrocoStyleBackbone::<TestBackend>::new(&device, cfg);

        let b = 1usize;
        let v = 2usize;
        let h = 28usize;
        let w = 28usize;
        let image = Tensor::<TestBackend, 1>::from_floats(
            (0..b * v * 3 * h * w)
                .map(|idx| (idx % 31) as f32 / 31.0)
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        )
        .reshape([b as i32, v as i32, 3, h as i32, w as i32]);
        let intrinsics = Tensor::<TestBackend, 1>::from_floats(
            [
                1.0f32, 0.0, 0.5, //
                0.0, 1.0, 0.5, //
                0.0, 0.0, 1.0, //
            ]
            .repeat(b * v)
            .as_slice(),
            &device,
        )
        .reshape([b as i32, v as i32, 3, 3]);

        let out = model.forward(image, Some(intrinsics));
        let [bv, n, c] = out.hidden.shape().dims::<3>();
        assert_eq!(bv, b * v);
        assert_eq!(n, 4 + 5); // 2x2 patch tokens + 5 register tokens
        assert_eq!(c, 128); // concat of last two 64-d layers
        assert_eq!(out.patch_start_idx, 5);
    }

    #[test]
    fn croco_style_backbone_matches_manual_forward_composition() {
        let device = <TestBackend as Backend>::Device::default();
        let cfg = CrocoStyleBackboneConfig::new()
            .with_encoder(
                burn_dino::model::dino::DinoVisionTransformerConfig::vits(Some(28), Some(14))
                    .without_register_tokens(),
            )
            .with_decoder_embed_dim(64)
            .with_decoder_heads(4)
            .with_decoder_depth(2)
            .with_register_token_count(5)
            .with_use_intrinsics_embedding(true);
        let model = CrocoStyleBackbone::<TestBackend>::new(&device, cfg);

        let b = 1usize;
        let v = 2usize;
        let h = 28usize;
        let w = 28usize;
        let image = Tensor::<TestBackend, 1>::from_floats(
            (0..b * v * 3 * h * w)
                .map(|idx| ((idx % 53) as f32 - 26.0) / 26.0)
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        )
        .reshape([b as i32, v as i32, 3, h as i32, w as i32]);
        let intrinsics = Tensor::<TestBackend, 1>::from_floats(
            [
                1.0f32, 0.0, 0.5, //
                0.0, 1.0, 0.5, //
                0.0, 0.0, 1.0, //
            ]
            .repeat(b * v)
            .as_slice(),
            &device,
        )
        .reshape([b as i32, v as i32, 3, 3]);

        let out = model.forward(image.clone(), Some(intrinsics.clone()));

        let cfg = &model.cfg.0;
        let [b, v, _, h, w] = image.shape().dims::<5>();
        let bv = b * v;
        let patch_h = h / cfg.encoder.patch_size;
        let patch_w = w / cfg.encoder.patch_size;

        let image_bv = normalize_imagenet(image.reshape([bv as i32, 3, h as i32, w as i32]));
        let encoder_out = model.encoder.forward(image_bv, None);
        let mut hidden = encoder_out.x_norm_patchtokens.clone();

        if let Some(proj) = &model.proj {
            hidden = proj.forward(hidden);
        }
        if let Some(embed) = &model.intrinsics_embed {
            let intr = intrinsic_embedding_image(
                intrinsics,
                h,
                w,
                cfg.intrinsics_embed_degree,
                cfg.encoder.patch_size,
                &hidden.device(),
            );
            hidden = hidden + embed.forward(intr);
        }

        let reg = model
            .register_token
            .val()
            .clone()
            .repeat_dim(0, bv)
            .reshape([
                bv as i32,
                cfg.register_token_count as i32,
                cfg.decoder_embed_dim as i32,
            ]);
        let mut hidden = Tensor::cat(vec![reg, hidden], 1);
        let pos_img =
            crate::model::ops::position_getter(bv, patch_h, patch_w, &hidden.device()) + 1.0;
        let pos_special = Tensor::<TestBackend, 3>::zeros(
            [bv as i32, cfg.register_token_count as i32, 2],
            &hidden.device(),
        );
        let mut pos = Tensor::cat(vec![pos_special, pos_img], 1);

        let mut outputs = Vec::with_capacity(2);
        for (idx, block) in model.blocks.iter().enumerate() {
            let token_count = hidden.shape().dims::<3>()[1];
            if cfg.alternating_local_global && idx % 2 == 1 {
                hidden = hidden.reshape([
                    b as i32,
                    (v * token_count) as i32,
                    cfg.decoder_embed_dim as i32,
                ]);
                pos = pos.reshape([b as i32, (v * token_count) as i32, 2]);
                hidden = block.forward(hidden, Some(&pos), None);
                hidden =
                    hidden.reshape([bv as i32, token_count as i32, cfg.decoder_embed_dim as i32]);
                pos = pos.reshape([bv as i32, token_count as i32, 2]);
            } else {
                hidden = block.forward(hidden, Some(&pos), None);
            }
            if idx + 2 >= model.blocks.len() {
                outputs.push(hidden.clone());
            }
        }

        let expected_hidden = if outputs.len() == 2 {
            Tensor::cat(vec![outputs[0].clone(), outputs[1].clone()], 2)
        } else {
            let last = hidden.clone();
            Tensor::cat(vec![last.clone(), last], 2)
        };

        assert!(max_abs_diff(out.hidden, expected_hidden) < 1e-6);
        assert!(max_abs_diff(out.pos, pos) < 1e-6);
        assert!(
            max_abs_diff(
                out.encoder.x_norm_patchtokens,
                encoder_out.x_norm_patchtokens
            ) < 1e-6
        );
        assert_eq!(out.patch_start_idx, cfg.register_token_count);
    }

    #[test]
    fn forward_with_normalized_intrinsics_matches_explicit_intrinsics() {
        let device = <TestBackend as Backend>::Device::default();
        let cfg = CrocoStyleBackboneConfig::new()
            .with_encoder(
                burn_dino::model::dino::DinoVisionTransformerConfig::vits(Some(28), Some(14))
                    .without_register_tokens(),
            )
            .with_decoder_embed_dim(64)
            .with_decoder_heads(4)
            .with_decoder_depth(2)
            .with_register_token_count(5)
            .with_use_intrinsics_embedding(true)
            .with_intrinsics_embed_degree(4);
        let model = CrocoStyleBackbone::<TestBackend>::new(&device, cfg);

        let b = 1usize;
        let v = 2usize;
        let h = 28usize;
        let w = 28usize;
        let image = Tensor::<TestBackend, 1>::from_floats(
            (0..b * v * 3 * h * w)
                .map(|idx| ((idx % 37) as f32 - 18.0) / 18.0)
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        )
        .reshape([b as i32, v as i32, 3, h as i32, w as i32]);
        let intrinsics = Tensor::<TestBackend, 1>::from_floats(
            [
                1.0f32, 0.0, 0.5, //
                0.0, 1.0, 0.5, //
                0.0, 0.0, 1.0, //
            ]
            .repeat(b * v)
            .as_slice(),
            &device,
        )
        .reshape([b as i32, v as i32, 3, 3]);

        let out_explicit = model.forward(image.clone(), Some(intrinsics));
        let out_auto = model.forward_with_normalized_intrinsics(image);

        assert!(max_abs_diff(out_auto.hidden, out_explicit.hidden) < 1e-6);
        assert!(max_abs_diff(out_auto.pos, out_explicit.pos) < 1e-6);
        assert_eq!(out_auto.patch_start_idx, out_explicit.patch_start_idx);
    }
}
