#![recursion_limit = "512"]

use std::path::Path;

use burn::prelude::*;
use burn_gaussian_splatting::{
    burn_yono::full_backbone_config,
    correctness::{compute_stats, load_safetensors, read_tensor, tensor_to_vec},
    import::load_yono_backbone_from_safetensors,
    model::ops::position_getter,
};

type BackendImpl = burn_gaussian_splatting::backend::BackendImpl;

const DEBUG_REFERENCE_PATH: &str = "tmp/cli_test/python_multiview_debug.safetensors";
const BACKBONE_WEIGHTS: &str = "assets/models/yono_backbone_weights.safetensors";

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

fn normalize_imagenet<B: Backend>(image: Tensor<B, 4>) -> Tensor<B, 4> {
    let mean = Tensor::<B, 1>::from_floats([0.485f32, 0.456, 0.406].as_slice(), &image.device())
        .reshape([1, 3, 1, 1]);
    let std = Tensor::<B, 1>::from_floats([0.229f32, 0.224, 0.225].as_slice(), &image.device())
        .reshape([1, 3, 1, 1]);
    (image - mean) / std
}

fn report(name: &str, actual: Vec<f32>, expected: Vec<f32>) {
    let stats = compute_stats(&actual, &expected).expect("stats should compute");
    println!(
        "{name}: mean_abs={:.6} max_abs={:.6} max_rel={:.6} mse={:.6}",
        stats.mean_abs, stats.max_abs, stats.max_rel, stats.mse
    );
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !Path::new(DEBUG_REFERENCE_PATH).exists() {
        return Err(format!("reference missing at {DEBUG_REFERENCE_PATH}").into());
    }
    if !Path::new(BACKBONE_WEIGHTS).exists() {
        return Err(format!("backbone weights missing at {BACKBONE_WEIGHTS}").into());
    }

    let device = <BackendImpl as Backend>::Device::default();
    let tensors = load_safetensors(DEBUG_REFERENCE_PATH)?;

    let (image_v, image_s) = read_tensor(&tensors, "image")?;
    let (intrinsics_v, intrinsics_s) = read_tensor(&tensors, "intrinsics")?;
    let (imgs_norm_ref_v, imgs_norm_ref_s) = read_tensor(&tensors, "imgs_norm")?;
    let (dino_patch_ref_v, dino_patch_ref_s) = read_tensor(&tensors, "dino_x_norm_patchtokens")?;
    let (dino_cls_ref_v, dino_cls_ref_s) = read_tensor(&tensors, "dino_x_norm_clstoken")?;
    let (dino_reg_ref_v, dino_reg_ref_s) = read_tensor(&tensors, "dino_x_norm_regtokens")?;
    let (intr_emb_img_ref_v, _) = read_tensor(&tensors, "intrinsics_emb_img")?;
    let (intr_emb_tokens_ref_v, _) = read_tensor(&tensors, "intrinsics_emb_tokens")?;
    let (hidden_pre_decode_ref_v, _) = read_tensor(&tensors, "hidden_pre_decode")?;
    let (decode_block_34_ref_v, _) = read_tensor(&tensors, "decode_block_34")?;
    let (decode_block_35_ref_v, _) = read_tensor(&tensors, "decode_block_35")?;
    let (backbone_hidden_ref_v, _) = read_tensor(&tensors, "backbone_hidden")?;
    let (backbone_pos_ref_v, _) = read_tensor(&tensors, "backbone_pos")?;

    let image = tensor_from_f32::<5>(
        image_v,
        [image_s[0], image_s[1], image_s[2], image_s[3], image_s[4]],
        &device,
    );
    let intrinsics = tensor_from_f32::<4>(
        intrinsics_v,
        [
            intrinsics_s[0],
            intrinsics_s[1],
            intrinsics_s[2],
            intrinsics_s[3],
        ],
        &device,
    );

    let (backbone, backbone_apply) = load_yono_backbone_from_safetensors::<BackendImpl>(
        &device,
        full_backbone_config(),
        Path::new(BACKBONE_WEIGHTS),
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

    let [b, v, _, h, w] = image.shape().dims::<5>();
    let bv = b * v;
    let image_bv = image.reshape([bv as i32, 3, h as i32, w as i32]);
    let imgs_norm = normalize_imagenet(image_bv);

    report(
        "imgs_norm",
        tensor_to_vec(imgs_norm.clone())?,
        imgs_norm_ref_v.clone(),
    );

    let dino = backbone.encoder.forward(imgs_norm, None);
    report(
        "dino_x_norm_patchtokens",
        tensor_to_vec(dino.x_norm_patchtokens.clone())?,
        dino_patch_ref_v.clone(),
    );
    report(
        "dino_x_norm_clstoken",
        tensor_to_vec(dino.x_norm_clstoken.clone())?,
        dino_cls_ref_v.clone(),
    );
    let reg = dino
        .x_norm_regtokens
        .expect("dino backbone should emit register tokens");
    report(
        "dino_x_norm_regtokens",
        tensor_to_vec(reg)?,
        dino_reg_ref_v.clone(),
    );

    let intr_emb_img = intrinsic_embedding_image_deg4(intrinsics.clone(), h, w, &device);
    report(
        "intrinsics_emb_img",
        tensor_to_vec(intr_emb_img.clone())?,
        intr_emb_img_ref_v,
    );

    let intr_embed = backbone
        .intrinsics_embed
        .as_ref()
        .expect("full config should include intrinsics embed layer")
        .forward(intr_emb_img);
    report(
        "intrinsics_emb_tokens",
        tensor_to_vec(intr_embed.clone())?,
        intr_emb_tokens_ref_v,
    );

    let mut hidden_pre_decode = dino.x_norm_patchtokens.clone();
    if let Some(proj) = &backbone.proj {
        hidden_pre_decode = proj.forward(hidden_pre_decode);
    }
    hidden_pre_decode = hidden_pre_decode + intr_embed;
    report(
        "hidden_pre_decode",
        tensor_to_vec(hidden_pre_decode.clone())?,
        hidden_pre_decode_ref_v,
    );

    let cfg = full_backbone_config();
    let patch_h = h / cfg.encoder.patch_size;
    let patch_w = w / cfg.encoder.patch_size;

    let reg_tokens = backbone
        .register_token
        .val()
        .clone()
        .repeat_dim(0, bv)
        .reshape([
            bv as i32,
            cfg.register_token_count as i32,
            cfg.decoder_embed_dim as i32,
        ]);
    let mut hidden = Tensor::cat(vec![reg_tokens, hidden_pre_decode], 1);

    let pos_img = position_getter(bv, patch_h, patch_w, &device) + 1.0;
    let pos_special =
        Tensor::<BackendImpl, 3>::zeros([bv as i32, cfg.register_token_count as i32, 2], &device);
    let mut pos = Tensor::cat(vec![pos_special, pos_img], 1);

    let mut out_34 = None;
    let mut out_35 = None;
    let mut out_pos = None;
    for (idx, block) in backbone.blocks.iter().enumerate() {
        let token_count = hidden.shape().dims::<3>()[1];
        if cfg.alternating_local_global && idx % 2 == 1 {
            hidden = hidden.reshape([
                b as i32,
                (v * token_count) as i32,
                cfg.decoder_embed_dim as i32,
            ]);
            pos = pos.reshape([b as i32, (v * token_count) as i32, 2]);
            hidden = block.forward(hidden, Some(&pos), None);
            hidden = hidden.reshape([bv as i32, token_count as i32, cfg.decoder_embed_dim as i32]);
            pos = pos.reshape([bv as i32, token_count as i32, 2]);
        } else {
            hidden = block.forward(hidden, Some(&pos), None);
        }

        if idx == 34 {
            out_34 = Some(hidden.clone());
        } else if idx == 35 {
            out_35 = Some(hidden.clone());
            out_pos = Some(pos.clone());
        }
    }
    let out_34 = out_34.expect("expected block 34 output");
    let out_35 = out_35.expect("expected block 35 output");
    report(
        "decode_block_34",
        tensor_to_vec(out_34.clone())?,
        decode_block_34_ref_v,
    );
    report(
        "decode_block_35",
        tensor_to_vec(out_35.clone())?,
        decode_block_35_ref_v,
    );

    let backbone_hidden = Tensor::cat(vec![out_34, out_35], 2);
    report(
        "backbone_hidden_recomposed",
        tensor_to_vec(backbone_hidden.clone())?,
        backbone_hidden_ref_v,
    );
    report(
        "backbone_pos_recomposed",
        tensor_to_vec(out_pos.expect("pos should be captured"))?,
        backbone_pos_ref_v,
    );

    assert_eq!(
        dino_patch_ref_s,
        vec![bv, dino_patch_ref_s[1], dino_patch_ref_s[2]]
    );
    assert_eq!(dino_cls_ref_s, vec![bv, dino_cls_ref_s[1]]);
    assert_eq!(
        dino_reg_ref_s,
        vec![bv, dino_reg_ref_s[1], dino_reg_ref_s[2]]
    );
    assert_eq!(
        imgs_norm_ref_s,
        vec![image_s[0], image_s[1], image_s[2], image_s[3], image_s[4]]
    );

    Ok(())
}
