use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use burn::prelude::*;
use burn_dino::model::dino::DinoVisionTransformerConfig;
#[cfg(feature = "io")]
use image::imageops::FilterType;

#[cfg(feature = "import")]
use crate::import::{
    load_yono_backbone_from_burnpack, load_yono_backbone_from_safetensors,
    load_yono_head_from_burnpack, load_yono_head_from_safetensors,
};
use crate::model::{
    CrocoStyleBackbone, CrocoStyleBackboneConfig, TransformerDecoderSpec, YonoHeadConfig,
    YonoHeadInput, YonoHeadOutput, YonoHeadPipeline,
};

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum YonoWeightFormat {
    #[default]
    Safetensors,
    Burnpack,
}

#[derive(Debug, Clone)]
pub struct YonoWeights {
    pub backbone: PathBuf,
    pub head: PathBuf,
    pub format: YonoWeightFormat,
}

impl YonoWeights {
    pub fn new(backbone: impl Into<PathBuf>, head: impl Into<PathBuf>) -> Self {
        Self {
            backbone: backbone.into(),
            head: head.into(),
            format: YonoWeightFormat::Safetensors,
        }
    }

    pub fn safetensors(backbone: impl Into<PathBuf>, head: impl Into<PathBuf>) -> Self {
        Self::new(backbone, head)
    }

    pub fn burnpack(backbone: impl Into<PathBuf>, head: impl Into<PathBuf>) -> Self {
        Self {
            backbone: backbone.into(),
            head: head.into(),
            format: YonoWeightFormat::Burnpack,
        }
    }

    pub fn with_format(mut self, format: YonoWeightFormat) -> Self {
        self.format = format;
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct ForwardTimings {
    pub image_load: Duration,
    pub backbone: Duration,
    pub head: Duration,
    pub total: Duration,
}

#[derive(Debug, Clone)]
pub struct ApplySummary {
    pub applied: usize,
    pub missing: Vec<String>,
    pub unused: Vec<String>,
    pub skipped: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct YonoLoadReport {
    pub backbone: ApplySummary,
    pub head: ApplySummary,
}

#[derive(Module, Debug)]
pub struct YonoModelBundle<B: Backend> {
    pub backbone: CrocoStyleBackbone<B>,
    pub head: YonoHeadPipeline<B>,
}

#[derive(Debug, thiserror::Error)]
pub enum YonoPipelineError {
    #[error("image-size must be divisible by 14, got {0}")]
    InvalidImageSize(usize),
    #[error("expected at least two input images, got {0}")]
    NotEnoughViews(usize),
    #[error("backbone weights not found at {0}")]
    MissingBackboneWeights(String),
    #[error("head weights not found at {0}")]
    MissingHeadWeights(String),
    #[cfg(feature = "import")]
    #[error("failed to import model weights: {0}")]
    Import(#[from] crate::import::ImportError),
    #[cfg(feature = "io")]
    #[error("failed to load image `{path}`: {source}")]
    ImageLoad {
        path: String,
        source: image::ImageError,
    },
}

impl<B: Backend> YonoModelBundle<B> {
    #[cfg(feature = "import")]
    pub fn load_from_weights(
        device: &B::Device,
        weights: &YonoWeights,
    ) -> Result<(Self, YonoLoadReport), YonoPipelineError> {
        if !weights.backbone.exists() {
            return Err(YonoPipelineError::MissingBackboneWeights(
                weights.backbone.display().to_string(),
            ));
        }
        if !weights.head.exists() {
            return Err(YonoPipelineError::MissingHeadWeights(
                weights.head.display().to_string(),
            ));
        }

        let (backbone, backbone_apply, head, head_apply) = match weights.format {
            YonoWeightFormat::Safetensors => {
                let (backbone, backbone_apply) = load_yono_backbone_from_safetensors::<B>(
                    device,
                    full_backbone_config(),
                    weights.backbone.as_path(),
                )?;
                let (head, head_apply) = load_yono_head_from_safetensors::<B>(
                    device,
                    full_head_config(),
                    weights.head.as_path(),
                )?;
                (backbone, backbone_apply, head, head_apply)
            }
            YonoWeightFormat::Burnpack => {
                let (backbone, backbone_apply) = load_yono_backbone_from_burnpack::<B>(
                    device,
                    full_backbone_config(),
                    weights.backbone.as_path(),
                )?;
                let (head, head_apply) = load_yono_head_from_burnpack::<B>(
                    device,
                    full_head_config(),
                    weights.head.as_path(),
                )?;
                (backbone, backbone_apply, head, head_apply)
            }
        };

        let report = YonoLoadReport {
            backbone: ApplySummary::from_apply_result(&backbone_apply),
            head: ApplySummary::from_apply_result(&head_apply),
        };

        Ok((Self { backbone, head }, report))
    }

    #[cfg(feature = "import")]
    pub fn load_from_safetensors(
        device: &B::Device,
        weights: &YonoWeights,
    ) -> Result<(Self, YonoLoadReport), YonoPipelineError> {
        let safetensor_weights = YonoWeights {
            backbone: weights.backbone.clone(),
            head: weights.head.clone(),
            format: YonoWeightFormat::Safetensors,
        };
        Self::load_from_weights(device, &safetensor_weights)
    }

    pub fn forward_from_tensors(
        &self,
        image: Tensor<B, 5>,
        intrinsics: Tensor<B, 4>,
    ) -> YonoHeadOutput<B> {
        self.forward_from_tensors_timed_with_sync(image, intrinsics, false)
            .0
    }

    pub fn forward_from_tensors_timed(
        &self,
        image: Tensor<B, 5>,
        intrinsics: Tensor<B, 4>,
    ) -> (YonoHeadOutput<B>, ForwardTimings) {
        self.forward_from_tensors_timed_with_sync(image, intrinsics, true)
    }

    pub fn forward_from_tensors_timed_with_sync(
        &self,
        image: Tensor<B, 5>,
        intrinsics: Tensor<B, 4>,
        synchronize: bool,
    ) -> (YonoHeadOutput<B>, ForwardTimings) {
        let total_start = Instant::now();

        let backbone_start = Instant::now();
        let backbone_out = self.backbone.forward(image.clone(), Some(intrinsics));
        if synchronize {
            sync_tensor_3d(backbone_out.hidden.clone());
        }
        let backbone_duration = backbone_start.elapsed();

        let head_start = Instant::now();
        let output = self.head.forward(YonoHeadInput {
            image,
            hidden: backbone_out.hidden,
            pos: backbone_out.pos,
            hidden_upsampled: None,
            pos_upsampled: None,
            patch_start_idx: backbone_out.patch_start_idx,
            global_step: 0,
            training: false,
            extrinsics: None,
            use_predicted_pose: true,
            scheduled_sampling_draw: None,
        });
        if synchronize {
            sync_tensor_2d(output.gaussians_flat.opacities.clone());
        }
        let head_duration = head_start.elapsed();

        let timings = ForwardTimings {
            image_load: Duration::ZERO,
            backbone: backbone_duration,
            head: head_duration,
            total: total_start.elapsed(),
        };

        (output, timings)
    }

    #[cfg(feature = "io")]
    fn forward_from_image_tensor_timed_with_sync(
        &self,
        image: Tensor<B, 5>,
        synchronize: bool,
    ) -> (YonoHeadOutput<B>, ForwardTimings) {
        let total_start = Instant::now();

        let backbone_start = Instant::now();
        let backbone_out = self
            .backbone
            .forward_with_normalized_intrinsics(image.clone());
        if synchronize {
            sync_tensor_3d(backbone_out.hidden.clone());
        }
        let backbone_duration = backbone_start.elapsed();

        let head_start = Instant::now();
        let output = self.head.forward(YonoHeadInput {
            image,
            hidden: backbone_out.hidden,
            pos: backbone_out.pos,
            hidden_upsampled: None,
            pos_upsampled: None,
            patch_start_idx: backbone_out.patch_start_idx,
            global_step: 0,
            training: false,
            extrinsics: None,
            use_predicted_pose: true,
            scheduled_sampling_draw: None,
        });
        if synchronize {
            sync_tensor_2d(output.gaussians_flat.opacities.clone());
        }
        let head_duration = head_start.elapsed();

        let timings = ForwardTimings {
            image_load: Duration::ZERO,
            backbone: backbone_duration,
            head: head_duration,
            total: total_start.elapsed(),
        };

        (output, timings)
    }

    #[cfg(feature = "io")]
    pub fn forward_from_image_paths(
        &self,
        image_paths: &[PathBuf],
        image_size: usize,
        device: &B::Device,
    ) -> Result<YonoHeadOutput<B>, YonoPipelineError> {
        self.forward_from_image_paths_timed_with_sync(image_paths, image_size, device, false)
            .map(|(output, _)| output)
    }

    #[cfg(feature = "io")]
    pub fn forward_from_image_paths_timed(
        &self,
        image_paths: &[PathBuf],
        image_size: usize,
        device: &B::Device,
    ) -> Result<(YonoHeadOutput<B>, ForwardTimings), YonoPipelineError> {
        self.forward_from_image_paths_timed_with_sync(image_paths, image_size, device, false)
    }

    #[cfg(feature = "io")]
    pub fn forward_from_image_paths_timed_with_sync(
        &self,
        image_paths: &[PathBuf],
        image_size: usize,
        device: &B::Device,
        synchronize: bool,
    ) -> Result<(YonoHeadOutput<B>, ForwardTimings), YonoPipelineError> {
        let total_start = Instant::now();

        let load_start = Instant::now();
        let image = load_multi_image_tensor::<B>(image_paths, image_size, device)?;
        let image_load = load_start.elapsed();

        let (output, mut timings) =
            self.forward_from_image_tensor_timed_with_sync(image, synchronize);
        timings.image_load = image_load;
        timings.total = total_start.elapsed();

        Ok((output, timings))
    }
}

impl ApplySummary {
    #[cfg(feature = "import")]
    fn from_apply_result(result: &burn_store::ApplyResult) -> Self {
        Self {
            applied: result.applied.len(),
            missing: result.missing.clone(),
            unused: result.unused.clone(),
            skipped: result.skipped.clone(),
        }
    }
}

pub fn full_backbone_config() -> CrocoStyleBackboneConfig {
    CrocoStyleBackboneConfig::new()
        .with_encoder(
            DinoVisionTransformerConfig::vitl(Some(224), Some(14)).with_register_tokens(4),
        )
        .with_decoder_embed_dim(1024)
        .with_decoder_heads(16)
        .with_decoder_depth(36)
        .with_decoder_mlp_ratio(4.0)
        .with_rope_frequency(100.0)
        .with_decoder_qk_norm(true)
        .with_register_token_count(5)
        .with_alternating_local_global(true)
        .with_use_intrinsics_embedding(true)
        .with_intrinsics_embed_degree(4)
}

pub fn full_head_config() -> YonoHeadConfig {
    let point_decoder = TransformerDecoderSpec::new()
        .with_in_dim(2048)
        .with_embed_dim(1024)
        .with_out_dim(1024)
        .with_depth(5)
        .with_num_heads(16)
        .with_mlp_ratio(4.0)
        .with_need_project(true)
        .with_qk_norm(false)
        .with_rope_frequency(100.0);

    let camera_decoder = TransformerDecoderSpec::new()
        .with_in_dim(2048)
        .with_embed_dim(1024)
        .with_out_dim(512)
        .with_depth(5)
        .with_num_heads(16)
        .with_mlp_ratio(4.0)
        .with_need_project(true)
        .with_qk_norm(false)
        .with_rope_frequency(100.0);

    YonoHeadConfig::new()
        .with_patch_size(14)
        .with_gaussians_per_axis(14)
        .with_upscale_token_ratio(2)
        .with_gaussian_downsample_ratio(1)
        .with_num_surfaces(1)
        .with_pose_free(true)
        .with_point_decoder(point_decoder.clone())
        .with_gaussian_decoder(point_decoder)
        .with_camera_decoder(camera_decoder)
        .with_share_point_decoder_init(false)
}

#[cfg(feature = "io")]
pub fn load_multi_image_tensor<B: Backend>(
    paths: &[PathBuf],
    image_size: usize,
    device: &B::Device,
) -> Result<Tensor<B, 5>, YonoPipelineError> {
    if paths.len() < 2 {
        return Err(YonoPipelineError::NotEnoughViews(paths.len()));
    }
    if !image_size.is_multiple_of(14) {
        return Err(YonoPipelineError::InvalidImageSize(image_size));
    }

    let v = paths.len();
    let hw = image_size * image_size;
    let mut data = Vec::with_capacity(v * 3 * hw);

    for path in paths {
        let image = image::open(path).map_err(|source| YonoPipelineError::ImageLoad {
            path: path.display().to_string(),
            source,
        })?;

        let resized = image
            .resize_exact(image_size as u32, image_size as u32, FilterType::Triangle)
            .to_rgb8();

        let mut chw = vec![0.0f32; 3 * hw];
        for y in 0..image_size {
            for x in 0..image_size {
                let pixel = resized.get_pixel(x as u32, y as u32).0;
                let off = y * image_size + x;
                chw[off] = pixel[0] as f32 / 255.0;
                chw[hw + off] = pixel[1] as f32 / 255.0;
                chw[2 * hw + off] = pixel[2] as f32 / 255.0;
            }
        }
        data.extend_from_slice(chw.as_slice());
    }

    Ok(
        Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([
            1,
            v as i32,
            3,
            image_size as i32,
            image_size as i32,
        ]),
    )
}

pub fn normalized_intrinsics<B: Backend>(device: &B::Device, views: usize) -> Tensor<B, 4> {
    Tensor::<B, 1>::from_floats(
        [
            1.0f32, 0.0, 0.5, //
            0.0, 1.0, 0.5, //
            0.0, 0.0, 1.0, //
        ]
        .repeat(views)
        .as_slice(),
        device,
    )
    .reshape([1, views as i32, 3, 3])
}

fn sync_tensor_3d<B: Backend>(tensor: Tensor<B, 3>) {
    let [batch, tokens, channels] = tensor.shape().dims::<3>();
    if batch == 0 || tokens == 0 || channels == 0 {
        return;
    }
    let _ = tensor.slice([0..1, 0..1, 0..1]).into_data().to_vec::<f32>();
}

fn sync_tensor_2d<B: Backend>(tensor: Tensor<B, 2>) {
    let [batch, elems] = tensor.shape().dims::<2>();
    if batch == 0 || elems == 0 {
        return;
    }
    let _ = tensor.slice([0..1, 0..1]).into_data().to_vec::<f32>();
}
