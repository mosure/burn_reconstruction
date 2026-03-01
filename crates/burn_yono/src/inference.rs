use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use burn::prelude::*;
use burn_dino::model::dino::DinoVisionTransformerConfig;
#[cfg(feature = "io")]
use image::imageops::FilterType;
#[cfg(feature = "io")]
use image::DynamicImage;

#[cfg(feature = "import")]
use crate::import::{
    load_yono_backbone_from_burnpack_candidates_with_progress,
    load_yono_backbone_from_burnpack_part_bytes_with_progress, load_yono_backbone_from_safetensors,
    load_yono_head_from_burnpack_candidates_with_progress,
    load_yono_head_from_burnpack_part_bytes_with_progress, load_yono_head_from_safetensors,
};
use crate::model::{
    CrocoStyleBackbone, CrocoStyleBackboneConfig, TransformerDecoderSpec, YonoHeadConfig,
    YonoHeadInput, YonoHeadOutput, YonoHeadPipeline,
};
#[cfg(feature = "import")]
use crate::parts::{burnpack_parts_manifest_path, manifest_is_complete};

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum YonoWeightFormat {
    #[default]
    Safetensors,
    Burnpack,
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum YonoWeightPrecision {
    F32,
    #[default]
    F16,
}

#[derive(Debug, Clone)]
pub struct YonoWeights {
    pub backbone: PathBuf,
    pub head: PathBuf,
    pub format: YonoWeightFormat,
    pub precision: YonoWeightPrecision,
}

impl YonoWeights {
    pub fn new(backbone: impl Into<PathBuf>, head: impl Into<PathBuf>) -> Self {
        Self {
            backbone: backbone.into(),
            head: head.into(),
            format: YonoWeightFormat::Safetensors,
            precision: YonoWeightPrecision::default(),
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
            precision: YonoWeightPrecision::default(),
        }
    }

    pub fn burnpack_with_precision(
        backbone: impl Into<PathBuf>,
        head: impl Into<PathBuf>,
        precision: YonoWeightPrecision,
    ) -> Self {
        Self::burnpack(backbone, head).with_precision(precision)
    }

    pub fn with_format(mut self, format: YonoWeightFormat) -> Self {
        self.format = format;
        self
    }

    pub fn with_precision(mut self, precision: YonoWeightPrecision) -> Self {
        self.precision = precision;
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
    #[error(
        "backbone weights not found (expected .bpk or complete .bpk.parts.json bundle) at {0}"
    )]
    MissingBackboneWeights(String),
    #[error("head weights not found (expected .bpk or complete .bpk.parts.json bundle) at {0}")]
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
    #[cfg(feature = "io")]
    #[error("failed to decode in-memory image `{name}`: {source}")]
    ImageDecode {
        name: String,
        source: image::ImageError,
    },
}

impl<B: Backend> YonoModelBundle<B> {
    #[cfg(feature = "import")]
    pub fn load_from_weights(
        device: &B::Device,
        weights: &YonoWeights,
    ) -> Result<(Self, YonoLoadReport), YonoPipelineError> {
        Self::load_from_weights_with_progress(device, weights, |_| {})
    }

    #[cfg(feature = "import")]
    pub fn load_from_weights_with_progress<F>(
        device: &B::Device,
        weights: &YonoWeights,
        progress: F,
    ) -> Result<(Self, YonoLoadReport), YonoPipelineError>
    where
        F: Fn(String),
    {
        match weights.format {
            YonoWeightFormat::Safetensors => {
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
            }
            YonoWeightFormat::Burnpack => {
                let backbone_candidates =
                    burnpack_precision_candidates(weights.backbone.as_path(), weights.precision);
                let head_candidates =
                    burnpack_precision_candidates(weights.head.as_path(), weights.precision);
                if !burnpack_or_parts_available_any(backbone_candidates.as_slice()) {
                    return Err(YonoPipelineError::MissingBackboneWeights(
                        format_candidates(backbone_candidates.as_slice()),
                    ));
                }
                if !burnpack_or_parts_available_any(head_candidates.as_slice()) {
                    return Err(YonoPipelineError::MissingHeadWeights(format_candidates(
                        head_candidates.as_slice(),
                    )));
                }
            }
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
                let backbone_candidates =
                    burnpack_precision_candidates(weights.backbone.as_path(), weights.precision);
                let head_candidates =
                    burnpack_precision_candidates(weights.head.as_path(), weights.precision);
                let (backbone, backbone_apply) =
                    load_yono_backbone_from_burnpack_candidates_with_progress::<B, _>(
                        device,
                        full_backbone_config(),
                        backbone_candidates.as_slice(),
                        &progress,
                    )?;
                let (head, head_apply) =
                    load_yono_head_from_burnpack_candidates_with_progress::<B, _>(
                        device,
                        full_head_config(),
                        head_candidates.as_slice(),
                        &progress,
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
            precision: weights.precision,
        };
        Self::load_from_weights(device, &safetensor_weights)
    }

    #[cfg(feature = "import")]
    pub fn load_from_burnpack_part_bytes(
        device: &B::Device,
        backbone_parts: &[Vec<u8>],
        head_parts: &[Vec<u8>],
    ) -> Result<(Self, YonoLoadReport), YonoPipelineError> {
        Self::load_from_burnpack_part_bytes_with_progress(
            device,
            backbone_parts,
            head_parts,
            |_| {},
        )
    }

    #[cfg(feature = "import")]
    pub fn load_from_burnpack_part_bytes_with_progress<F>(
        device: &B::Device,
        backbone_parts: &[Vec<u8>],
        head_parts: &[Vec<u8>],
        progress: F,
    ) -> Result<(Self, YonoLoadReport), YonoPipelineError>
    where
        F: Fn(String),
    {
        let (backbone, backbone_apply) =
            load_yono_backbone_from_burnpack_part_bytes_with_progress::<B, _>(
                device,
                full_backbone_config(),
                backbone_parts,
                &progress,
            )?;
        let (head, head_apply) = load_yono_head_from_burnpack_part_bytes_with_progress::<B, _>(
            device,
            full_head_config(),
            head_parts,
            &progress,
        )?;

        let report = YonoLoadReport {
            backbone: ApplySummary::from_apply_result(&backbone_apply),
            head: ApplySummary::from_apply_result(&head_apply),
        };

        Ok((Self { backbone, head }, report))
    }

    pub fn forward_from_tensors(
        &self,
        image: Tensor<B, 5>,
        intrinsics: Tensor<B, 4>,
    ) -> YonoHeadOutput<B> {
        self.forward_from_tensors_with_sync(image, intrinsics, false)
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

    fn forward_from_tensors_with_sync(
        &self,
        image: Tensor<B, 5>,
        intrinsics: Tensor<B, 4>,
        synchronize: bool,
    ) -> YonoHeadOutput<B> {
        let backbone_out = self.backbone.forward(image.clone(), Some(intrinsics));
        if synchronize {
            sync_tensor_3d(backbone_out.hidden.clone());
        }

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
        output
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
    fn forward_from_image_tensor_with_sync(
        &self,
        image: Tensor<B, 5>,
        synchronize: bool,
    ) -> YonoHeadOutput<B> {
        let backbone_out = self
            .backbone
            .forward_with_normalized_intrinsics(image.clone());
        if synchronize {
            sync_tensor_3d(backbone_out.hidden.clone());
        }

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
        output
    }

    #[cfg(feature = "io")]
    pub fn forward_from_image_paths(
        &self,
        image_paths: &[PathBuf],
        image_size: usize,
        device: &B::Device,
    ) -> Result<YonoHeadOutput<B>, YonoPipelineError> {
        let image = load_multi_image_tensor_from_paths::<B>(image_paths, image_size, device)?;
        Ok(self.forward_from_image_tensor_with_sync(image, false))
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
        let image = load_multi_image_tensor_from_paths::<B>(image_paths, image_size, device)?;
        let image_load = load_start.elapsed();

        let (output, mut timings) =
            self.forward_from_image_tensor_timed_with_sync(image, synchronize);
        timings.image_load = image_load;
        timings.total = total_start.elapsed();

        Ok((output, timings))
    }

    #[cfg(feature = "io")]
    pub fn forward_from_image_bytes(
        &self,
        named_images: &[(&str, &[u8])],
        image_size: usize,
        device: &B::Device,
    ) -> Result<YonoHeadOutput<B>, YonoPipelineError> {
        let image = load_multi_image_tensor_from_bytes::<B>(named_images, image_size, device)?;
        Ok(self.forward_from_image_tensor_with_sync(image, false))
    }

    #[cfg(feature = "io")]
    pub fn forward_from_image_bytes_timed(
        &self,
        named_images: &[(&str, &[u8])],
        image_size: usize,
        device: &B::Device,
    ) -> Result<(YonoHeadOutput<B>, ForwardTimings), YonoPipelineError> {
        self.forward_from_image_bytes_timed_with_sync(named_images, image_size, device, false)
    }

    #[cfg(feature = "io")]
    pub fn forward_from_image_bytes_timed_with_sync(
        &self,
        named_images: &[(&str, &[u8])],
        image_size: usize,
        device: &B::Device,
        synchronize: bool,
    ) -> Result<(YonoHeadOutput<B>, ForwardTimings), YonoPipelineError> {
        let total_start = Instant::now();

        let load_start = Instant::now();
        let image = load_multi_image_tensor_from_bytes::<B>(named_images, image_size, device)?;
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

#[cfg(feature = "import")]
fn burnpack_or_parts_available(path: &Path) -> bool {
    if path.exists() {
        return true;
    }
    let manifest = burnpack_parts_manifest_path(path);
    manifest_is_complete(manifest.as_path()).unwrap_or(false)
}

#[cfg(feature = "import")]
fn burnpack_or_parts_available_any(candidates: &[PathBuf]) -> bool {
    candidates
        .iter()
        .any(|candidate| burnpack_or_parts_available(candidate.as_path()))
}

#[cfg(feature = "import")]
fn format_candidates(candidates: &[PathBuf]) -> String {
    candidates
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn burnpack_path_for_precision(path: &Path, precision: YonoWeightPrecision) -> PathBuf {
    let path = if path
        .extension()
        .map(|ext| ext.eq_ignore_ascii_case("bpk"))
        .unwrap_or(false)
    {
        path.to_path_buf()
    } else {
        path.with_extension("bpk")
    };

    match precision {
        YonoWeightPrecision::F16 => with_file_stem_suffix(path.as_path(), "_f16"),
        YonoWeightPrecision::F32 => strip_file_stem_suffix(path.as_path(), "_f16"),
    }
}

pub fn burnpack_precision_candidates(path: &Path, preferred: YonoWeightPrecision) -> Vec<PathBuf> {
    let preferred_path = burnpack_path_for_precision(path, preferred);
    let fallback_precision = match preferred {
        YonoWeightPrecision::F16 => YonoWeightPrecision::F32,
        YonoWeightPrecision::F32 => YonoWeightPrecision::F16,
    };
    let fallback_path = burnpack_path_for_precision(path, fallback_precision);

    if fallback_path == preferred_path {
        vec![preferred_path]
    } else {
        vec![preferred_path, fallback_path]
    }
}

fn with_file_stem_suffix(path: &Path, suffix: &str) -> PathBuf {
    let Some(stem) = path.file_stem() else {
        return path.to_path_buf();
    };
    let stem = stem.to_string_lossy();
    if stem.ends_with(suffix) {
        return path.to_path_buf();
    }

    let ext = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
    let mut file_name = format!("{stem}{suffix}");
    if !ext.is_empty() {
        file_name.push('.');
        file_name.push_str(ext);
    }
    path.with_file_name(file_name)
}

fn strip_file_stem_suffix(path: &Path, suffix: &str) -> PathBuf {
    let Some(stem) = path.file_stem() else {
        return path.to_path_buf();
    };
    let stem = stem.to_string_lossy();
    let Some(stripped) = stem.strip_suffix(suffix) else {
        return path.to_path_buf();
    };

    let ext = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
    let mut file_name = stripped.to_string();
    if !ext.is_empty() {
        file_name.push('.');
        file_name.push_str(ext);
    }
    path.with_file_name(file_name)
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
    load_multi_image_tensor_from_paths(paths, image_size, device)
}

#[cfg(feature = "io")]
pub fn load_multi_image_tensor_from_paths<B: Backend>(
    paths: &[PathBuf],
    image_size: usize,
    device: &B::Device,
) -> Result<Tensor<B, 5>, YonoPipelineError> {
    validate_multi_image_inputs(paths.len(), image_size)?;

    let mut images = Vec::with_capacity(paths.len());
    for path in paths {
        let image = image::open(path).map_err(|source| YonoPipelineError::ImageLoad {
            path: path.display().to_string(),
            source,
        })?;
        images.push(image);
    }

    build_multi_image_tensor_from_dynamic_images::<B>(images.as_slice(), image_size, device)
}

#[cfg(feature = "io")]
pub fn load_multi_image_tensor_from_bytes<B: Backend>(
    named_images: &[(&str, &[u8])],
    image_size: usize,
    device: &B::Device,
) -> Result<Tensor<B, 5>, YonoPipelineError> {
    validate_multi_image_inputs(named_images.len(), image_size)?;

    let mut images = Vec::with_capacity(named_images.len());
    for (name, bytes) in named_images {
        let image =
            image::load_from_memory(bytes).map_err(|source| YonoPipelineError::ImageDecode {
                name: (*name).to_string(),
                source,
            })?;
        images.push(image);
    }

    build_multi_image_tensor_from_dynamic_images::<B>(images.as_slice(), image_size, device)
}

#[cfg(feature = "io")]
fn validate_multi_image_inputs(
    image_count: usize,
    image_size: usize,
) -> Result<(), YonoPipelineError> {
    if image_count < 2 {
        return Err(YonoPipelineError::NotEnoughViews(image_count));
    }
    if !image_size.is_multiple_of(14) {
        return Err(YonoPipelineError::InvalidImageSize(image_size));
    }
    Ok(())
}

#[cfg(feature = "io")]
fn build_multi_image_tensor_from_dynamic_images<B: Backend>(
    images: &[DynamicImage],
    image_size: usize,
    device: &B::Device,
) -> Result<Tensor<B, 5>, YonoPipelineError> {
    let v = images.len();
    let hw = image_size * image_size;
    // One contiguous host buffer avoids per-view temporary CHW allocations when
    // scaling to many views.
    let mut data = vec![0.0f32; v * 3 * hw];

    for (image_idx, image) in images.iter().enumerate() {
        let cropped = center_crop_to_aspect(image, image_size as u32, image_size as u32);
        let resized = cropped
            .resize_exact(image_size as u32, image_size as u32, FilterType::Triangle)
            .to_rgb8();

        let base = image_idx * 3 * hw;
        for y in 0..image_size {
            for x in 0..image_size {
                let pixel = resized.get_pixel(x as u32, y as u32).0;
                let off = y * image_size + x;
                data[base + off] = pixel[0] as f32 / 255.0;
                data[base + hw + off] = pixel[1] as f32 / 255.0;
                data[base + 2 * hw + off] = pixel[2] as f32 / 255.0;
            }
        }
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

#[cfg(feature = "io")]
fn center_crop_to_aspect(
    image: &DynamicImage,
    target_width: u32,
    target_height: u32,
) -> DynamicImage {
    let src_width = image.width();
    let src_height = image.height();

    if src_width == 0 || src_height == 0 || target_width == 0 || target_height == 0 {
        return image.clone();
    }

    let src_ratio_num = src_width as u64 * target_height as u64;
    let src_ratio_den = target_width as u64 * src_height as u64;

    if src_ratio_num == src_ratio_den {
        return image.clone();
    }

    if src_ratio_num > src_ratio_den {
        let crop_width = (src_height as u64 * target_width as u64 / target_height as u64) as u32;
        let crop_x = (src_width - crop_width) / 2;
        image.crop_imm(crop_x, 0, crop_width, src_height)
    } else {
        let crop_height = (src_width as u64 * target_height as u64 / target_width as u64) as u32;
        let crop_y = (src_height - crop_height) / 2;
        image.crop_imm(0, crop_y, src_width, crop_height)
    }
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

#[cfg(test)]
mod precision_tests {
    use std::path::Path;

    use crate::inference::{
        burnpack_path_for_precision, burnpack_precision_candidates, YonoWeightPrecision,
    };

    #[test]
    fn burnpack_precision_path_roundtrip() {
        let base = Path::new("weights/yono_backbone.bpk");
        let f16 = burnpack_path_for_precision(base, YonoWeightPrecision::F16);
        let f32 = burnpack_path_for_precision(f16.as_path(), YonoWeightPrecision::F32);
        assert_eq!(f16, Path::new("weights/yono_backbone_f16.bpk"));
        assert_eq!(f32, base);
    }

    #[test]
    fn burnpack_precision_candidates_order_matches_preference() {
        let base = Path::new("weights/yono_head.bpk");
        let f16_first = burnpack_precision_candidates(base, YonoWeightPrecision::F16);
        assert_eq!(
            f16_first,
            vec![
                Path::new("weights/yono_head_f16.bpk").to_path_buf(),
                Path::new("weights/yono_head.bpk").to_path_buf()
            ]
        );

        let f32_first = burnpack_precision_candidates(base, YonoWeightPrecision::F32);
        assert_eq!(
            f32_first,
            vec![
                Path::new("weights/yono_head.bpk").to_path_buf(),
                Path::new("weights/yono_head_f16.bpk").to_path_buf()
            ]
        );
    }
}

#[cfg(all(test, feature = "io"))]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use burn::{backend::NdArray, prelude::*};
    use image::{DynamicImage, Rgb, RgbImage};

    use crate::inference::{
        build_multi_image_tensor_from_dynamic_images, load_multi_image_tensor_from_bytes,
        load_multi_image_tensor_from_paths,
    };

    #[test]
    fn bytes_loader_matches_paths_loader() {
        type TestBackend = NdArray<f32>;
        let device = <TestBackend as Backend>::Device::default();
        let image_size = 14usize;

        let image_a = synth_image(image_size, image_size, 11);
        let image_b = synth_image(image_size, image_size, 53);
        let bytes_a = encode_png(&image_a);
        let bytes_b = encode_png(&image_b);

        let test_dir = unique_temp_dir("burn_yono_bytes_loader");
        fs::create_dir_all(&test_dir).expect("failed to create test temp directory");
        let path_a = test_dir.join("view_a.png");
        let path_b = test_dir.join("view_b.png");
        image_a
            .save(&path_a)
            .expect("failed to persist test image a");
        image_b
            .save(&path_b)
            .expect("failed to persist test image b");

        let from_paths = load_multi_image_tensor_from_paths::<TestBackend>(
            [path_a.clone(), path_b.clone()].as_slice(),
            image_size,
            &device,
        )
        .expect("path loader failed");

        let from_bytes = load_multi_image_tensor_from_bytes::<TestBackend>(
            [
                ("view_a.png", bytes_a.as_slice()),
                ("view_b.png", bytes_b.as_slice()),
            ]
            .as_slice(),
            image_size,
            &device,
        )
        .expect("byte loader failed");

        let lhs = from_paths
            .into_data()
            .to_vec::<f32>()
            .expect("path tensor should be readable");
        let rhs = from_bytes
            .into_data()
            .to_vec::<f32>()
            .expect("byte tensor should be readable");
        let max_abs = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs <= 1e-7, "loader mismatch max_abs={max_abs}");

        let _ = fs::remove_file(path_a);
        let _ = fs::remove_file(path_b);
        let _ = fs::remove_dir_all(test_dir);
    }

    #[test]
    fn preprocessing_center_crops_wide_images_before_resize() {
        type TestBackend = NdArray<f32>;
        let device = <TestBackend as Backend>::Device::default();
        let image_size = 2usize;

        let mut image = RgbImage::new(4, 2);
        for y in 0..2 {
            image.put_pixel(0, y, Rgb([0, 0, 0]));
            image.put_pixel(1, y, Rgb([255, 0, 0]));
            image.put_pixel(2, y, Rgb([0, 255, 0]));
            image.put_pixel(3, y, Rgb([0, 0, 255]));
        }
        let tensor = build_multi_image_tensor_from_dynamic_images::<TestBackend>(
            [DynamicImage::ImageRgb8(image)].as_slice(),
            image_size,
            &device,
        )
        .expect("preprocessing should succeed");

        let data = tensor
            .into_data()
            .to_vec::<f32>()
            .expect("tensor should be readable");
        let expected = vec![
            1.0, 0.0, 1.0, 0.0, // R channel
            0.0, 1.0, 0.0, 1.0, // G channel
            0.0, 0.0, 0.0, 0.0, // B channel
        ];
        assert_tensor_close(data.as_slice(), expected.as_slice(), 1e-7);
    }

    #[test]
    fn preprocessing_center_crops_tall_images_before_resize() {
        type TestBackend = NdArray<f32>;
        let device = <TestBackend as Backend>::Device::default();
        let image_size = 2usize;

        let mut image = RgbImage::new(2, 4);
        for x in 0..2 {
            image.put_pixel(x, 0, Rgb([0, 0, 255]));
            image.put_pixel(x, 1, Rgb([255, 0, 0]));
            image.put_pixel(x, 2, Rgb([0, 255, 0]));
            image.put_pixel(x, 3, Rgb([255, 255, 255]));
        }
        let tensor = build_multi_image_tensor_from_dynamic_images::<TestBackend>(
            [DynamicImage::ImageRgb8(image)].as_slice(),
            image_size,
            &device,
        )
        .expect("preprocessing should succeed");

        let data = tensor
            .into_data()
            .to_vec::<f32>()
            .expect("tensor should be readable");
        let expected = vec![
            1.0, 1.0, 0.0, 0.0, // R channel
            0.0, 0.0, 1.0, 1.0, // G channel
            0.0, 0.0, 0.0, 0.0, // B channel
        ];
        assert_tensor_close(data.as_slice(), expected.as_slice(), 1e-7);
    }

    fn assert_tensor_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len(), "tensor length mismatch");
        let max_abs = actual
            .iter()
            .zip(expected.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= tol,
            "tensor mismatch max_abs={max_abs}, tol={tol}"
        );
    }

    fn synth_image(width: usize, height: usize, seed: u8) -> DynamicImage {
        let mut image = RgbImage::new(width as u32, height as u32);
        for y in 0..height {
            for x in 0..width {
                let r = ((x * 17 + y * 13 + seed as usize) % 255) as u8;
                let g = ((x * 7 + y * 29 + seed as usize * 3) % 255) as u8;
                let b = ((x * 3 + y * 11 + seed as usize * 5) % 255) as u8;
                image.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
            }
        }
        DynamicImage::ImageRgb8(image)
    }

    fn encode_png(image: &DynamicImage) -> Vec<u8> {
        let mut cursor = std::io::Cursor::new(Vec::new());
        image
            .write_to(&mut cursor, image::ImageFormat::Png)
            .expect("failed to encode png");
        cursor.into_inner()
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}_{}_{}", std::process::id(), stamp))
    }
}
