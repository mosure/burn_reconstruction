use std::path::{Path, PathBuf};
#[cfg(target_arch = "wasm32")]
use std::time::Duration;

use crate::backend::{default_device, BackendDevice, BackendImpl};
use crate::bootstrap::{
    resolve_or_bootstrap_yono_weights_with_config,
    resolve_or_bootstrap_yono_weights_with_precision,
    resolve_or_bootstrap_yono_weights_with_precision_and_progress,
    resolve_or_bootstrap_zipsplat_weights_with_config,
    resolve_or_bootstrap_zipsplat_weights_with_precision,
    resolve_or_bootstrap_zipsplat_weights_with_precision_and_progress, ModelBootstrapError,
    YonoBootstrapConfig, ZipSplatBootstrapConfig,
};
use burn::tensor::Tensor;
use burn_yono::{
    glb::{GlbExportOptions, GlbSortMode},
    inference::{ApplySummary, ForwardTimings, YonoModelBundle, YonoPipelineError},
    model::gaussian::FlatGaussians,
    YonoWeightFormat, YonoWeightPrecision, YonoWeights,
};
use burn_zipsplat::{
    ZipSplatCompression, ZipSplatConfig, ZipSplatForwardTimings, ZipSplatModelBundle,
    ZipSplatPipelineError, ZipSplatWeightFormat, ZipSplatWeightPrecision, ZipSplatWeights,
};

/// Model families supported by the outer multi-view pipeline.
///
/// Kept as an enum so additional multi-view -> 3DGS methods can be added
/// without changing call sites.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipelineModel {
    Yono,
    ZipSplat,
}

impl PipelineModel {
    pub fn id(self) -> &'static str {
        match self {
            Self::Yono => "yono",
            Self::ZipSplat => burn_zipsplat::MODEL_ID,
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Yono => "YoNoSplat",
            Self::ZipSplat => burn_zipsplat::MODEL_DISPLAY_NAME,
        }
    }

    pub fn default_remote_root(self) -> &'static str {
        match self {
            Self::Yono => "yono",
            Self::ZipSplat => burn_zipsplat::DEFAULT_REMOTE_ROOT,
        }
    }

    pub fn capabilities(self) -> PipelineModelCapabilities {
        match self {
            Self::Yono => PipelineModelCapabilities {
                model: self,
                display_name: self.display_name(),
                min_views: 2,
                max_views: None,
                default_image_size: 224,
                image_size_multiple: 14,
                quality: PipelineQualityControl::ExportOnly,
            },
            Self::ZipSplat => PipelineModelCapabilities {
                model: self,
                display_name: self.display_name(),
                min_views: burn_zipsplat::MIN_VIEWS,
                max_views: Some(burn_zipsplat::MAX_VIEWS),
                default_image_size: burn_zipsplat::DEFAULT_IMAGE_SIZE,
                image_size_multiple: burn_zipsplat::IMAGE_SIZE_MULTIPLE,
                quality: PipelineQualityControl::ZipSplatCompression {
                    min_r: burn_zipsplat::MIN_COMPRESSION_R,
                    max_r: burn_zipsplat::MAX_COMPRESSION_R,
                    presets: &ZIP_SPLAT_QUALITY_PRESETS,
                },
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PipelineModelCapabilities {
    pub model: PipelineModel,
    pub display_name: &'static str,
    pub min_views: usize,
    pub max_views: Option<usize>,
    pub default_image_size: usize,
    pub image_size_multiple: usize,
    pub quality: PipelineQualityControl,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipelineQualityControl {
    ExportOnly,
    ZipSplatCompression {
        min_r: usize,
        max_r: usize,
        presets: &'static [ZipSplatQualityLevel],
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ZipSplatQualityLevel {
    pub name: &'static str,
    pub r: usize,
}

pub const ZIP_SPLAT_QUALITY_PRESETS: [ZipSplatQualityLevel; 4] = [
    ZipSplatQualityLevel { name: "full", r: 1 },
    ZipSplatQualityLevel {
        name: "balanced",
        r: 2,
    },
    ZipSplatQualityLevel {
        name: "compact",
        r: 4,
    },
    ZipSplatQualityLevel {
        name: "preview",
        r: 8,
    },
];

impl std::str::FromStr for PipelineModel {
    type Err = PipelineError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "yono" | "yonosplat" => Ok(Self::Yono),
            "zipsplat" | "zip" => Ok(Self::ZipSplat),
            other => Err(PipelineError::InvalidModel(other.to_string())),
        }
    }
}

/// Export/inference quality presets.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PipelineQuality {
    Fast,
    Full,
    #[default]
    Balanced,
    Compact,
    Preview,
    High,
}

impl PipelineQuality {
    /// Maps quality presets to GLB export policy.
    pub fn export_options(self) -> GlbExportOptions {
        match self {
            Self::Fast => GlbExportOptions {
                max_gaussians: 2048,
                opacity_threshold: 0.05,
                sort_mode: GlbSortMode::Opacity,
            },
            Self::Preview => GlbExportOptions {
                max_gaussians: 2048,
                opacity_threshold: 0.05,
                sort_mode: GlbSortMode::Opacity,
            },
            Self::Balanced => GlbExportOptions::default(),
            Self::Compact => GlbExportOptions {
                max_gaussians: 4096,
                opacity_threshold: 0.01,
                sort_mode: GlbSortMode::Opacity,
            },
            Self::Full => GlbExportOptions {
                max_gaussians: 200_000,
                opacity_threshold: 0.0,
                sort_mode: GlbSortMode::Index,
            },
            Self::High => GlbExportOptions {
                max_gaussians: 200_000,
                opacity_threshold: 0.0,
                sort_mode: GlbSortMode::Index,
            },
        }
    }

    pub fn default_zipsplat_r(self) -> usize {
        match self {
            Self::Full | Self::High => ZipSplatCompression::FULL.get(),
            Self::Balanced => ZipSplatCompression::BALANCED.get(),
            Self::Compact => ZipSplatCompression::COMPACT.get(),
            Self::Fast | Self::Preview => ZipSplatCompression::PREVIEW.get(),
        }
    }
}

/// High-level model/runtime settings for image -> gaussian inference.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PipelineConfig {
    pub model: PipelineModel,
    pub quality: PipelineQuality,
    pub image_size: usize,
    pub zipsplat_r: usize,
}

/// Model-load identity for a pipeline.
///
/// Quality/export settings and ZipSplat compression are runtime choices; changing
/// them must not invalidate already-loaded weights.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PipelineLoadKey {
    pub model: PipelineModel,
    pub image_size: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            model: PipelineModel::Yono,
            quality: PipelineQuality::Balanced,
            image_size: 224,
            zipsplat_r: ZipSplatCompression::BALANCED.get(),
        }
    }
}

impl PipelineConfig {
    pub fn load_key(&self) -> PipelineLoadKey {
        PipelineLoadKey {
            model: self.model,
            image_size: self.image_size,
        }
    }

    pub fn effective_zipsplat_r(&self) -> usize {
        ZipSplatCompression::new(self.zipsplat_r).get()
    }

    pub fn with_model(mut self, model: PipelineModel) -> Self {
        self.model = model;
        self
    }

    pub fn with_quality(mut self, quality: PipelineQuality) -> Self {
        self.quality = quality;
        self.zipsplat_r = quality.default_zipsplat_r();
        self
    }

    pub fn with_zipsplat_r(mut self, r: usize) -> Self {
        self.zipsplat_r = ZipSplatCompression::new(r).get();
        self
    }
}

/// Weight configuration for loading the active model family.
#[derive(Debug, Clone)]
pub enum PipelineWeights {
    Yono(YonoWeights),
    ZipSplat(ZipSplatWeights),
}

impl PipelineWeights {
    /// Uses repository default safetensor checkpoint paths.
    pub fn default_yono_safetensors() -> Self {
        Self::Yono(YonoWeights::safetensors(
            "assets/models/yono_backbone_weights.safetensors",
            "assets/models/yono_head_weights.safetensors",
        ))
    }

    /// Uses repository default burnpack checkpoint paths.
    pub fn default_yono_burnpack() -> Self {
        Self::Yono(YonoWeights::burnpack(
            "assets/models/yono_backbone.bpk",
            "assets/models/yono_head.bpk",
        ))
    }

    /// Uses repository default ZipSplat burnpack checkpoint path.
    pub fn default_zipsplat_burnpack() -> Self {
        Self::ZipSplat(ZipSplatWeights::burnpack("assets/models/zipsplat.bpk"))
    }

    /// Creates weights from explicit backbone/head paths.
    pub fn from_paths(backbone: impl Into<PathBuf>, head: impl Into<PathBuf>) -> Self {
        Self::Yono(YonoWeights::new(backbone, head))
    }

    /// Creates weights from prebuilt YoNoSplat weight settings.
    pub fn from_yono(yono: YonoWeights) -> Self {
        Self::Yono(yono)
    }

    /// Creates weights from prebuilt ZipSplat weight settings.
    pub fn from_zipsplat(zipsplat: ZipSplatWeights) -> Self {
        Self::ZipSplat(zipsplat)
    }

    pub fn model(&self) -> PipelineModel {
        match self {
            Self::Yono(_) => PipelineModel::Yono,
            Self::ZipSplat(_) => PipelineModel::ZipSplat,
        }
    }

    pub fn yono(&self) -> Option<&YonoWeights> {
        match self {
            Self::Yono(weights) => Some(weights),
            Self::ZipSplat(_) => None,
        }
    }

    pub fn zipsplat(&self) -> Option<&ZipSplatWeights> {
        match self {
            Self::Yono(_) => None,
            Self::ZipSplat(weights) => Some(weights),
        }
    }

    /// Overrides checkpoint format for the YoNoSplat weights.
    pub fn with_format(self, format: YonoWeightFormat) -> Self {
        match self {
            Self::Yono(yono) => Self::Yono(yono.with_format(format)),
            other => other,
        }
    }

    /// Sets preferred burnpack precision for YoNo model loading.
    pub fn with_precision(self, precision: YonoWeightPrecision) -> Self {
        match self {
            Self::Yono(yono) => Self::Yono(yono.with_precision(precision)),
            other => other,
        }
    }

    pub fn with_zipsplat_format(self, format: ZipSplatWeightFormat) -> Self {
        match self {
            Self::ZipSplat(zipsplat) => Self::ZipSplat(zipsplat.with_format(format)),
            other => other,
        }
    }

    pub fn with_zipsplat_precision(self, precision: ZipSplatWeightPrecision) -> Self {
        match self {
            Self::ZipSplat(zipsplat) => Self::ZipSplat(zipsplat.with_precision(precision)),
            other => other,
        }
    }

    /// Resolves YoNoSplat model files from cache, downloading from remote when missing.
    pub fn resolve_or_bootstrap_yono(
        format: YonoWeightFormat,
    ) -> Result<Self, ModelBootstrapError> {
        Self::resolve_or_bootstrap_yono_with_precision(format, YonoWeightPrecision::F16)
    }

    /// Resolves YoNoSplat model files with explicit precision selection.
    pub fn resolve_or_bootstrap_yono_with_precision(
        format: YonoWeightFormat,
        precision: YonoWeightPrecision,
    ) -> Result<Self, ModelBootstrapError> {
        Ok(Self::Yono(
            resolve_or_bootstrap_yono_weights_with_precision(format, precision)?,
        ))
    }

    /// Resolves YoNoSplat model files with explicit precision and progress callbacks.
    pub fn resolve_or_bootstrap_yono_with_precision_and_progress<F>(
        format: YonoWeightFormat,
        precision: YonoWeightPrecision,
        progress: F,
    ) -> Result<Self, ModelBootstrapError>
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        Ok(Self::Yono(
            resolve_or_bootstrap_yono_weights_with_precision_and_progress(
                format, precision, progress,
            )?,
        ))
    }

    /// Resolves YoNoSplat model files with explicit bootstrap configuration.
    pub fn resolve_or_bootstrap_yono_with_config(
        format: YonoWeightFormat,
        bootstrap_cfg: &YonoBootstrapConfig,
    ) -> Result<Self, ModelBootstrapError> {
        Self::resolve_or_bootstrap_yono_with_config_and_precision(
            format,
            bootstrap_cfg,
            bootstrap_cfg.burnpack_precision,
        )
    }

    /// Resolves YoNoSplat model files with explicit bootstrap configuration and precision.
    pub fn resolve_or_bootstrap_yono_with_config_and_precision(
        format: YonoWeightFormat,
        bootstrap_cfg: &YonoBootstrapConfig,
        precision: YonoWeightPrecision,
    ) -> Result<Self, ModelBootstrapError> {
        let mut cfg = bootstrap_cfg.clone();
        cfg.burnpack_precision = precision;
        Ok(Self::Yono(resolve_or_bootstrap_yono_weights_with_config(
            format, &cfg,
        )?))
    }

    /// Resolves ZipSplat native burnpack from cache, downloading/importing the official
    /// PyTorch checkpoint when missing.
    pub fn resolve_or_bootstrap_zipsplat() -> Result<Self, ModelBootstrapError> {
        Self::resolve_or_bootstrap_zipsplat_with_precision(ZipSplatWeightPrecision::F16)
    }

    /// Resolves ZipSplat native burnpack with explicit precision selection.
    pub fn resolve_or_bootstrap_zipsplat_with_precision(
        precision: ZipSplatWeightPrecision,
    ) -> Result<Self, ModelBootstrapError> {
        Ok(Self::ZipSplat(
            resolve_or_bootstrap_zipsplat_weights_with_precision(precision)?,
        ))
    }

    /// Resolves ZipSplat native burnpack with explicit precision and progress callbacks.
    pub fn resolve_or_bootstrap_zipsplat_with_precision_and_progress<F>(
        precision: ZipSplatWeightPrecision,
        progress: F,
    ) -> Result<Self, ModelBootstrapError>
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        Ok(Self::ZipSplat(
            resolve_or_bootstrap_zipsplat_weights_with_precision_and_progress(precision, progress)?,
        ))
    }

    /// Resolves ZipSplat native burnpack with explicit bootstrap configuration.
    pub fn resolve_or_bootstrap_zipsplat_with_config(
        bootstrap_cfg: &ZipSplatBootstrapConfig,
    ) -> Result<Self, ModelBootstrapError> {
        Ok(Self::ZipSplat(
            resolve_or_bootstrap_zipsplat_weights_with_config(bootstrap_cfg)?,
        ))
    }
}

/// Canonical gaussian tensor output type for this crate.
pub type PipelineGaussians = FlatGaussians<BackendImpl>;

/// Timed inference result.
#[derive(Debug)]
pub struct PipelineRunOutput {
    pub gaussians: PipelineGaussians,
    pub timings: ForwardTimings,
}

/// In-memory image input for image -> gaussian inference.
#[derive(Debug, Clone, Copy)]
pub struct PipelineInputImage<'a> {
    pub name: &'a str,
    pub bytes: &'a [u8],
}

/// Timed inference result with predicted camera poses.
#[derive(Debug)]
pub struct PipelineRunWithCameras {
    pub gaussians: PipelineGaussians,
    pub timings: ForwardTimings,
    pub camera_poses: Vec<[[f32; 4]; 4]>,
}

/// End-to-end `infer + export` report.
#[derive(Debug, Clone)]
pub struct PipelineExportReport {
    pub selected_gaussians: usize,
    pub select_millis: f64,
    pub write_millis: f64,
}

/// Model load metadata for transparency/debugging.
#[derive(Debug, Clone)]
pub struct PipelineLoadReport {
    pub backbone: ComponentLoadReport,
    pub head: ComponentLoadReport,
}

impl PipelineLoadReport {
    pub fn is_strict_success(&self) -> bool {
        self.backbone.is_strict_success() && self.head.is_strict_success()
    }

    pub fn strict_failures(&self) -> Vec<String> {
        let mut failures = self.backbone.strict_failures("backbone");
        failures.extend(self.head.strict_failures("head"));
        failures
    }
}

/// Per-component checkpoint application report.
#[derive(Debug, Clone)]
pub struct ComponentLoadReport {
    pub applied: usize,
    pub missing: Vec<String>,
    pub unused: Vec<String>,
    pub skipped: Vec<String>,
}

impl ComponentLoadReport {
    pub fn synthetic_success(applied: usize) -> Self {
        Self {
            applied,
            missing: Vec::new(),
            unused: Vec::new(),
            skipped: Vec::new(),
        }
    }

    pub fn is_strict_success(&self) -> bool {
        self.missing.is_empty() && self.unused.is_empty() && self.skipped.is_empty()
    }

    pub fn strict_failures(&self, component: &str) -> Vec<String> {
        let mut failures = Vec::new();
        for key in &self.missing {
            failures.push(format!("{component} missing tensor: {key}"));
        }
        for key in &self.unused {
            failures.push(format!("{component} unused tensor: {key}"));
        }
        for key in &self.skipped {
            failures.push(format!("{component} skipped tensor: {key}"));
        }
        failures
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("invalid model `{0}`; expected `yono` or `zipsplat`")]
    InvalidModel(String),
    #[error("unsupported model selection")]
    UnsupportedModel,
    #[error("pipeline config selects {config_model:?}, but weights are for {weights_model:?}")]
    WeightModelMismatch {
        config_model: PipelineModel,
        weights_model: PipelineModel,
    },
    #[error("loaded pipeline key {loaded:?} cannot run request key {requested:?}")]
    PipelineLoadKeyMismatch {
        loaded: PipelineLoadKey,
        requested: PipelineLoadKey,
    },
    #[error("failed to resolve/bootstrap model weights: {0}")]
    Bootstrap(#[from] ModelBootstrapError),
    #[error("yono pipeline error: {0}")]
    Yono(#[from] YonoPipelineError),
    #[error("zipsplat pipeline error: {0}")]
    ZipSplat(#[from] ZipSplatPipelineError),
    #[error("failed to read debug tensor from backend: {0}")]
    TensorData(String),
    #[error("gaussian glb export failed: {0}")]
    Export(#[from] burn_yono::glb::GlbExportError),
}

/// High-level, WGPU-native multi-view pipeline.
///
/// This type intentionally locks the outer API to the validated WGPU path.
#[derive(Debug)]
pub struct ImageToGaussianPipeline {
    cfg: PipelineConfig,
    device: BackendDevice,
    model: LoadedPipeline,
}

#[derive(Debug)]
enum LoadedPipeline {
    Yono(Box<YonoModelBundle<BackendImpl>>),
    ZipSplat(Box<ZipSplatModelBundle<BackendImpl>>),
}

impl ImageToGaussianPipeline {
    /// Loads model weights on a user-provided WGPU device.
    pub fn load(
        device: BackendDevice,
        cfg: PipelineConfig,
        weights: PipelineWeights,
    ) -> Result<(Self, PipelineLoadReport), PipelineError> {
        Self::load_with_progress(device, cfg, weights, |_| {})
    }

    /// Loads model weights on a user-provided WGPU device and reports progress.
    pub fn load_with_progress<F>(
        device: BackendDevice,
        cfg: PipelineConfig,
        weights: PipelineWeights,
        progress: F,
    ) -> Result<(Self, PipelineLoadReport), PipelineError>
    where
        F: Fn(String),
    {
        let weights_model = weights.model();
        match (cfg.model, weights) {
            (PipelineModel::Yono, PipelineWeights::Yono(yono_weights)) => {
                let (yono, report) = YonoModelBundle::load_from_weights_with_progress(
                    &device,
                    &yono_weights,
                    progress,
                )?;
                Ok((
                    Self {
                        cfg,
                        device,
                        model: LoadedPipeline::Yono(Box::new(yono)),
                    },
                    PipelineLoadReport {
                        backbone: from_apply_summary(&report.backbone),
                        head: from_apply_summary(&report.head),
                    },
                ))
            }
            (PipelineModel::ZipSplat, PipelineWeights::ZipSplat(weights)) => {
                let model_cfg = ZipSplatConfig {
                    image_size: cfg.image_size,
                    ..ZipSplatConfig::default()
                };
                let (zipsplat, report) = ZipSplatModelBundle::load_from_weights_with_config(
                    &device, &weights, model_cfg,
                )?;
                Ok((
                    Self {
                        cfg,
                        device,
                        model: LoadedPipeline::ZipSplat(Box::new(zipsplat)),
                    },
                    PipelineLoadReport {
                        backbone: from_zipsplat_apply_summary(&report.model),
                        head: ComponentLoadReport::synthetic_success(0),
                    },
                ))
            }
            (config_model, _) => Err(PipelineError::WeightModelMismatch {
                config_model,
                weights_model,
            }),
        }
    }

    /// Loads model weights on the default WGPU device.
    pub fn load_default(
        cfg: PipelineConfig,
        weights: PipelineWeights,
    ) -> Result<(Self, PipelineLoadReport), PipelineError> {
        Self::load_with_progress(default_device(), cfg, weights, |_| {})
    }

    /// Loads model weights on the default WGPU device and reports progress.
    pub fn load_default_with_progress<F>(
        cfg: PipelineConfig,
        weights: PipelineWeights,
        progress: F,
    ) -> Result<(Self, PipelineLoadReport), PipelineError>
    where
        F: Fn(String),
    {
        Self::load_with_progress(default_device(), cfg, weights, progress)
    }

    /// Loads YoNo models directly from burnpack parts bytes.
    ///
    /// This is primarily intended for wasm/web runtimes that fetch
    /// `*.bpk.parts.json` + `*.bpk.part-*` assets from HTTP storage.
    pub fn load_from_yono_parts(
        device: BackendDevice,
        cfg: PipelineConfig,
        backbone_parts: &[Vec<u8>],
        head_parts: &[Vec<u8>],
    ) -> Result<(Self, PipelineLoadReport), PipelineError> {
        Self::load_from_yono_parts_with_progress(device, cfg, backbone_parts, head_parts, |_| {})
    }

    /// Loads YoNo models directly from burnpack parts bytes and reports progress.
    pub fn load_from_yono_parts_with_progress<F>(
        device: BackendDevice,
        cfg: PipelineConfig,
        backbone_parts: &[Vec<u8>],
        head_parts: &[Vec<u8>],
        progress: F,
    ) -> Result<(Self, PipelineLoadReport), PipelineError>
    where
        F: Fn(String),
    {
        match cfg.model {
            PipelineModel::Yono => {
                let (yono, report) = YonoModelBundle::load_from_burnpack_part_bytes_with_progress(
                    &device,
                    backbone_parts,
                    head_parts,
                    progress,
                )?;
                Ok((
                    Self {
                        cfg,
                        device,
                        model: LoadedPipeline::Yono(Box::new(yono)),
                    },
                    PipelineLoadReport {
                        backbone: from_apply_summary(&report.backbone),
                        head: from_apply_summary(&report.head),
                    },
                ))
            }
            PipelineModel::ZipSplat => Err(PipelineError::UnsupportedModel),
        }
    }

    /// Loads ZipSplat directly from burnpack parts bytes.
    pub fn load_from_zipsplat_parts(
        device: BackendDevice,
        cfg: PipelineConfig,
        model_parts: &[Vec<u8>],
    ) -> Result<(Self, PipelineLoadReport), PipelineError> {
        Self::load_from_zipsplat_parts_with_progress(device, cfg, model_parts, |_| {})
    }

    /// Loads ZipSplat directly from burnpack parts bytes and reports progress.
    pub fn load_from_zipsplat_parts_with_progress<F>(
        device: BackendDevice,
        cfg: PipelineConfig,
        model_parts: &[Vec<u8>],
        progress: F,
    ) -> Result<(Self, PipelineLoadReport), PipelineError>
    where
        F: Fn(String),
    {
        match cfg.model {
            PipelineModel::Yono => Err(PipelineError::UnsupportedModel),
            PipelineModel::ZipSplat => {
                let model_cfg = ZipSplatConfig {
                    image_size: cfg.image_size,
                    ..ZipSplatConfig::default()
                };
                let (zipsplat, report) =
                    ZipSplatModelBundle::load_from_burnpack_part_bytes_with_progress(
                        &device,
                        model_cfg,
                        model_parts,
                        progress,
                    )?;
                Ok((
                    Self {
                        cfg,
                        device,
                        model: LoadedPipeline::ZipSplat(Box::new(zipsplat)),
                    },
                    PipelineLoadReport {
                        backbone: from_zipsplat_apply_summary(&report.model),
                        head: ComponentLoadReport::synthetic_success(0),
                    },
                ))
            }
        }
    }

    /// Loads the pipeline using cache-first, auto-downloaded default weights.
    pub fn load_default_bootstrapped(
        cfg: PipelineConfig,
        format: YonoWeightFormat,
    ) -> Result<(Self, PipelineLoadReport), PipelineError> {
        if cfg.model != PipelineModel::Yono {
            return Err(PipelineError::UnsupportedModel);
        }
        let weights = PipelineWeights::resolve_or_bootstrap_yono_with_precision(
            format,
            YonoWeightPrecision::F16,
        )?;
        Self::load_default(cfg, weights)
    }

    /// Loads the pipeline using cache-first auto-downloaded default weights and explicit precision.
    pub fn load_default_bootstrapped_with_precision(
        cfg: PipelineConfig,
        format: YonoWeightFormat,
        precision: YonoWeightPrecision,
    ) -> Result<(Self, PipelineLoadReport), PipelineError> {
        if cfg.model != PipelineModel::Yono {
            return Err(PipelineError::UnsupportedModel);
        }
        let weights = PipelineWeights::resolve_or_bootstrap_yono_with_precision(format, precision)?;
        Self::load_default(cfg, weights)
    }

    /// Loads the pipeline using explicit bootstrap configuration.
    pub fn load_default_bootstrapped_with_config(
        cfg: PipelineConfig,
        format: YonoWeightFormat,
        bootstrap_cfg: &YonoBootstrapConfig,
    ) -> Result<(Self, PipelineLoadReport), PipelineError> {
        if cfg.model != PipelineModel::Yono {
            return Err(PipelineError::UnsupportedModel);
        }
        let weights = PipelineWeights::resolve_or_bootstrap_yono_with_config_and_precision(
            format,
            bootstrap_cfg,
            bootstrap_cfg.burnpack_precision,
        )?;
        Self::load_default(cfg, weights)
    }

    /// Loads the pipeline using explicit bootstrap configuration and precision.
    pub fn load_default_bootstrapped_with_config_and_precision(
        cfg: PipelineConfig,
        format: YonoWeightFormat,
        bootstrap_cfg: &YonoBootstrapConfig,
        precision: YonoWeightPrecision,
    ) -> Result<(Self, PipelineLoadReport), PipelineError> {
        if cfg.model != PipelineModel::Yono {
            return Err(PipelineError::UnsupportedModel);
        }
        let weights = PipelineWeights::resolve_or_bootstrap_yono_with_config_and_precision(
            format,
            bootstrap_cfg,
            precision,
        )?;
        Self::load_default(cfg, weights)
    }

    /// Returns the loaded runtime configuration.
    pub fn config(&self) -> &PipelineConfig {
        &self.cfg
    }

    /// Returns the fields that require a distinct model load.
    pub fn load_key(&self) -> PipelineLoadKey {
        self.cfg.load_key()
    }

    /// Returns the internal WGPU device used for inference.
    pub fn device(&self) -> &BackendDevice {
        &self.device
    }

    fn yono(&self) -> Result<&YonoModelBundle<BackendImpl>, PipelineError> {
        match &self.model {
            LoadedPipeline::Yono(model) => Ok(model.as_ref()),
            LoadedPipeline::ZipSplat(_) => Err(PipelineError::UnsupportedModel),
        }
    }

    fn zipsplat(&self) -> Result<&ZipSplatModelBundle<BackendImpl>, PipelineError> {
        match &self.model {
            LoadedPipeline::Yono(_) => Err(PipelineError::UnsupportedModel),
            LoadedPipeline::ZipSplat(model) => Ok(model.as_ref()),
        }
    }

    fn validate_run_config(&self, run_cfg: &PipelineConfig) -> Result<(), PipelineError> {
        let loaded = self.load_key();
        let requested = run_cfg.load_key();
        if loaded != requested {
            return Err(PipelineError::PipelineLoadKeyMismatch { loaded, requested });
        }
        Ok(())
    }

    /// Runs multi-view inference from image paths.
    pub fn run_images<P: AsRef<Path>>(
        &self,
        image_paths: &[P],
    ) -> Result<PipelineGaussians, PipelineError> {
        if self.cfg.model == PipelineModel::ZipSplat {
            return Ok(self.run_images_timed(image_paths, false)?.gaussians);
        }
        let paths = normalize_paths(image_paths);
        let output = self.yono()?.forward_from_image_paths(
            paths.as_slice(),
            self.cfg.image_size,
            &self.device,
        )?;
        Ok(output.gaussians_flat)
    }

    /// Runs multi-view inference from in-memory image bytes.
    pub fn run_image_bytes(
        &self,
        images: &[PipelineInputImage<'_>],
    ) -> Result<PipelineGaussians, PipelineError> {
        if self.cfg.model == PipelineModel::ZipSplat {
            return Ok(self.run_image_bytes_timed(images, false)?.gaussians);
        }
        let named_images = normalize_input_images(images);
        let output = self.yono()?.forward_from_image_bytes(
            named_images.as_slice(),
            self.cfg.image_size,
            &self.device,
        )?;
        Ok(output.gaussians_flat)
    }

    /// Runs multi-view inference with timing information.
    pub fn run_images_timed<P: AsRef<Path>>(
        &self,
        image_paths: &[P],
        synchronize: bool,
    ) -> Result<PipelineRunOutput, PipelineError> {
        let paths = normalize_paths(image_paths);
        match self.cfg.model {
            PipelineModel::Yono => {
                let (output, timings) = self.yono()?.forward_from_image_paths_timed_with_sync(
                    paths.as_slice(),
                    self.cfg.image_size,
                    &self.device,
                    synchronize,
                )?;
                Ok(PipelineRunOutput {
                    gaussians: output.gaussians_flat,
                    timings,
                })
            }
            PipelineModel::ZipSplat => self.run_zipsplat_paths_timed(paths.as_slice(), synchronize),
        }
    }

    /// Runs in-memory multi-view inference with timing information.
    pub fn run_image_bytes_timed(
        &self,
        images: &[PipelineInputImage<'_>],
        synchronize: bool,
    ) -> Result<PipelineRunOutput, PipelineError> {
        match self.cfg.model {
            PipelineModel::Yono => {
                let named_images = normalize_input_images(images);
                let (output, timings) = self.yono()?.forward_from_image_bytes_timed_with_sync(
                    named_images.as_slice(),
                    self.cfg.image_size,
                    &self.device,
                    synchronize,
                )?;
                Ok(PipelineRunOutput {
                    gaussians: output.gaussians_flat,
                    timings,
                })
            }
            PipelineModel::ZipSplat => self.run_zipsplat_image_bytes_timed(images, synchronize),
        }
    }

    /// Runs path-based inference and returns camera poses for visualization/debugging.
    pub fn run_images_timed_with_cameras<P: AsRef<Path>>(
        &self,
        image_paths: &[P],
        synchronize: bool,
    ) -> Result<PipelineRunWithCameras, PipelineError> {
        match self.cfg.model {
            PipelineModel::Yono => {
                let paths = normalize_paths(image_paths);
                let (output, timings) = self.yono()?.forward_from_image_paths_timed_with_sync(
                    paths.as_slice(),
                    self.cfg.image_size,
                    &self.device,
                    synchronize,
                )?;
                let camera_poses = decode_camera_poses(output.camera_poses)?;
                Ok(PipelineRunWithCameras {
                    gaussians: output.gaussians_flat,
                    timings,
                    camera_poses,
                })
            }
            PipelineModel::ZipSplat => {
                let paths = normalize_paths(image_paths);
                let output = self.run_zipsplat_paths_timed(paths.as_slice(), synchronize)?;
                Ok(PipelineRunWithCameras {
                    gaussians: output.gaussians,
                    timings: output.timings,
                    camera_poses: identity_camera_poses(paths.len()),
                })
            }
        }
    }

    /// Runs in-memory inference and returns camera poses for visualization/debugging.
    pub fn run_image_bytes_timed_with_cameras(
        &self,
        images: &[PipelineInputImage<'_>],
        synchronize: bool,
    ) -> Result<PipelineRunWithCameras, PipelineError> {
        match self.cfg.model {
            PipelineModel::Yono => {
                let named_images = normalize_input_images(images);
                let (output, timings) = self.yono()?.forward_from_image_bytes_timed_with_sync(
                    named_images.as_slice(),
                    self.cfg.image_size,
                    &self.device,
                    synchronize,
                )?;
                let camera_poses = decode_camera_poses(output.camera_poses)?;
                Ok(PipelineRunWithCameras {
                    gaussians: output.gaussians_flat,
                    timings,
                    camera_poses,
                })
            }
            PipelineModel::ZipSplat => {
                let output = self.run_zipsplat_image_bytes_timed(images, synchronize)?;
                Ok(PipelineRunWithCameras {
                    gaussians: output.gaussians,
                    timings: output.timings,
                    camera_poses: identity_camera_poses(images.len()),
                })
            }
        }
    }

    /// Runs in-memory inference with per-request runtime settings.
    ///
    /// The loaded model family and image size must match the pipeline load key,
    /// but export quality and ZipSplat compression can vary between requests.
    pub fn run_image_bytes_timed_with_cameras_with_config(
        &self,
        images: &[PipelineInputImage<'_>],
        run_cfg: &PipelineConfig,
        synchronize: bool,
    ) -> Result<PipelineRunWithCameras, PipelineError> {
        self.validate_run_config(run_cfg)?;
        match run_cfg.model {
            PipelineModel::Yono => {
                let named_images = normalize_input_images(images);
                let (output, timings) = self.yono()?.forward_from_image_bytes_timed_with_sync(
                    named_images.as_slice(),
                    run_cfg.image_size,
                    &self.device,
                    synchronize,
                )?;
                let camera_poses = decode_camera_poses(output.camera_poses)?;
                Ok(PipelineRunWithCameras {
                    gaussians: output.gaussians_flat,
                    timings,
                    camera_poses,
                })
            }
            PipelineModel::ZipSplat => {
                let output = self.run_zipsplat_image_bytes_timed_with_r(
                    images,
                    run_cfg.effective_zipsplat_r(),
                    synchronize,
                )?;
                Ok(PipelineRunWithCameras {
                    gaussians: output.gaussians,
                    timings: output.timings,
                    camera_poses: identity_camera_poses(images.len()),
                })
            }
        }
    }

    /// Runs in-memory inference and returns camera poses for visualization/debugging.
    #[cfg(target_arch = "wasm32")]
    pub async fn run_image_bytes_timed_with_cameras_async(
        &self,
        images: &[PipelineInputImage<'_>],
        _synchronize: bool,
    ) -> Result<PipelineRunWithCameras, PipelineError> {
        match self.cfg.model {
            PipelineModel::Yono => {
                let named_images = normalize_input_images(images);
                let total_start_ms = wasm_now_ms();
                let output = self.yono()?.forward_from_image_bytes(
                    named_images.as_slice(),
                    self.cfg.image_size,
                    &self.device,
                )?;
                let camera_poses = decode_camera_poses_async(output.camera_poses).await?;
                let mut timings = ForwardTimings::default();
                let total_secs = ((wasm_now_ms() - total_start_ms) / 1000.0).max(0.0);
                if total_secs.is_finite() {
                    timings.total = Duration::from_secs_f64(total_secs);
                }
                Ok(PipelineRunWithCameras {
                    gaussians: output.gaussians_flat,
                    timings,
                    camera_poses,
                })
            }
            PipelineModel::ZipSplat => {
                let output = self.run_zipsplat_image_bytes_timed(images, _synchronize)?;
                Ok(PipelineRunWithCameras {
                    gaussians: output.gaussians,
                    timings: output.timings,
                    camera_poses: identity_camera_poses(images.len()),
                })
            }
        }
    }

    /// Runs in-memory inference with per-request runtime settings.
    ///
    /// The loaded model family and image size must match the pipeline load key,
    /// but export quality and ZipSplat compression can vary between requests.
    #[cfg(target_arch = "wasm32")]
    pub async fn run_image_bytes_timed_with_cameras_with_config_async(
        &self,
        images: &[PipelineInputImage<'_>],
        run_cfg: &PipelineConfig,
        _synchronize: bool,
    ) -> Result<PipelineRunWithCameras, PipelineError> {
        self.validate_run_config(run_cfg)?;
        match run_cfg.model {
            PipelineModel::Yono => {
                let named_images = normalize_input_images(images);
                let total_start_ms = wasm_now_ms();
                let output = self.yono()?.forward_from_image_bytes(
                    named_images.as_slice(),
                    run_cfg.image_size,
                    &self.device,
                )?;
                let camera_poses = decode_camera_poses_async(output.camera_poses).await?;
                let mut timings = ForwardTimings::default();
                let total_secs = ((wasm_now_ms() - total_start_ms) / 1000.0).max(0.0);
                if total_secs.is_finite() {
                    timings.total = Duration::from_secs_f64(total_secs);
                }
                Ok(PipelineRunWithCameras {
                    gaussians: output.gaussians_flat,
                    timings,
                    camera_poses,
                })
            }
            PipelineModel::ZipSplat => {
                let output = self.run_zipsplat_image_bytes_timed_with_r(
                    images,
                    run_cfg.effective_zipsplat_r(),
                    _synchronize,
                )?;
                Ok(PipelineRunWithCameras {
                    gaussians: output.gaussians,
                    timings: output.timings,
                    camera_poses: identity_camera_poses(images.len()),
                })
            }
        }
    }

    /// Saves gaussian tensors to GLB with `KHR_gaussian_splatting`.
    pub fn save_glb(
        &self,
        output: impl AsRef<Path>,
        gaussians: &PipelineGaussians,
        options: &GlbExportOptions,
    ) -> Result<burn_yono::glb::GlbExportReport, PipelineError> {
        Ok(burn_yono::glb::save_gaussians_to_glb_timed(
            output.as_ref(),
            gaussians,
            options,
        )?)
    }

    /// Converts gaussian tensors into a host-side export bundle.
    pub fn select_export_gaussians(
        &self,
        gaussians: &PipelineGaussians,
        options: &GlbExportOptions,
    ) -> Result<burn_yono::glb::ExportGaussians, PipelineError> {
        Ok(burn_yono::glb::select_export_gaussians(gaussians, options)?)
    }

    /// Converts gaussian tensors into a host-side export bundle.
    #[cfg(target_arch = "wasm32")]
    pub async fn select_export_gaussians_async(
        &self,
        gaussians: &PipelineGaussians,
        options: &GlbExportOptions,
    ) -> Result<burn_yono::glb::ExportGaussians, PipelineError> {
        Ok(burn_yono::glb::select_export_gaussians_async(gaussians, options).await?)
    }

    /// One-shot helper that runs inference then writes a GLB using quality defaults.
    pub fn run_images_to_glb<P: AsRef<Path>>(
        &self,
        image_paths: &[P],
        output: impl AsRef<Path>,
    ) -> Result<PipelineExportReport, PipelineError> {
        let gaussians = self.run_images(image_paths)?;
        let report = self.save_glb(output, &gaussians, &self.cfg.quality.export_options())?;
        Ok(PipelineExportReport {
            selected_gaussians: report.selected_gaussians,
            select_millis: report.select_millis,
            write_millis: report.write_millis,
        })
    }

    fn run_zipsplat_paths_timed(
        &self,
        image_paths: &[PathBuf],
        synchronize: bool,
    ) -> Result<PipelineRunOutput, PipelineError> {
        let (gaussians, timings) = self.zipsplat()?.forward_from_image_paths_timed_with_sync(
            image_paths,
            self.cfg.image_size,
            ZipSplatCompression::new(self.cfg.effective_zipsplat_r()),
            &self.device,
            synchronize,
        )?;
        Ok(PipelineRunOutput {
            gaussians,
            timings: from_zipsplat_timings(timings),
        })
    }

    fn run_zipsplat_image_bytes_timed(
        &self,
        images: &[PipelineInputImage<'_>],
        synchronize: bool,
    ) -> Result<PipelineRunOutput, PipelineError> {
        self.run_zipsplat_image_bytes_timed_with_r(
            images,
            self.cfg.effective_zipsplat_r(),
            synchronize,
        )
    }

    fn run_zipsplat_image_bytes_timed_with_r(
        &self,
        images: &[PipelineInputImage<'_>],
        zipsplat_r: usize,
        synchronize: bool,
    ) -> Result<PipelineRunOutput, PipelineError> {
        let named_images = normalize_input_images(images);
        let (gaussians, timings) = self.zipsplat()?.forward_from_image_bytes_timed_with_sync(
            named_images.as_slice(),
            self.cfg.image_size,
            ZipSplatCompression::new(zipsplat_r),
            &self.device,
            synchronize,
        )?;
        Ok(PipelineRunOutput {
            gaussians,
            timings: from_zipsplat_timings(timings),
        })
    }
}

fn normalize_paths<P: AsRef<Path>>(image_paths: &[P]) -> Vec<PathBuf> {
    image_paths
        .iter()
        .map(|path| path.as_ref().to_path_buf())
        .collect()
}

fn normalize_input_images<'a>(images: &'a [PipelineInputImage<'a>]) -> Vec<(&'a str, &'a [u8])> {
    images
        .iter()
        .map(|image| (image.name, image.bytes))
        .collect()
}

fn from_zipsplat_timings(timings: ZipSplatForwardTimings) -> ForwardTimings {
    ForwardTimings {
        image_load: timings.image_load,
        backbone: timings.backbone,
        head: timings.head,
        total: timings.total,
    }
}

fn from_zipsplat_apply_summary(
    summary: &burn_zipsplat::ZipSplatApplySummary,
) -> ComponentLoadReport {
    ComponentLoadReport {
        applied: summary.applied,
        missing: summary.missing.clone(),
        unused: summary.unused.clone(),
        skipped: summary.skipped.clone(),
    }
}

fn identity_camera_poses(views: usize) -> Vec<[[f32; 4]; 4]> {
    (0..views)
        .map(|_| {
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        })
        .collect()
}

fn decode_camera_poses(poses: Tensor<BackendImpl, 4>) -> Result<Vec<[[f32; 4]; 4]>, PipelineError> {
    let [batch, views, _, _] = poses.shape().dims::<4>();
    let values = poses
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| PipelineError::TensorData(format!("{err:?}")))?;
    Ok(decode_camera_poses_values(values.as_slice(), batch, views))
}

#[cfg(target_arch = "wasm32")]
async fn decode_camera_poses_async(
    poses: Tensor<BackendImpl, 4>,
) -> Result<Vec<[[f32; 4]; 4]>, PipelineError> {
    let [batch, views, _, _] = poses.shape().dims::<4>();
    let values = poses
        .into_data_async()
        .await
        .to_vec::<f32>()
        .map_err(|err| PipelineError::TensorData(format!("{err:?}")))?;
    Ok(decode_camera_poses_values(values.as_slice(), batch, views))
}

fn decode_camera_poses_values(values: &[f32], batch: usize, views: usize) -> Vec<[[f32; 4]; 4]> {
    let mut out = Vec::with_capacity(batch * views);
    for idx in 0..(batch * views) {
        let off = idx * 16;
        out.push([
            [
                values[off],
                values[off + 1],
                values[off + 2],
                values[off + 3],
            ],
            [
                values[off + 4],
                values[off + 5],
                values[off + 6],
                values[off + 7],
            ],
            [
                values[off + 8],
                values[off + 9],
                values[off + 10],
                values[off + 11],
            ],
            [
                values[off + 12],
                values[off + 13],
                values[off + 14],
                values[off + 15],
            ],
        ]);
    }
    out
}

#[cfg(target_arch = "wasm32")]
fn wasm_now_ms() -> f64 {
    js_sys::Date::now()
}

fn from_apply_summary(summary: &ApplySummary) -> ComponentLoadReport {
    ComponentLoadReport {
        applied: summary.applied,
        missing: summary.missing.clone(),
        unused: summary.unused.clone(),
        skipped: summary.skipped.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_parse_accepts_public_ids() {
        assert_eq!(
            "yono".parse::<PipelineModel>().unwrap(),
            PipelineModel::Yono
        );
        assert_eq!(
            "YoNoSplat".parse::<PipelineModel>().unwrap(),
            PipelineModel::Yono
        );
        assert_eq!(
            "zipsplat".parse::<PipelineModel>().unwrap(),
            PipelineModel::ZipSplat
        );
    }

    #[test]
    fn zipsplat_capabilities_expose_compression_control() {
        let caps = PipelineModel::ZipSplat.capabilities();
        assert_eq!(caps.min_views, 1);
        assert_eq!(caps.max_views, Some(24));
        assert_eq!(caps.default_image_size, 252);
        match caps.quality {
            PipelineQualityControl::ZipSplatCompression {
                min_r,
                max_r,
                presets,
            } => {
                assert_eq!(min_r, 1);
                assert_eq!(max_r, 16);
                assert_eq!(
                    presets.iter().map(|preset| preset.r).collect::<Vec<_>>(),
                    vec![1, 2, 4, 8]
                );
            }
            PipelineQualityControl::ExportOnly => panic!("ZipSplat must expose compression"),
        }
    }

    #[test]
    fn quality_presets_map_to_zipsplat_r() {
        assert_eq!(PipelineQuality::Full.default_zipsplat_r(), 1);
        assert_eq!(PipelineQuality::Balanced.default_zipsplat_r(), 2);
        assert_eq!(PipelineQuality::Compact.default_zipsplat_r(), 4);
        assert_eq!(PipelineQuality::Preview.default_zipsplat_r(), 8);
    }

    #[test]
    fn pipeline_config_clamps_zipsplat_r() {
        let cfg = PipelineConfig::default()
            .with_model(PipelineModel::ZipSplat)
            .with_zipsplat_r(999);
        assert_eq!(cfg.effective_zipsplat_r(), 16);
    }

    #[test]
    fn pipeline_load_key_ignores_runtime_quality_controls() {
        let base = PipelineConfig::default()
            .with_model(PipelineModel::ZipSplat)
            .with_quality(PipelineQuality::Balanced);
        let compact = base
            .clone()
            .with_quality(PipelineQuality::Compact)
            .with_zipsplat_r(8);
        assert_eq!(base.load_key(), compact.load_key());

        let larger_image = PipelineConfig {
            image_size: base.image_size
                + PipelineModel::ZipSplat.capabilities().image_size_multiple,
            ..base.clone()
        };
        assert_ne!(base.load_key(), larger_image.load_key());

        let yono = PipelineConfig::default().with_model(PipelineModel::Yono);
        assert_ne!(base.load_key(), yono.load_key());
    }

    #[test]
    fn pipeline_weights_report_model_family() {
        assert_eq!(
            PipelineWeights::default_yono_burnpack().model(),
            PipelineModel::Yono
        );
        assert_eq!(
            PipelineWeights::default_zipsplat_burnpack().model(),
            PipelineModel::ZipSplat
        );
    }

    #[test]
    fn strict_load_report_lists_all_tensor_mismatches() {
        let report = ComponentLoadReport {
            applied: 1,
            missing: vec!["a".to_string()],
            unused: vec!["b".to_string()],
            skipped: vec!["c".to_string()],
        };
        assert!(!report.is_strict_success());
        assert_eq!(
            report.strict_failures("head"),
            vec![
                "head missing tensor: a".to_string(),
                "head unused tensor: b".to_string(),
                "head skipped tensor: c".to_string()
            ]
        );
    }

    #[test]
    fn zipsplat_missing_weights_reports_native_checkpoint_path() {
        type TestBackend = burn::backend::NdArray<f32>;

        let device = <TestBackend as burn::prelude::Backend>::Device::default();
        let weights = ZipSplatWeights::burnpack("definitely_missing_zipsplat.bpk");
        let err = burn_zipsplat::ZipSplatModelBundle::<TestBackend>::load_from_weights_with_config(
            &device,
            &weights,
            ZipSplatConfig::tiny_for_tests(),
        )
        .expect_err("missing ZipSplat weights should fail before model execution");

        match err {
            ZipSplatPipelineError::MissingWeights(path) => {
                assert!(path.contains("definitely_missing_zipsplat.bpk"));
            }
            other => panic!("expected missing native weights, got {other:?}"),
        }
    }
}
