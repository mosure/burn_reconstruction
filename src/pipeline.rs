use std::path::{Path, PathBuf};

use crate::backend::{default_device, BackendDevice, BackendImpl};
use crate::bootstrap::{
    resolve_or_bootstrap_yono_weights, resolve_or_bootstrap_yono_weights_with_config,
    ModelBootstrapError, YonoBootstrapConfig,
};
use burn_yono::{
    glb::{GlbExportOptions, GlbSortMode},
    inference::{ApplySummary, ForwardTimings, YonoModelBundle, YonoPipelineError},
    model::gaussian::FlatGaussians,
    YonoWeightFormat, YonoWeights,
};

/// Model families supported by the outer multi-view pipeline.
///
/// Kept as an enum so additional multi-view -> 3DGS methods can be added
/// without changing call sites.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PipelineModel {
    Yono,
}

/// Export/inference quality presets.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PipelineQuality {
    Fast,
    #[default]
    Balanced,
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
            Self::Balanced => GlbExportOptions::default(),
            Self::High => GlbExportOptions {
                max_gaussians: 200_000,
                opacity_threshold: 0.0,
                sort_mode: GlbSortMode::Index,
            },
        }
    }
}

/// High-level model/runtime settings for image -> gaussian inference.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub model: PipelineModel,
    pub quality: PipelineQuality,
    pub image_size: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            model: PipelineModel::Yono,
            quality: PipelineQuality::Balanced,
            image_size: 224,
        }
    }
}

/// Weight configuration for loading the active model family.
#[derive(Debug, Clone)]
pub struct PipelineWeights {
    pub yono: YonoWeights,
}

impl PipelineWeights {
    /// Uses repository default safetensor checkpoint paths.
    pub fn default_yono_safetensors() -> Self {
        Self {
            yono: YonoWeights::safetensors(
                "assets/models/yono_backbone_weights.safetensors",
                "assets/models/yono_head_weights.safetensors",
            ),
        }
    }

    /// Uses repository default burnpack checkpoint paths.
    pub fn default_yono_burnpack() -> Self {
        Self {
            yono: YonoWeights::burnpack(
                "assets/models/yono_backbone.bpk",
                "assets/models/yono_head.bpk",
            ),
        }
    }

    /// Creates weights from explicit backbone/head paths.
    pub fn from_paths(backbone: impl Into<PathBuf>, head: impl Into<PathBuf>) -> Self {
        Self {
            yono: YonoWeights::new(backbone, head),
        }
    }

    /// Creates weights from prebuilt YoNoSplat weight settings.
    pub fn from_yono(yono: YonoWeights) -> Self {
        Self { yono }
    }

    /// Overrides checkpoint format for the YoNoSplat weights.
    pub fn with_format(mut self, format: YonoWeightFormat) -> Self {
        self.yono = self.yono.with_format(format);
        self
    }

    /// Resolves YoNoSplat model files from cache, downloading from remote when missing.
    pub fn resolve_or_bootstrap_yono(
        format: YonoWeightFormat,
    ) -> Result<Self, ModelBootstrapError> {
        Ok(Self {
            yono: resolve_or_bootstrap_yono_weights(format)?,
        })
    }

    /// Resolves YoNoSplat model files with explicit bootstrap configuration.
    pub fn resolve_or_bootstrap_yono_with_config(
        format: YonoWeightFormat,
        bootstrap_cfg: &YonoBootstrapConfig,
    ) -> Result<Self, ModelBootstrapError> {
        Ok(Self {
            yono: resolve_or_bootstrap_yono_weights_with_config(format, bootstrap_cfg)?,
        })
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

/// Per-component checkpoint application report.
#[derive(Debug, Clone)]
pub struct ComponentLoadReport {
    pub applied: usize,
    pub missing: Vec<String>,
    pub unused: Vec<String>,
    pub skipped: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("unsupported model selection")]
    UnsupportedModel,
    #[error("failed to resolve/bootstrap model weights: {0}")]
    Bootstrap(#[from] ModelBootstrapError),
    #[error("yono pipeline error: {0}")]
    Yono(#[from] YonoPipelineError),
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
    yono: YonoModelBundle<BackendImpl>,
}

impl ImageToGaussianPipeline {
    /// Loads model weights on a user-provided WGPU device.
    pub fn load(
        device: BackendDevice,
        cfg: PipelineConfig,
        weights: PipelineWeights,
    ) -> Result<(Self, PipelineLoadReport), PipelineError> {
        match cfg.model {
            PipelineModel::Yono => {
                let (yono, report) = YonoModelBundle::load_from_weights(&device, &weights.yono)?;
                Ok((
                    Self { cfg, device, yono },
                    PipelineLoadReport {
                        backbone: from_apply_summary(&report.backbone),
                        head: from_apply_summary(&report.head),
                    },
                ))
            }
        }
    }

    /// Loads model weights on the default WGPU device.
    pub fn load_default(
        cfg: PipelineConfig,
        weights: PipelineWeights,
    ) -> Result<(Self, PipelineLoadReport), PipelineError> {
        Self::load(default_device(), cfg, weights)
    }

    /// Loads the pipeline using cache-first, auto-downloaded default weights.
    pub fn load_default_bootstrapped(
        cfg: PipelineConfig,
        format: YonoWeightFormat,
    ) -> Result<(Self, PipelineLoadReport), PipelineError> {
        let weights = PipelineWeights::resolve_or_bootstrap_yono(format)?;
        Self::load_default(cfg, weights)
    }

    /// Loads the pipeline using explicit bootstrap configuration.
    pub fn load_default_bootstrapped_with_config(
        cfg: PipelineConfig,
        format: YonoWeightFormat,
        bootstrap_cfg: &YonoBootstrapConfig,
    ) -> Result<(Self, PipelineLoadReport), PipelineError> {
        let weights =
            PipelineWeights::resolve_or_bootstrap_yono_with_config(format, bootstrap_cfg)?;
        Self::load_default(cfg, weights)
    }

    /// Returns the loaded runtime configuration.
    pub fn config(&self) -> &PipelineConfig {
        &self.cfg
    }

    /// Returns the internal WGPU device used for inference.
    pub fn device(&self) -> &BackendDevice {
        &self.device
    }

    /// Runs multi-view inference from image paths.
    pub fn run_images<P: AsRef<Path>>(
        &self,
        image_paths: &[P],
    ) -> Result<PipelineGaussians, PipelineError> {
        let paths = normalize_paths(image_paths);
        let output = self.yono.forward_from_image_paths(
            paths.as_slice(),
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
        let (output, timings) = self.yono.forward_from_image_paths_timed_with_sync(
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
}

fn normalize_paths<P: AsRef<Path>>(image_paths: &[P]) -> Vec<PathBuf> {
    image_paths
        .iter()
        .map(|path| path.as_ref().to_path_buf())
        .collect()
}

fn from_apply_summary(summary: &ApplySummary) -> ComponentLoadReport {
    ComponentLoadReport {
        applied: summary.applied,
        missing: summary.missing.clone(),
        unused: summary.unused.clone(),
        skipped: summary.skipped.clone(),
    }
}
