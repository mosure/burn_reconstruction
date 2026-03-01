use std::path::{Path, PathBuf};

use crate::backend::{default_device, BackendDevice, BackendImpl};
use crate::bootstrap::{
    resolve_or_bootstrap_yono_weights_with_config,
    resolve_or_bootstrap_yono_weights_with_precision,
    resolve_or_bootstrap_yono_weights_with_precision_and_progress, ModelBootstrapError,
    YonoBootstrapConfig,
};
use burn::tensor::Tensor;
use burn_yono::{
    glb::{GlbExportOptions, GlbSortMode},
    inference::{ApplySummary, ForwardTimings, YonoModelBundle, YonoPipelineError},
    model::gaussian::FlatGaussians,
    YonoWeightFormat, YonoWeightPrecision, YonoWeights,
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

    /// Sets preferred burnpack precision for YoNo model loading.
    pub fn with_precision(mut self, precision: YonoWeightPrecision) -> Self {
        self.yono = self.yono.with_precision(precision);
        self
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
        Ok(Self {
            yono: resolve_or_bootstrap_yono_weights_with_precision(format, precision)?,
        })
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
        Ok(Self {
            yono: resolve_or_bootstrap_yono_weights_with_precision_and_progress(
                format, precision, progress,
            )?,
        })
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
        Ok(Self {
            yono: resolve_or_bootstrap_yono_weights_with_config(format, &cfg)?,
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
        match cfg.model {
            PipelineModel::Yono => {
                let (yono, report) = YonoModelBundle::load_from_burnpack_part_bytes(
                    &device,
                    backbone_parts,
                    head_parts,
                )?;
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

    /// Loads the pipeline using cache-first, auto-downloaded default weights.
    pub fn load_default_bootstrapped(
        cfg: PipelineConfig,
        format: YonoWeightFormat,
    ) -> Result<(Self, PipelineLoadReport), PipelineError> {
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
        let weights = PipelineWeights::resolve_or_bootstrap_yono_with_precision(format, precision)?;
        Self::load_default(cfg, weights)
    }

    /// Loads the pipeline using explicit bootstrap configuration.
    pub fn load_default_bootstrapped_with_config(
        cfg: PipelineConfig,
        format: YonoWeightFormat,
        bootstrap_cfg: &YonoBootstrapConfig,
    ) -> Result<(Self, PipelineLoadReport), PipelineError> {
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

    /// Runs multi-view inference from in-memory image bytes.
    pub fn run_image_bytes(
        &self,
        images: &[PipelineInputImage<'_>],
    ) -> Result<PipelineGaussians, PipelineError> {
        let named_images = normalize_input_images(images);
        let output = self.yono.forward_from_image_bytes(
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

    /// Runs in-memory multi-view inference with timing information.
    pub fn run_image_bytes_timed(
        &self,
        images: &[PipelineInputImage<'_>],
        synchronize: bool,
    ) -> Result<PipelineRunOutput, PipelineError> {
        let named_images = normalize_input_images(images);
        let (output, timings) = self.yono.forward_from_image_bytes_timed_with_sync(
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

    /// Runs path-based inference and returns camera poses for visualization/debugging.
    pub fn run_images_timed_with_cameras<P: AsRef<Path>>(
        &self,
        image_paths: &[P],
        synchronize: bool,
    ) -> Result<PipelineRunWithCameras, PipelineError> {
        let paths = normalize_paths(image_paths);
        let (output, timings) = self.yono.forward_from_image_paths_timed_with_sync(
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

    /// Runs in-memory inference and returns camera poses for visualization/debugging.
    pub fn run_image_bytes_timed_with_cameras(
        &self,
        images: &[PipelineInputImage<'_>],
        synchronize: bool,
    ) -> Result<PipelineRunWithCameras, PipelineError> {
        let named_images = normalize_input_images(images);
        let (output, timings) = self.yono.forward_from_image_bytes_timed_with_sync(
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

    /// Runs in-memory inference and returns camera poses for visualization/debugging.
    #[cfg(target_arch = "wasm32")]
    pub async fn run_image_bytes_timed_with_cameras_async(
        &self,
        images: &[PipelineInputImage<'_>],
        _synchronize: bool,
    ) -> Result<PipelineRunWithCameras, PipelineError> {
        let named_images = normalize_input_images(images);
        // On wasm, avoid std::time::Instant-backed timing paths.
        let output = self.yono.forward_from_image_bytes(
            named_images.as_slice(),
            self.cfg.image_size,
            &self.device,
        )?;
        let camera_poses = decode_camera_poses_async(output.camera_poses).await?;
        Ok(PipelineRunWithCameras {
            gaussians: output.gaussians_flat,
            timings: ForwardTimings::default(),
            camera_poses,
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

fn from_apply_summary(summary: &ApplySummary) -> ComponentLoadReport {
    ComponentLoadReport {
        applied: summary.applied,
        missing: summary.missing.clone(),
        unused: summary.unused.clone(),
        skipped: summary.skipped.clone(),
    }
}
