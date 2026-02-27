use std::path::{Path, PathBuf};

use burn::prelude::Backend;

#[cfg(feature = "yono")]
pub use burn_yono::inference::{YonoWeightFormat, YonoWeights};

#[cfg(feature = "yono")]
use burn_yono::{
    glb::{GlbExportOptions, GlbSortMode},
    inference::{ForwardTimings, YonoLoadReport, YonoModelBundle, YonoPipelineError},
    model::gaussian::FlatGaussians,
};

#[cfg(not(feature = "yono"))]
#[derive(Debug, Clone)]
pub struct YonoWeights;

#[cfg(not(feature = "yono"))]
#[derive(Debug, Clone)]
pub struct YonoLoadReport;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PipelineModel {
    Yono,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipelineQuality {
    Fast,
    Balanced,
    High,
}

impl PipelineQuality {
    #[cfg(feature = "yono")]
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

#[derive(Debug)]
pub struct PipelineRunOutput<B: Backend> {
    #[cfg(feature = "yono")]
    pub gaussians: FlatGaussians<B>,
    #[cfg(feature = "yono")]
    pub timings: ForwardTimings,
}

#[cfg(feature = "yono")]
pub type PipelineGaussians<B> = FlatGaussians<B>;

#[cfg(not(feature = "yono"))]
pub type PipelineGaussians<B> = std::marker::PhantomData<B>;

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("unsupported model selection")]
    UnsupportedModel,
    #[cfg(feature = "yono")]
    #[error("yono pipeline error: {0}")]
    Yono(#[from] YonoPipelineError),
    #[cfg(feature = "yono")]
    #[error("gaussian glb export failed: {0}")]
    Export(#[from] burn_yono::glb::GlbExportError),
    #[error("the `yono` feature is disabled in this build")]
    YonoFeatureDisabled,
    #[error("the `import` feature is required for loading model weights")]
    ImportFeatureDisabled,
    #[error("the `io` feature is required for image/glb operations")]
    IoFeatureDisabled,
}

#[derive(Debug)]
pub struct ImageToGaussianPipeline<B: Backend> {
    cfg: PipelineConfig,
    #[cfg(feature = "yono")]
    yono: YonoModelBundle<B>,
}

impl<B: Backend> ImageToGaussianPipeline<B> {
    pub fn load(
        device: &B::Device,
        cfg: PipelineConfig,
        yono_weights: Option<YonoWeights>,
    ) -> Result<(Self, Option<YonoLoadReport>), PipelineError> {
        match cfg.model {
            PipelineModel::Yono => {
                #[cfg(not(feature = "yono"))]
                {
                    let _ = device;
                    let _ = yono_weights;
                    return Err(PipelineError::YonoFeatureDisabled);
                }

                #[cfg(feature = "yono")]
                {
                    #[cfg(not(feature = "import"))]
                    let _ = device;

                    if !cfg!(feature = "import") {
                        return Err(PipelineError::ImportFeatureDisabled);
                    }

                    let Some(weights) = yono_weights else {
                        return Err(PipelineError::UnsupportedModel);
                    };

                    #[cfg(feature = "import")]
                    {
                        let (yono, report) = YonoModelBundle::load_from_weights(device, &weights)?;
                        Ok((Self { cfg, yono }, Some(report)))
                    }

                    #[cfg(not(feature = "import"))]
                    {
                        let _ = weights;
                        Err(PipelineError::ImportFeatureDisabled)
                    }
                }
            }
        }
    }

    pub fn config(&self) -> &PipelineConfig {
        &self.cfg
    }

    pub fn run_images(
        &self,
        image_paths: &[PathBuf],
        device: &B::Device,
    ) -> Result<PipelineGaussians<B>, PipelineError> {
        #[cfg(not(feature = "yono"))]
        {
            let _ = image_paths;
            let _ = device;
            Err(PipelineError::YonoFeatureDisabled)
        }

        #[cfg(feature = "yono")]
        {
            #[cfg(not(feature = "io"))]
            {
                let _ = (&self.yono, image_paths, device);
            }

            if !cfg!(feature = "io") {
                return Err(PipelineError::IoFeatureDisabled);
            }

            #[cfg(feature = "io")]
            {
                let output =
                    self.yono
                        .forward_from_image_paths(image_paths, self.cfg.image_size, device)?;
                Ok(output.gaussians_flat)
            }

            #[cfg(not(feature = "io"))]
            {
                Err(PipelineError::IoFeatureDisabled)
            }
        }
    }

    pub fn run_images_timed(
        &self,
        image_paths: &[PathBuf],
        device: &B::Device,
        _synchronize: bool,
    ) -> Result<PipelineRunOutput<B>, PipelineError> {
        #[cfg(not(feature = "yono"))]
        {
            let _ = image_paths;
            let _ = device;
            Err(PipelineError::YonoFeatureDisabled)
        }

        #[cfg(feature = "yono")]
        {
            #[cfg(not(feature = "io"))]
            {
                let _ = (&self.yono, image_paths, device);
            }

            if !cfg!(feature = "io") {
                return Err(PipelineError::IoFeatureDisabled);
            }

            #[cfg(feature = "io")]
            {
                let (output, timings) = self.yono.forward_from_image_paths_timed_with_sync(
                    image_paths,
                    self.cfg.image_size,
                    device,
                    _synchronize,
                )?;
                Ok(PipelineRunOutput {
                    gaussians: output.gaussians_flat,
                    timings,
                })
            }

            #[cfg(not(feature = "io"))]
            {
                Err(PipelineError::IoFeatureDisabled)
            }
        }
    }

    #[cfg(feature = "yono")]
    pub fn save_glb(
        &self,
        output: &Path,
        gaussians: &FlatGaussians<B>,
        options: &GlbExportOptions,
    ) -> Result<burn_yono::glb::GlbExportReport, PipelineError> {
        if !cfg!(feature = "io") {
            return Err(PipelineError::IoFeatureDisabled);
        }

        #[cfg(feature = "io")]
        {
            Ok(burn_yono::glb::save_gaussians_to_glb_timed(
                output, gaussians, options,
            )?)
        }

        #[cfg(not(feature = "io"))]
        {
            let _ = output;
            let _ = gaussians;
            let _ = options;
            Err(PipelineError::IoFeatureDisabled)
        }
    }
}
