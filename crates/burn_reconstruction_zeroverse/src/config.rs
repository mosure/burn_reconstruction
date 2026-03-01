use crate::error::ZeroverseError;

/// Synthetic data generation settings expected from a Zeroverse-like scene source.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ZeroverseDataConfig {
    pub scenes_per_step: usize,
    pub conditioning_views_per_scene: usize,
    pub target_views_per_scene: usize,
    pub image_size: usize,
    pub novel_view_ratio: f32,
}

impl Default for ZeroverseDataConfig {
    fn default() -> Self {
        Self {
            scenes_per_step: 1,
            conditioning_views_per_scene: 2,
            target_views_per_scene: 1,
            image_size: 224,
            novel_view_ratio: 0.5,
        }
    }
}

/// Differentiable render/loss bridge settings expected from a Brush-like renderer.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BrushRenderConfig {
    pub max_gaussians: usize,
    pub opacity_threshold: f32,
    pub photometric_loss_weight: f32,
    pub regularization_loss_weight: f32,
}

impl Default for BrushRenderConfig {
    fn default() -> Self {
        Self {
            max_gaussians: 100_000,
            opacity_threshold: 0.005,
            photometric_loss_weight: 1.0,
            regularization_loss_weight: 0.1,
        }
    }
}

/// Top-level fine-tune loop settings for YoNo with synthetic novel-view supervision.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct YonoFineTuneConfig {
    pub steps_per_epoch: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub freeze_backbone: bool,
    pub novel_view_loss_weight: f32,
    pub zeroverse: ZeroverseDataConfig,
    pub brush: BrushRenderConfig,
}

impl Default for YonoFineTuneConfig {
    fn default() -> Self {
        Self {
            steps_per_epoch: 128,
            batch_size: 1,
            learning_rate: 1e-4,
            freeze_backbone: false,
            novel_view_loss_weight: 1.0,
            zeroverse: ZeroverseDataConfig::default(),
            brush: BrushRenderConfig::default(),
        }
    }
}

impl YonoFineTuneConfig {
    /// Validates the config for basic bounds and finite numeric values.
    pub fn validate(&self) -> Result<(), ZeroverseError> {
        if self.steps_per_epoch == 0 {
            return Err(ZeroverseError::InvalidConfig(
                "steps_per_epoch must be > 0".into(),
            ));
        }
        if self.batch_size == 0 {
            return Err(ZeroverseError::InvalidConfig(
                "batch_size must be > 0".into(),
            ));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(ZeroverseError::InvalidConfig(
                "learning_rate must be finite and > 0".into(),
            ));
        }
        if !self.novel_view_loss_weight.is_finite() || self.novel_view_loss_weight < 0.0 {
            return Err(ZeroverseError::InvalidConfig(
                "novel_view_loss_weight must be finite and >= 0".into(),
            ));
        }

        let zv = &self.zeroverse;
        if zv.scenes_per_step == 0 {
            return Err(ZeroverseError::InvalidConfig(
                "zeroverse.scenes_per_step must be > 0".into(),
            ));
        }
        if zv.conditioning_views_per_scene == 0 {
            return Err(ZeroverseError::InvalidConfig(
                "zeroverse.conditioning_views_per_scene must be > 0".into(),
            ));
        }
        if zv.target_views_per_scene == 0 {
            return Err(ZeroverseError::InvalidConfig(
                "zeroverse.target_views_per_scene must be > 0".into(),
            ));
        }
        if zv.image_size == 0 {
            return Err(ZeroverseError::InvalidConfig(
                "zeroverse.image_size must be > 0".into(),
            ));
        }
        if !zv.novel_view_ratio.is_finite() || !(0.0..=1.0).contains(&zv.novel_view_ratio) {
            return Err(ZeroverseError::InvalidConfig(
                "zeroverse.novel_view_ratio must be finite and in [0, 1]".into(),
            ));
        }

        let brush = &self.brush;
        if brush.max_gaussians == 0 {
            return Err(ZeroverseError::InvalidConfig(
                "brush.max_gaussians must be > 0".into(),
            ));
        }
        if !brush.opacity_threshold.is_finite() || !(0.0..=1.0).contains(&brush.opacity_threshold) {
            return Err(ZeroverseError::InvalidConfig(
                "brush.opacity_threshold must be finite and in [0, 1]".into(),
            ));
        }
        if !brush.photometric_loss_weight.is_finite() || brush.photometric_loss_weight < 0.0 {
            return Err(ZeroverseError::InvalidConfig(
                "brush.photometric_loss_weight must be finite and >= 0".into(),
            ));
        }
        if !brush.regularization_loss_weight.is_finite() || brush.regularization_loss_weight < 0.0 {
            return Err(ZeroverseError::InvalidConfig(
                "brush.regularization_loss_weight must be finite and >= 0".into(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::YonoFineTuneConfig;

    #[test]
    fn default_config_is_valid() {
        let cfg = YonoFineTuneConfig::default();
        cfg.validate().expect("default config should validate");
    }

    #[test]
    fn rejects_invalid_steps() {
        let cfg = YonoFineTuneConfig {
            steps_per_epoch: 0,
            ..YonoFineTuneConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_out_of_range_ratio() {
        let mut cfg = YonoFineTuneConfig::default();
        cfg.zeroverse.novel_view_ratio = 1.1;
        assert!(cfg.validate().is_err());
    }
}
