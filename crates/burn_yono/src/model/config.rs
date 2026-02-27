use burn::config::Config;

#[derive(Config, Debug)]
pub struct OpacityMappingConfig {
    #[config(default = 0.0)]
    pub initial: f32,
    #[config(default = 0.0)]
    pub final_value: f32,
    #[config(default = 1)]
    pub warm_up: usize,
}

#[derive(Config, Debug)]
pub struct UnifiedGaussianAdapterConfig {
    #[config(default = 0)]
    pub sh_degree: usize,
}

#[derive(Config, Debug)]
pub struct TransformerDecoderSpec {
    #[config(default = 2048)]
    pub in_dim: usize,
    #[config(default = 1024)]
    pub out_dim: usize,
    #[config(default = 1024)]
    pub embed_dim: usize,
    #[config(default = 5)]
    pub depth: usize,
    #[config(default = 16)]
    pub num_heads: usize,
    #[config(default = 4.0)]
    pub mlp_ratio: f32,
    #[config(default = 100.0)]
    pub rope_frequency: f32,
    #[config(default = true)]
    pub need_project: bool,
    #[config(default = false)]
    pub qk_norm: bool,
}

#[derive(Config, Debug)]
pub struct YonoHeadConfig {
    #[config(default = 14)]
    pub patch_size: usize,
    #[config(default = 14)]
    pub gaussians_per_axis: usize,
    #[config(default = 2)]
    pub upscale_token_ratio: usize,
    #[config(default = 1)]
    pub gaussian_downsample_ratio: usize,
    #[config(default = 1)]
    pub num_surfaces: usize,
    #[config(default = true)]
    pub pose_free: bool,
    #[config(default = 1000)]
    pub gt_pose_sampling_decay_start_step: usize,
    #[config(default = 5000)]
    pub gt_pose_sampling_decay_end_step: usize,
    #[config(default = 0.9)]
    pub gt_pose_final_sample_ratio: f32,
    #[config(default = "OpacityMappingConfig::new()")]
    pub opacity_mapping: OpacityMappingConfig,
    #[config(default = "UnifiedGaussianAdapterConfig::new()")]
    pub gaussian_adapter: UnifiedGaussianAdapterConfig,
    #[config(default = "TransformerDecoderSpec::new()")]
    pub point_decoder: TransformerDecoderSpec,
    #[config(default = "TransformerDecoderSpec::new()")]
    pub gaussian_decoder: TransformerDecoderSpec,
    #[config(default = "TransformerDecoderSpec::new().with_out_dim(512)")]
    pub camera_decoder: TransformerDecoderSpec,
    #[config(default = true)]
    pub share_point_decoder_init: bool,
}

impl Default for YonoHeadConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl YonoHeadConfig {
    pub fn raw_gaussian_dim(&self) -> usize {
        1 + self.gaussian_adapter_dim_in()
    }

    pub fn gaussian_adapter_dim_in(&self) -> usize {
        7 + 3 * self.d_sh()
    }

    pub fn d_sh(&self) -> usize {
        (self.gaussian_adapter.sh_degree + 1).pow(2)
    }

    pub fn point_patch_size(&self) -> usize {
        self.patch_size / self.upscale_token_ratio
    }

    pub fn effective_gaussians_per_axis(&self) -> usize {
        let limit = self.patch_size / self.gaussian_downsample_ratio.max(1);
        self.gaussians_per_axis.min(limit.max(1))
    }

    pub fn point_per_axis(&self) -> usize {
        self.effective_gaussians_per_axis() / self.upscale_token_ratio
    }

    pub fn scheduled_pose_epsilon(&self, global_step: usize) -> f32 {
        let start = self.gt_pose_sampling_decay_start_step;
        let end = self.gt_pose_sampling_decay_end_step;
        let epsilon_start = 1.0_f32;
        let epsilon_end = self.gt_pose_final_sample_ratio.clamp(0.0, 1.0);

        if start >= end {
            return epsilon_end;
        }
        if global_step < start {
            return epsilon_start;
        }
        if global_step >= end {
            return epsilon_end;
        }

        let steps_into_decay = (global_step - start) as f32;
        let total_decay_steps = (end - start) as f32;
        let decay_fraction = steps_into_decay / total_decay_steps;

        epsilon_start - (epsilon_start - epsilon_end) * decay_fraction
    }
}

#[cfg(test)]
mod tests {
    use super::YonoHeadConfig;

    #[test]
    fn gaussians_per_axis_is_clamped_to_patch_limit() {
        let cfg = YonoHeadConfig::new()
            .with_patch_size(14)
            .with_gaussian_downsample_ratio(1)
            .with_gaussians_per_axis(32)
            .with_upscale_token_ratio(2);

        assert_eq!(cfg.effective_gaussians_per_axis(), 14);
        assert_eq!(cfg.point_per_axis(), 7);
    }

    #[test]
    fn scheduled_pose_epsilon_follows_linear_decay() {
        let cfg = YonoHeadConfig::new()
            .with_gt_pose_sampling_decay_start_step(1000)
            .with_gt_pose_sampling_decay_end_step(5000)
            .with_gt_pose_final_sample_ratio(0.9);

        assert!((cfg.scheduled_pose_epsilon(500) - 1.0).abs() < 1e-6);
        assert!((cfg.scheduled_pose_epsilon(1000) - 1.0).abs() < 1e-6);
        assert!((cfg.scheduled_pose_epsilon(3000) - 0.95).abs() < 1e-6);
        assert!((cfg.scheduled_pose_epsilon(5000) - 0.9).abs() < 1e-6);
        assert!((cfg.scheduled_pose_epsilon(6000) - 0.9).abs() < 1e-6);
    }
}
