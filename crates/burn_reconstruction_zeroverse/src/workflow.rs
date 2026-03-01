use crate::{config::YonoFineTuneConfig, error::ZeroverseError};

/// Minimal scene batch descriptor for synthetic multi-view training.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SyntheticBatch {
    pub scene_ids: Vec<String>,
    pub conditioning_views: usize,
    pub target_views: usize,
}

impl SyntheticBatch {
    pub fn total_views(&self) -> usize {
        self.conditioning_views + self.target_views
    }
}

/// Source boundary for synthetic scene batches (intended for `bevy_zeroverse`).
pub trait ZeroverseSceneProvider {
    type Batch;

    fn next_batch(&mut self, cfg: &YonoFineTuneConfig) -> Result<Self::Batch, ZeroverseError>;
}

/// Small trait to normalize scalar loss output types.
pub trait LossValue {
    fn scalar(&self) -> f32;
}

impl LossValue for f32 {
    fn scalar(&self) -> f32 {
        *self
    }
}

/// Differentiable renderer/loss bridge (intended for `brush`).
pub trait BrushDifferentiableRenderer<Batch> {
    type Loss: LossValue;

    fn render_loss(
        &mut self,
        batch: &Batch,
        cfg: &YonoFineTuneConfig,
    ) -> Result<Self::Loss, ZeroverseError>;
}

/// YoNo trainer boundary receiving synthetic batches and differentiable render loss.
pub trait YonoTrainer<Batch, Loss: LossValue> {
    fn train_step(
        &mut self,
        batch: &Batch,
        loss: &Loss,
        cfg: &YonoFineTuneConfig,
    ) -> Result<(), ZeroverseError>;
}

/// Per-step metrics emitted by the orchestration loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepMetrics {
    pub step_index: usize,
    pub render_loss: f32,
}

/// Per-epoch aggregate metrics emitted by the orchestration loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochMetrics {
    pub steps: usize,
    pub mean_render_loss: f32,
    pub max_render_loss: f32,
}

/// Coordinates synthetic data, differentiable rendering, and YoNo updates.
///
/// This is intentionally backend-agnostic and stub-friendly.
#[derive(Debug)]
pub struct ZeroverseFineTuneOrchestrator<P, R, T>
where
    P: ZeroverseSceneProvider,
    R: BrushDifferentiableRenderer<P::Batch>,
    T: YonoTrainer<P::Batch, R::Loss>,
{
    cfg: YonoFineTuneConfig,
    provider: P,
    renderer: R,
    trainer: T,
    last_step: Option<StepMetrics>,
}

impl<P, R, T> ZeroverseFineTuneOrchestrator<P, R, T>
where
    P: ZeroverseSceneProvider,
    R: BrushDifferentiableRenderer<P::Batch>,
    T: YonoTrainer<P::Batch, R::Loss>,
{
    pub fn new(cfg: YonoFineTuneConfig, provider: P, renderer: R, trainer: T) -> Self {
        Self {
            cfg,
            provider,
            renderer,
            trainer,
            last_step: None,
        }
    }

    pub fn config(&self) -> &YonoFineTuneConfig {
        &self.cfg
    }

    pub fn config_mut(&mut self) -> &mut YonoFineTuneConfig {
        &mut self.cfg
    }

    pub fn last_step(&self) -> Option<StepMetrics> {
        self.last_step
    }

    pub fn into_parts(self) -> (YonoFineTuneConfig, P, R, T) {
        (self.cfg, self.provider, self.renderer, self.trainer)
    }

    pub fn run_epoch(&mut self) -> Result<EpochMetrics, ZeroverseError> {
        self.cfg.validate()?;

        let mut sum = 0.0_f32;
        let mut max = 0.0_f32;
        let steps = self.cfg.steps_per_epoch;

        for step in 0..steps {
            let batch = self.provider.next_batch(&self.cfg)?;
            let loss = self.renderer.render_loss(&batch, &self.cfg)?;
            self.trainer.train_step(&batch, &loss, &self.cfg)?;

            let render_loss = loss.scalar();
            if step == 0 {
                max = render_loss;
            } else {
                max = max.max(render_loss);
            }
            sum += render_loss;

            self.last_step = Some(StepMetrics {
                step_index: step,
                render_loss,
            });
        }

        Ok(EpochMetrics {
            steps,
            mean_render_loss: sum / steps as f32,
            max_render_loss: max,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BrushDifferentiableRenderer, LossValue, YonoTrainer, ZeroverseFineTuneOrchestrator,
        ZeroverseSceneProvider,
    };
    use crate::{config::YonoFineTuneConfig, error::ZeroverseError, workflow::SyntheticBatch};

    #[derive(Debug)]
    struct MockProvider {
        calls: usize,
    }

    impl ZeroverseSceneProvider for MockProvider {
        type Batch = SyntheticBatch;

        fn next_batch(&mut self, _cfg: &YonoFineTuneConfig) -> Result<Self::Batch, ZeroverseError> {
            let id = self.calls;
            self.calls += 1;
            Ok(SyntheticBatch {
                scene_ids: vec![format!("scene_{id}")],
                conditioning_views: 2,
                target_views: 1,
            })
        }
    }

    #[derive(Debug)]
    struct MockRenderer {
        calls: usize,
    }

    impl BrushDifferentiableRenderer<SyntheticBatch> for MockRenderer {
        type Loss = f32;

        fn render_loss(
            &mut self,
            batch: &SyntheticBatch,
            _cfg: &YonoFineTuneConfig,
        ) -> Result<Self::Loss, ZeroverseError> {
            self.calls += 1;
            Ok(batch.total_views() as f32 + self.calls as f32 * 0.25)
        }
    }

    #[derive(Debug, Default)]
    struct MockTrainer {
        calls: usize,
        total_loss: f32,
    }

    impl YonoTrainer<SyntheticBatch, f32> for MockTrainer {
        fn train_step(
            &mut self,
            _batch: &SyntheticBatch,
            loss: &f32,
            _cfg: &YonoFineTuneConfig,
        ) -> Result<(), ZeroverseError> {
            self.calls += 1;
            self.total_loss += loss.scalar();
            Ok(())
        }
    }

    #[test]
    fn run_epoch_reports_loss_stats() {
        let cfg = YonoFineTuneConfig {
            steps_per_epoch: 3,
            ..YonoFineTuneConfig::default()
        };

        let provider = MockProvider { calls: 0 };
        let renderer = MockRenderer { calls: 0 };
        let trainer = MockTrainer::default();
        let mut orchestrator = ZeroverseFineTuneOrchestrator::new(cfg, provider, renderer, trainer);

        let metrics = orchestrator
            .run_epoch()
            .expect("mock epoch should run successfully");
        assert_eq!(metrics.steps, 3);
        assert!(metrics.mean_render_loss > 3.0);
        assert!(metrics.max_render_loss >= metrics.mean_render_loss);

        let last_step = orchestrator.last_step().expect("last step should exist");
        assert_eq!(last_step.step_index, 2);
        assert!(last_step.render_loss > 3.0);

        let (_cfg, provider, renderer, trainer) = orchestrator.into_parts();
        assert_eq!(provider.calls, 3);
        assert_eq!(renderer.calls, 3);
        assert_eq!(trainer.calls, 3);
        assert!(trainer.total_loss > 0.0);
    }

    #[test]
    fn run_epoch_fails_for_invalid_config() {
        let cfg = YonoFineTuneConfig {
            steps_per_epoch: 0,
            ..YonoFineTuneConfig::default()
        };

        let provider = MockProvider { calls: 0 };
        let renderer = MockRenderer { calls: 0 };
        let trainer = MockTrainer::default();
        let mut orchestrator = ZeroverseFineTuneOrchestrator::new(cfg, provider, renderer, trainer);

        let err = orchestrator
            .run_epoch()
            .expect_err("invalid config should fail");
        assert!(matches!(err, ZeroverseError::InvalidConfig(_)));
    }
}
