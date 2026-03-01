use crate::{config::YonoFineTuneConfig, error::ZeroverseError, workflow::SyntheticBatch};

/// Integration status for external runtime adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrationStatus {
    StubPending,
}

/// Placeholder adapter boundary for `bevy_zeroverse` synthetic scene sampling.
#[derive(Debug, Default)]
pub struct BevyZeroverseAdapter;

impl BevyZeroverseAdapter {
    pub const fn status(&self) -> IntegrationStatus {
        IntegrationStatus::StubPending
    }

    pub fn next_synthetic_batch(
        &mut self,
        _cfg: &YonoFineTuneConfig,
    ) -> Result<SyntheticBatch, ZeroverseError> {
        Err(ZeroverseError::IntegrationUnavailable(
            "bevy_zeroverse adapter is stub-only in this crate revision".into(),
        ))
    }
}

/// Placeholder adapter boundary for `brush` differentiable gaussian render/loss.
#[derive(Debug, Default)]
pub struct BrushDifferentiableAdapter;

impl BrushDifferentiableAdapter {
    pub const fn status(&self) -> IntegrationStatus {
        IntegrationStatus::StubPending
    }

    pub fn render_step_loss(
        &mut self,
        _batch: &SyntheticBatch,
        _cfg: &YonoFineTuneConfig,
    ) -> Result<f32, ZeroverseError> {
        Err(ZeroverseError::IntegrationUnavailable(
            "brush adapter is stub-only in this crate revision".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{BevyZeroverseAdapter, BrushDifferentiableAdapter, IntegrationStatus};

    #[test]
    fn adapters_are_stubbed() {
        assert_eq!(
            BevyZeroverseAdapter.status(),
            IntegrationStatus::StubPending
        );
        assert_eq!(
            BrushDifferentiableAdapter.status(),
            IntegrationStatus::StubPending
        );
    }
}
