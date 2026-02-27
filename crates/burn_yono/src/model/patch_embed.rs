use burn::{module::Initializer, prelude::*};
use burn_dino::layers::layer_norm::{LayerNorm, LayerNormConfig};

#[derive(Config, Debug)]
pub struct PatchEmbedNormConfig {
    pub patch_size: usize,
    pub in_chans: usize,
    pub embed_dim: usize,
    #[config(default = true)]
    pub use_norm: bool,
}

impl PatchEmbedNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PatchEmbedNorm<B> {
        PatchEmbedNorm::new(device, self)
    }
}

#[derive(Module, Debug)]
pub struct PatchEmbedNorm<B: Backend> {
    pub proj: nn::conv::Conv2d<B>,
    pub norm: Option<LayerNorm<B>>,
}

impl<B: Backend> PatchEmbedNorm<B> {
    pub fn new(device: &B::Device, config: &PatchEmbedNormConfig) -> Self {
        let kernel = [config.patch_size, config.patch_size];
        let proj = nn::conv::Conv2dConfig::new([config.in_chans, config.embed_dim], kernel)
            .with_stride(kernel)
            .with_initializer(Initializer::Zeros)
            .init(device);

        let norm = if config.use_norm {
            Some(LayerNormConfig::new(config.embed_dim).init(device))
        } else {
            None
        };

        Self { proj, norm }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let x = self.proj.forward(x).flatten(2, 3).swap_dims(1, 2);
        if let Some(norm) = &self.norm {
            norm.forward(x)
        } else {
            x
        }
    }
}
