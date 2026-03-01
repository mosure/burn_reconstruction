use burn::{module::Ignored, prelude::*};

use super::config::UnifiedGaussianAdapterConfig;

#[derive(Debug)]
pub struct StructuredGaussians<B: Backend> {
    pub means: Tensor<B, 5>,
    pub covariances: Tensor<B, 6>,
    pub harmonics: Tensor<B, 6>,
    pub opacities: Tensor<B, 4>,
    pub rotations: Tensor<B, 5>,
    pub scales: Tensor<B, 5>,
}

#[derive(Debug)]
pub struct StructuredGaussiansSpp<B: Backend> {
    pub means: Tensor<B, 6>,
    pub covariances: Tensor<B, 6>,
    pub harmonics: Tensor<B, 6>,
    pub opacities: Tensor<B, 5>,
    pub rotations: Tensor<B, 6>,
    pub scales: Tensor<B, 6>,
}

#[derive(Debug, Clone)]
pub struct FlatGaussians<B: Backend> {
    pub means: Tensor<B, 3>,
    pub covariances: Tensor<B, 4>,
    pub harmonics: Tensor<B, 4>,
    pub opacities: Tensor<B, 2>,
    pub rotations: Tensor<B, 3>,
    pub scales: Tensor<B, 3>,
}

#[derive(Module, Debug)]
pub struct UnifiedGaussianAdapter<B: Backend> {
    pub sh_mask: Tensor<B, 1>,
    d_sh: Ignored<usize>,
}

impl<B: Backend> UnifiedGaussianAdapter<B> {
    pub fn new(device: &B::Device, cfg: &UnifiedGaussianAdapterConfig) -> Self {
        let d_sh = (cfg.sh_degree + 1).pow(2);
        let mut sh_mask = vec![1.0f32; d_sh];
        for degree in 1..=cfg.sh_degree {
            let start = degree * degree;
            let end = (degree + 1) * (degree + 1);
            let weight = 0.1f32 * 0.25f32.powi(degree as i32);
            for value in sh_mask.iter_mut().take(end).skip(start) {
                *value = weight;
            }
        }

        Self {
            sh_mask: Tensor::<B, 1>::from_floats(sh_mask.as_slice(), device),
            d_sh: Ignored(d_sh),
        }
    }

    pub fn d_sh(&self) -> usize {
        self.d_sh.0
    }

    pub fn d_in(&self) -> usize {
        7 + 3 * self.d_sh()
    }

    pub fn forward(
        &self,
        means: Tensor<B, 5>,
        _depths: Tensor<B, 5>,
        opacities: Tensor<B, 4>,
        raw_gaussians: Tensor<B, 5>,
        extrinsics: Option<Tensor<B, 6>>,
    ) -> StructuredGaussians<B> {
        let [b, v, r, srf, _] = means.shape().dims::<5>();
        let gaussians = self.forward_spp(
            means.unsqueeze_dim(4),
            _depths.unsqueeze_dim(4),
            opacities.unsqueeze_dim(4),
            raw_gaussians.unsqueeze_dim(4),
            extrinsics.map(|t| {
                t.reshape([b as i32, v as i32, r as i32, srf as i32, 16])
                    .unsqueeze_dim(4)
            }),
        );
        squeeze_spp_structured_gaussians(gaussians)
    }

    pub fn forward_spp(
        &self,
        means: Tensor<B, 6>,
        _depths: Tensor<B, 6>,
        opacities: Tensor<B, 5>,
        raw_gaussians: Tensor<B, 6>,
        extrinsics: Option<Tensor<B, 6>>,
    ) -> StructuredGaussiansSpp<B> {
        let [b, v, r, srf, spp, _] = means.shape().dims::<6>();
        let count = b * v * r * srf * spp;

        let means_flat = means.reshape([count as i32, 3]);
        let raw_flat = raw_gaussians.reshape([count as i32, -1]);
        let opacities_flat = opacities.reshape([count as i32, 1]);

        let feature_count = raw_flat.shape().dims::<2>()[1];
        let d_sh = self.d_sh();

        let scales = raw_flat.clone().slice([0..count as i32, 0..3]);
        let rotations = raw_flat.clone().slice([0..count as i32, 3..7]);
        let sh = raw_flat.slice([0..count as i32, 7..feature_count as i32]);

        let scales = softplus_pytorch(scales, 1.0, 20.0)
            .mul_scalar(0.001)
            .clamp_max(0.3);

        let rot_norm = rotations
            .clone()
            .powi_scalar(2)
            .sum_dim(1)
            .sqrt()
            .clamp_min(1e-8);
        let rotations = rotations / rot_norm;

        let sh = sh.reshape([count as i32, 3, d_sh as i32]);
        let sh = sh * self.sh_mask.clone().reshape([1, 1, d_sh as i32]);
        let sh = sh.reshape([count as i32, (3 * d_sh) as i32]);

        let mut covariances =
            build_covariance_flat(scales.clone(), rotations.clone()).reshape([count as i32, 9]);
        let mut means = means_flat;

        if let Some(extrinsics) = extrinsics {
            let extrinsics = extrinsics
                .reshape([count as i32, 16])
                .reshape([count as i32, 4, 4]);
            let c2w = extrinsics.clone().slice([0..count as i32, 0..3, 0..3]);
            let covariances_m = covariances.clone().reshape([count as i32, 3, 3]);
            covariances = c2w
                .clone()
                .matmul(covariances_m)
                .matmul(c2w.clone().swap_dims(1, 2))
                .reshape([count as i32, 9]);

            let ones = Tensor::<B, 2>::ones([count as i32, 1], &c2w.device());
            let means_h = Tensor::cat(vec![means, ones], 1).reshape([count as i32, 4, 1]);
            means = extrinsics
                .matmul(means_h)
                .reshape([count as i32, 4])
                .slice([0..count as i32, 0..3]);
        }

        StructuredGaussiansSpp {
            means: means.reshape([b as i32, v as i32, r as i32, srf as i32, spp as i32, 3]),
            covariances: covariances
                .reshape([b as i32, v as i32, r as i32, srf as i32, spp as i32, 9]),
            harmonics: sh.reshape([
                b as i32,
                v as i32,
                r as i32,
                srf as i32,
                spp as i32,
                (3 * d_sh) as i32,
            ]),
            opacities: opacities_flat
                .reshape([b as i32, v as i32, r as i32, srf as i32, spp as i32]),
            rotations: rotations.reshape([b as i32, v as i32, r as i32, srf as i32, spp as i32, 4]),
            scales: scales.reshape([b as i32, v as i32, r as i32, srf as i32, spp as i32, 3]),
        }
    }
}

fn softplus_pytorch<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    beta: f32,
    threshold: f32,
) -> Tensor<B, D> {
    let beta_tensor = tensor.clone().mul_scalar(beta);
    let smooth = (beta_tensor.clone().exp() + 1.0).log().div_scalar(beta);
    let linear_mask = beta_tensor.greater_elem(threshold);
    smooth.mask_where(linear_mask, tensor)
}

pub fn build_covariance_flat<B: Backend>(
    scales: Tensor<B, 2>,
    rotations_xyzw: Tensor<B, 2>,
) -> Tensor<B, 3> {
    let rotation = quaternion_to_matrix_flat(rotations_xyzw);

    // Equivalent to R * diag(scale^2) * R^T.
    let rs = rotation * scales.unsqueeze_dim(1);
    rs.clone().matmul(rs.swap_dims(1, 2))
}

pub fn quaternion_to_matrix_flat<B: Backend>(quaternions: Tensor<B, 2>) -> Tensor<B, 3> {
    let [n, _] = quaternions.shape().dims::<2>();

    let i = quaternions.clone().slice([0..n as i32, 0..1]);
    let j = quaternions.clone().slice([0..n as i32, 1..2]);
    let k = quaternions.clone().slice([0..n as i32, 2..3]);
    let r = quaternions.clone().slice([0..n as i32, 3..4]);

    let two_s = quaternions
        .clone()
        .powi_scalar(2)
        .sum_dim(1)
        .clamp_min(1e-8)
        .recip()
        .mul_scalar(2.0);

    let o0 = Tensor::<B, 2>::ones([n as i32, 1], &quaternions.device())
        - two_s.clone() * (j.clone().powi_scalar(2) + k.clone().powi_scalar(2));
    let o1 = two_s.clone() * (i.clone() * j.clone() - k.clone() * r.clone());
    let o2 = two_s.clone() * (i.clone() * k.clone() + j.clone() * r.clone());

    let o3 = two_s.clone() * (i.clone() * j.clone() + k.clone() * r.clone());
    let o4 = Tensor::<B, 2>::ones([n as i32, 1], &quaternions.device())
        - two_s.clone() * (i.clone().powi_scalar(2) + k.clone().powi_scalar(2));
    let o5 = two_s.clone() * (j.clone() * k.clone() - i.clone() * r.clone());

    let o6 = two_s.clone() * (i.clone() * k.clone() - j.clone() * r.clone());
    let o7 = two_s.clone() * (j.clone() * k.clone() + i.clone() * r.clone());
    let o8 = Tensor::<B, 2>::ones([n as i32, 1], &quaternions.device())
        - two_s * (i.clone().powi_scalar(2) + j.clone().powi_scalar(2));

    Tensor::cat(vec![o0, o1, o2, o3, o4, o5, o6, o7, o8], 1).reshape([n as i32, 3, 3])
}

pub fn flatten_structured_gaussians<B: Backend>(
    gaussians: StructuredGaussians<B>,
) -> FlatGaussians<B> {
    let [b, v, r, srf, _] = gaussians.means.shape().dims::<5>();
    let count = v * r * srf;
    let d_sh = gaussians.harmonics.shape().dims::<6>()[5];

    FlatGaussians {
        means: gaussians.means.reshape([b as i32, count as i32, 3]),
        covariances: gaussians
            .covariances
            .reshape([b as i32, count as i32, 3, 3]),
        harmonics: gaussians
            .harmonics
            .reshape([b as i32, count as i32, 3, d_sh as i32]),
        opacities: gaussians.opacities.reshape([b as i32, count as i32]),
        rotations: gaussians.rotations.reshape([b as i32, count as i32, 4]),
        scales: gaussians.scales.reshape([b as i32, count as i32, 3]),
    }
}

pub fn squeeze_spp_structured_gaussians<B: Backend>(
    gaussians: StructuredGaussiansSpp<B>,
) -> StructuredGaussians<B> {
    let [b, v, r, srf, spp, _] = gaussians.means.shape().dims::<6>();
    assert_eq!(
        spp, 1,
        "squeeze_spp_structured_gaussians expects spp=1, got {spp}"
    );
    let d_sh = gaussians.harmonics.shape().dims::<6>()[5] / 3;

    StructuredGaussians {
        means: gaussians.means.squeeze_dim(4),
        covariances: gaussians
            .covariances
            .reshape([b as i32, v as i32, r as i32, srf as i32, 3, 3]),
        harmonics: gaussians.harmonics.reshape([
            b as i32,
            v as i32,
            r as i32,
            srf as i32,
            3,
            d_sh as i32,
        ]),
        opacities: gaussians.opacities.squeeze_dim(4),
        rotations: gaussians.rotations.squeeze_dim(4),
        scales: gaussians.scales.squeeze_dim(4),
    }
}

pub fn flatten_structured_gaussians_spp<B: Backend>(
    gaussians: StructuredGaussiansSpp<B>,
) -> FlatGaussians<B> {
    let [b, v, r, srf, spp, _] = gaussians.means.shape().dims::<6>();
    let count = v * r * srf * spp;
    let d_sh = gaussians.harmonics.shape().dims::<6>()[5] / 3;

    FlatGaussians {
        means: gaussians.means.reshape([b as i32, count as i32, 3]),
        covariances: gaussians
            .covariances
            .reshape([b as i32, count as i32, 3, 3]),
        harmonics: gaussians
            .harmonics
            .reshape([b as i32, count as i32, 3, d_sh as i32]),
        opacities: gaussians.opacities.reshape([b as i32, count as i32]),
        rotations: gaussians.rotations.reshape([b as i32, count as i32, 4]),
        scales: gaussians.scales.reshape([b as i32, count as i32, 3]),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        flatten_structured_gaussians, flatten_structured_gaussians_spp,
        squeeze_spp_structured_gaussians, UnifiedGaussianAdapter,
    };
    use crate::model::config::UnifiedGaussianAdapterConfig;
    use burn::{prelude::*, tensor::Tensor};

    type TestBackend = burn::backend::NdArray<f32>;

    #[test]
    fn spp_structured_and_collapsed_paths_match_for_singleton_spp() {
        let device = <TestBackend as Backend>::Device::default();
        let adapter = UnifiedGaussianAdapter::new(&device, &UnifiedGaussianAdapterConfig::new());

        let b = 1usize;
        let v = 2usize;
        let r = 3usize;
        let srf = 1usize;
        let spp = 1usize;
        let count = b * v * r * srf * spp;

        let means = Tensor::<TestBackend, 1>::from_floats(
            (0..count * 3)
                .map(|idx| (idx as f32) * 0.001)
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        )
        .reshape([b as i32, v as i32, r as i32, srf as i32, spp as i32, 3]);

        let depths = means.clone().slice([
            0..b as i32,
            0..v as i32,
            0..r as i32,
            0..srf as i32,
            0..spp as i32,
            2..3,
        ]);

        let opacities =
            Tensor::<TestBackend, 1>::from_floats(vec![0.5f32; count].as_slice(), &device)
                .reshape([b as i32, v as i32, r as i32, srf as i32, spp as i32]);

        let raw = Tensor::<TestBackend, 1>::from_floats(
            (0..count * adapter.d_in())
                .map(|idx| ((idx % 17) as f32 - 8.0) / 8.0)
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        )
        .reshape([
            b as i32,
            v as i32,
            r as i32,
            srf as i32,
            spp as i32,
            adapter.d_in() as i32,
        ]);

        let extrinsics = Tensor::<TestBackend, 1>::from_floats(
            [
                1.0f32, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0, //
            ]
            .repeat(count)
            .as_slice(),
            &device,
        )
        .reshape([b as i32, v as i32, r as i32, srf as i32, spp as i32, 16]);

        let spp_out = adapter.forward_spp(
            means.clone(),
            depths,
            opacities.clone(),
            raw.clone(),
            Some(extrinsics),
        );
        let collapsed = squeeze_spp_structured_gaussians(spp_out);

        let direct = adapter.forward(
            means.clone().squeeze_dim(4),
            means
                .clone()
                .slice([
                    0..b as i32,
                    0..v as i32,
                    0..r as i32,
                    0..srf as i32,
                    0..1,
                    2..3,
                ])
                .squeeze_dim(4),
            opacities.clone().squeeze_dim(4),
            raw.clone().squeeze_dim(4),
            Some(
                Tensor::<TestBackend, 1>::from_floats(
                    [
                        1.0f32, 0.0, 0.0, 0.0, //
                        0.0, 1.0, 0.0, 0.0, //
                        0.0, 0.0, 1.0, 0.0, //
                        0.0, 0.0, 0.0, 1.0, //
                    ]
                    .repeat(count)
                    .as_slice(),
                    &device,
                )
                .reshape([b as i32, v as i32, r as i32, srf as i32, 4, 4]),
            ),
        );

        let flat_from_spp = flatten_structured_gaussians_spp(
            adapter.forward_spp(
                means.clone(),
                means.clone().slice([
                    0..b as i32,
                    0..v as i32,
                    0..r as i32,
                    0..srf as i32,
                    0..spp as i32,
                    2..3,
                ]),
                opacities.clone(),
                raw.clone(),
                Some(
                    Tensor::<TestBackend, 1>::from_floats(
                        [
                            1.0f32, 0.0, 0.0, 0.0, //
                            0.0, 1.0, 0.0, 0.0, //
                            0.0, 0.0, 1.0, 0.0, //
                            0.0, 0.0, 0.0, 1.0, //
                        ]
                        .repeat(count)
                        .as_slice(),
                        &device,
                    )
                    .reshape([b as i32, v as i32, r as i32, srf as i32, spp as i32, 16]),
                ),
            ),
        );
        let flat_collapsed = flatten_structured_gaussians(collapsed);
        let flat_direct = flatten_structured_gaussians(direct);

        let to_vec = |t: Tensor<TestBackend, 3>| {
            t.into_data()
                .to_vec::<f32>()
                .expect("tensor should be readable")
        };
        assert_eq!(
            to_vec(flat_from_spp.means),
            to_vec(flat_collapsed.means.clone())
        );
        assert_eq!(to_vec(flat_direct.means), to_vec(flat_collapsed.means));
    }
}
