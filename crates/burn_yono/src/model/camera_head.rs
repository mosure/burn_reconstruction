use burn::{prelude::*, tensor::activation::relu};
use nalgebra::{Matrix3, Vector3};

#[derive(Module, Debug)]
pub struct ResConvBlock<B: Backend> {
    pub in_channels: usize,
    pub out_channels: usize,
    pub head_skip: Option<nn::Linear<B>>,
    pub res_conv1: nn::Linear<B>,
    pub res_conv2: nn::Linear<B>,
    pub res_conv3: nn::Linear<B>,
}

impl<B: Backend> ResConvBlock<B> {
    pub fn new(device: &B::Device, in_channels: usize, out_channels: usize) -> Self {
        let head_skip = if in_channels == out_channels {
            None
        } else {
            Some(nn::LinearConfig::new(in_channels, out_channels).init(device))
        };

        Self {
            in_channels,
            out_channels,
            head_skip,
            res_conv1: nn::LinearConfig::new(in_channels, out_channels).init(device),
            res_conv2: nn::LinearConfig::new(out_channels, out_channels).init(device),
            res_conv3: nn::LinearConfig::new(out_channels, out_channels).init(device),
        }
    }

    pub fn forward(&self, res: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = relu(self.res_conv1.forward(res.clone()));
        let x = relu(self.res_conv2.forward(x));
        let x = relu(self.res_conv3.forward(x));

        let skip = if let Some(head_skip) = &self.head_skip {
            head_skip.forward(res)
        } else {
            res
        };
        skip + x
    }
}

#[derive(Config, Debug)]
pub struct CameraHeadConfig {
    #[config(default = 512)]
    pub dim: usize,
}

#[derive(Module, Debug)]
pub struct CameraHead<B: Backend> {
    pub res_conv: Vec<ResConvBlock<B>>,
    pub mlp_0: nn::Linear<B>,
    pub mlp_1: nn::Linear<B>,
    pub fc_t: nn::Linear<B>,
    pub fc_rot: nn::Linear<B>,
}

impl<B: Backend> CameraHead<B> {
    pub fn new(device: &B::Device, cfg: &CameraHeadConfig) -> Self {
        let res_conv = vec![
            ResConvBlock::new(device, cfg.dim, cfg.dim),
            ResConvBlock::new(device, cfg.dim, cfg.dim),
        ];

        Self {
            res_conv,
            mlp_0: nn::LinearConfig::new(cfg.dim, cfg.dim).init(device),
            mlp_1: nn::LinearConfig::new(cfg.dim, cfg.dim).init(device),
            fc_t: nn::LinearConfig::new(cfg.dim, 3).init(device),
            fc_rot: nn::LinearConfig::new(cfg.dim, 9).init(device),
        }
    }

    pub fn forward_raw(
        &self,
        feat: Tensor<B, 3>,
        patch_h: usize,
        patch_w: usize,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [batch, _, channels] = feat.shape().dims::<3>();

        let mut feat = feat;
        for block in &self.res_conv {
            feat = block.forward(feat);
        }

        let feat = feat.swap_dims(1, 2).reshape([
            batch as i32,
            channels as i32,
            patch_h as i32,
            patch_w as i32,
        ]);

        let feat = feat
            .mean_dim(3)
            .mean_dim(2)
            .reshape([batch as i32, channels as i32]);
        let feat = relu(self.mlp_0.forward(feat));
        let feat = relu(self.mlp_1.forward(feat));

        let out_t = self.fc_t.forward(feat.clone());
        let out_r = self.fc_rot.forward(feat);

        (out_t, out_r)
    }

    pub fn forward_pose(&self, feat: Tensor<B, 3>, patch_h: usize, patch_w: usize) -> Tensor<B, 3> {
        let (out_t, out_r) = self.forward_raw(feat, patch_h, patch_w);
        self.convert_pose_to_4x4(out_r, out_t)
    }

    pub fn convert_pose_to_4x4(&self, out_r: Tensor<B, 2>, out_t: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, _] = out_t.shape().dims::<2>();
        let device = out_t.device();

        // Canonical YoNoSplat camera orthogonalization uses SVD. Burn 0.19 has
        // no tensor SVD op, so we do one packed readback (r|t) and map back.
        let packed = Tensor::cat(vec![out_r, out_t], 1);
        let values = packed
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("camera pose tensor must be readable");

        let mut pose_data = vec![0.0f32; batch * 16];

        for idx in 0..batch {
            let off = idx * 12;
            let rot = &values[off..off + 9];
            let trans = &values[off + 9..off + 12];

            let raw = Matrix3::from_row_slice(rot);
            let mut normalized = raw;
            for row in 0..3 {
                let row_vec = Vector3::new(
                    normalized[(row, 0)],
                    normalized[(row, 1)],
                    normalized[(row, 2)],
                );
                let norm = row_vec.norm().max(1e-8);
                normalized[(row, 0)] /= norm;
                normalized[(row, 1)] /= norm;
                normalized[(row, 2)] /= norm;
            }

            let m_transpose = normalized.transpose();
            let svd = m_transpose.svd(true, true);
            let u = svd.u.expect("svd must return U for 3x3 matrix");
            let v_t = svd.v_t.expect("svd must return Vt for 3x3 matrix");
            let v = v_t.transpose();

            let det = (v * u.transpose()).determinant();
            let mut v_fixed = v;
            v_fixed[(0, 2)] *= det;
            v_fixed[(1, 2)] *= det;
            v_fixed[(2, 2)] *= det;

            let r = v_fixed * u.transpose();

            let p_off = idx * 16;
            pose_data[p_off] = r[(0, 0)];
            pose_data[p_off + 1] = r[(0, 1)];
            pose_data[p_off + 2] = r[(0, 2)];
            pose_data[p_off + 3] = trans[0];

            pose_data[p_off + 4] = r[(1, 0)];
            pose_data[p_off + 5] = r[(1, 1)];
            pose_data[p_off + 6] = r[(1, 2)];
            pose_data[p_off + 7] = trans[1];

            pose_data[p_off + 8] = r[(2, 0)];
            pose_data[p_off + 9] = r[(2, 1)];
            pose_data[p_off + 10] = r[(2, 2)];
            pose_data[p_off + 11] = trans[2];

            pose_data[p_off + 12] = 0.0;
            pose_data[p_off + 13] = 0.0;
            pose_data[p_off + 14] = 0.0;
            pose_data[p_off + 15] = 1.0;
        }

        Tensor::<B, 1>::from_floats(pose_data.as_slice(), &device).reshape([batch as i32, 4, 4])
    }
}

#[cfg(test)]
mod tests {
    use super::{CameraHead, CameraHeadConfig};
    use burn::{prelude::*, tensor::Tensor};

    type TestBackend = burn::backend::NdArray<f32>;

    #[test]
    fn device_pose_head_returns_homogeneous_poses() {
        let device = <TestBackend as Backend>::Device::default();
        let head = CameraHead::new(&device, &CameraHeadConfig::new().with_dim(8));

        let batch = 2usize;
        let patch_h = 2usize;
        let patch_w = 2usize;
        let tokens = patch_h * patch_w;
        let channels = 8usize;

        let values = (0..batch * tokens * channels)
            .map(|idx| ((idx % 19) as f32 - 9.0) / 9.0)
            .collect::<Vec<_>>();
        let feat = Tensor::<TestBackend, 1>::from_floats(values.as_slice(), &device).reshape([
            batch as i32,
            tokens as i32,
            channels as i32,
        ]);

        let pose = head.forward_pose(feat, patch_h, patch_w);
        let bottom = pose
            .clone()
            .slice([0..batch as i32, 3..4, 0..4])
            .reshape([batch as i32, 4]);

        let bottom_expected = Tensor::<TestBackend, 1>::from_floats(
            [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0].as_slice(),
            &device,
        )
        .reshape([batch as i32, 4]);

        let bottom_diff = (bottom - bottom_expected)
            .abs()
            .into_data()
            .to_vec::<f32>()
            .expect("bottom row must be readable");
        let bottom_max = bottom_diff
            .into_iter()
            .fold(0.0f32, |acc, value| acc.max(value));
        assert!(bottom_max <= 1e-6, "bottom row mismatch: {bottom_max}");

        let rot = pose.slice([0..batch as i32, 0..3, 0..3]);
        let ortho = rot.clone().matmul(rot.swap_dims(1, 2));
        let eye = Tensor::<TestBackend, 1>::from_floats(
            [
                1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 1.0,
            ]
            .as_slice(),
            &device,
        )
        .reshape([batch as i32, 3, 3]);

        let ortho_diff = (ortho - eye)
            .abs()
            .into_data()
            .to_vec::<f32>()
            .expect("rotation matrix must be readable");
        let ortho_max = ortho_diff
            .into_iter()
            .fold(0.0f32, |acc, value| acc.max(value));
        assert!(
            ortho_max <= 1e-3,
            "rotation orthogonality mismatch: {ortho_max}"
        );
    }
}
