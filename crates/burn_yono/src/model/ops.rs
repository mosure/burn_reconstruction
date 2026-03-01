use burn::prelude::*;

pub fn position_getter<B: Backend>(
    batch: usize,
    height: usize,
    width: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let mut data = Vec::with_capacity(batch * height * width * 2);
    for _ in 0..batch {
        for y in 0..height {
            for x in 0..width {
                data.push(y as f32);
                data.push(x as f32);
            }
        }
    }

    Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([
        batch as i32,
        (height * width) as i32,
        2,
    ])
}

pub fn pixel_shuffle<B: Backend>(x: Tensor<B, 4>, upscale_factor: usize) -> Tensor<B, 4> {
    let [b, c, h, w] = x.shape().dims::<4>();
    let r = upscale_factor;
    assert!(
        c % (r * r) == 0,
        "channels must be divisible by upscale_factor^2"
    );

    let out_c = c / (r * r);

    x.reshape([
        b as i32,
        out_c as i32,
        r as i32,
        r as i32,
        h as i32,
        w as i32,
    ])
    .permute([0, 1, 4, 2, 5, 3])
    .reshape([b as i32, out_c as i32, (h * r) as i32, (w * r) as i32])
}

pub fn interpolate_bilinear_align_corners_false<B: Backend>(
    x: Tensor<B, 4>,
    output_size: [usize; 2],
) -> Tensor<B, 4> {
    let [_, _, in_h, in_w] = x.shape().dims::<4>();
    let [out_h, out_w] = output_size;

    if in_h == out_h && in_w == out_w {
        return x;
    }

    let (y0_idx, y1_idx, wy0, wy1) = bilinear_axis_table(in_h, out_h);
    let (x0_idx, x1_idx, wx0, wx1) = bilinear_axis_table(in_w, out_w);

    let device = x.device();
    let y0 = Tensor::<B, 1, Int>::from_ints(y0_idx.as_slice(), &device);
    let y1 = Tensor::<B, 1, Int>::from_ints(y1_idx.as_slice(), &device);
    let x0 = Tensor::<B, 1, Int>::from_ints(x0_idx.as_slice(), &device);
    let x1 = Tensor::<B, 1, Int>::from_ints(x1_idx.as_slice(), &device);

    let wy0 = Tensor::<B, 1>::from_floats(wy0.as_slice(), &device).reshape([1, 1, out_h as i32, 1]);
    let wy1 = Tensor::<B, 1>::from_floats(wy1.as_slice(), &device).reshape([1, 1, out_h as i32, 1]);
    let wx0 = Tensor::<B, 1>::from_floats(wx0.as_slice(), &device).reshape([1, 1, 1, out_w as i32]);
    let wx1 = Tensor::<B, 1>::from_floats(wx1.as_slice(), &device).reshape([1, 1, 1, out_w as i32]);

    let y0_sample = x.clone().select(2, y0);
    let y1_sample = x.select(2, y1);
    let y_interp = y0_sample * wy0 + y1_sample * wy1;

    let x0_sample = y_interp.clone().select(3, x0);
    let x1_sample = y_interp.select(3, x1);
    x0_sample * wx0 + x1_sample * wx1
}

fn bilinear_axis_table(
    in_size: usize,
    out_size: usize,
) -> (Vec<i32>, Vec<i32>, Vec<f32>, Vec<f32>) {
    let mut idx0 = Vec::with_capacity(out_size);
    let mut idx1 = Vec::with_capacity(out_size);
    let mut w0 = Vec::with_capacity(out_size);
    let mut w1 = Vec::with_capacity(out_size);

    for o in 0..out_size {
        let src = ((o as f32 + 0.5) * in_size as f32 / out_size as f32 - 0.5)
            .clamp(0.0, (in_size.saturating_sub(1)) as f32);
        let i0 = src.floor() as i32;
        let i1 = (i0 + 1).min((in_size.saturating_sub(1)) as i32);
        let t = src - i0 as f32;
        idx0.push(i0);
        idx1.push(i1);
        w0.push(1.0 - t);
        w1.push(t);
    }

    (idx0, idx1, w0, w1)
}

pub fn se3_inverse_flat<B: Backend>(pose: Tensor<B, 3>) -> Tensor<B, 3> {
    let [n, _, _] = pose.shape().dims::<3>();

    let rot = pose.clone().slice([0..n as i32, 0..3, 0..3]);
    let trans = pose.clone().slice([0..n as i32, 0..3, 3..4]);

    let rot_inv = rot.clone().swap_dims(1, 2);
    let trans_inv = rot_inv.clone().matmul(trans).mul_scalar(-1.0);

    let device = pose.device();
    let bottom = Tensor::<B, 1>::from_floats([0.0f32, 0.0, 0.0, 1.0].as_slice(), &device)
        .reshape([1, 4])
        .repeat_dim(0, n);

    let top = Tensor::cat(vec![rot_inv, trans_inv], 2);
    Tensor::cat(vec![top, bottom.reshape([n as i32, 1, 4])], 1)
}

pub fn flatten_prefix<const D: usize, B: Backend>(
    tensor: Tensor<B, D>,
    keep_last: usize,
) -> (Tensor<B, 2>, Vec<usize>) {
    let dims = tensor.shape().dims;
    let mut prefix = Vec::with_capacity(D.saturating_sub(keep_last));
    for dim in dims.iter().take(D - keep_last) {
        prefix.push(*dim);
    }

    let outer = prefix.iter().product::<usize>();
    let inner = dims.iter().skip(D - keep_last).product::<usize>();

    (tensor.reshape([outer as i32, inner as i32]), prefix)
}

#[cfg(test)]
mod tests {
    use super::interpolate_bilinear_align_corners_false;
    use burn::{prelude::*, tensor::Tensor};

    type TestBackend = burn::backend::NdArray<f32>;

    #[test]
    fn bilinear_align_corners_false_matches_known_grid() {
        let device = <TestBackend as Backend>::Device::default();
        let input =
            Tensor::<TestBackend, 1>::from_floats([1.0f32, 2.0, 3.0, 4.0].as_slice(), &device)
                .reshape([1, 1, 2, 2]);
        let out = interpolate_bilinear_align_corners_false(input, [4, 4]);
        let got = out
            .into_data()
            .to_vec::<f32>()
            .expect("output must be readable");

        let expected = [
            1.0, 1.25, 1.75, 2.0, //
            1.5, 1.75, 2.25, 2.5, //
            2.5, 2.75, 3.25, 3.5, //
            3.0, 3.25, 3.75, 4.0, //
        ];

        assert_eq!(got.len(), expected.len());
        let max_abs = got
            .iter()
            .zip(expected.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e-6,
            "unexpected interpolation mismatch: max_abs={max_abs}"
        );
    }
}
