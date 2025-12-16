// CIFAR-10 images are 32x32 pixels

use burn::{nn::conv::Conv2d, prelude::*};

const IMAGE_WH: usize = 32;
const PATCH_WH: usize = 8;
const SPLIT_ROWS: usize = IMAGE_WH / PATCH_WH;
const SPLIT_COLS: usize = IMAGE_WH / PATCH_WH;

#[derive(Module, Debug)]
pub struct Vit<B: Backend> {
    sample_conv: Conv2d<B>,
}

impl<B: Backend> Vit<B> {
    fn patchfy(&self, tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        let rows_patches = tensor.chunk(SPLIT_ROWS, 2);
        let horizontal: Tensor<B, 4> = Tensor::stack(rows_patches, 1);
        let cols_patches = horizontal.chunk(SPLIT_COLS, 4);
        let vertical = Tensor::stack(cols_patches, 1);

        vertical
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.patchfy(input);
        // flattenする次元がわからない...
        let x = x.flatten(2, 3);
        x
    }
}
