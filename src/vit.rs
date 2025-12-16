// CIFAR-10 images are 32x32 pixels

use burn::{
    nn::{
        Linear, LinearConfig,
        conv::Conv2d,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderLayer},
    },
    prelude::*,
};

// image params
const IMAGE_WH: usize = 32;
const CHANNEL: usize = 3;

// vit hyper prams
const PATCH_WH: usize = 8;
const SPLIT_ROWS: usize = IMAGE_WH / PATCH_WH;
const SPLIT_COLS: usize = IMAGE_WH / PATCH_WH;
const PATCH_TOTAL: usize = (IMAGE_WH / PATCH_WH).pow(2);
const PATCH_VECTOR_LEN: usize = CHANNEL * PATCH_WH.pow(2);
const EMBED_VECTOR_LEN: usize = PATCH_VECTOR_LEN / 2;

// transformer params
const HEAD: usize = 12;
const DIM_FEEDFORWARD: usize = EMBED_VECTOR_LEN;
const ACTIVATION: &str = "gelu";
const LAYERS: usize = 12;

#[derive(Config, Debug)]
pub struct VitConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub hidden_dim: usize,
    pub mlp_dim: usize,
    pub dropout: f64,
}

impl VitConfig {
    pub fn init<B: Backend>(&self) -> Vit<B> {
        let patch_embedding = LinearConfig::new(PATCH_VECTOR_LEN, EMBED_VECTOR_LEN);
        // let encoder_layer = TransformerEncoderLayer
        let transformer_encoder =
            TransformerEncoderConfig::new(EMBED_VECTOR_LEN, DIM_FEEDFORWARD, HEAD, LAYERS)
                .with_norm_first(true);
        let mlp_head = LinearConfig::new(EMBED_VECTOR_LEN, 10);

        todo!()
    }
}

#[derive(Module, Debug)]
pub struct Vit<B: Backend> {
    patch_embedding: Linear<B>,
    // cls: Parameter
    // position_embedding:
    encoder_layer: TransformerEncoderLayer<B>,
    transformer_encoder: TransformerEncoder<B>,
    mlp_head: Linear<B>,
}

impl<B: Backend> Vit<B> {
    fn patchfy(&self, tensor: Tensor<B, 4>) -> Tensor<B, 5> {
        let rows_patches = tensor.chunk(SPLIT_ROWS, 2);
        let horizontal: Tensor<B, 5> = Tensor::stack(rows_patches, 1);
        let cols_patches = horizontal.chunk(SPLIT_COLS, 4);
        let vertical = Tensor::cat(cols_patches, 1);

        vertical
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.patchfy(input);
        // 次元のインデックスは0スタート．2から4次元までを平坦化
        // 2つ次元が減るので，5から3次元になる
        let x: Tensor<B, 3> = x.flatten(2, 4);
        let x = self.patch_embedding.forward(x);

        todo!()
    }
}
