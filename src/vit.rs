// CIFAR-10 images are 32x32 pixels

use burn::{
    module::Param,
    nn::{
        Linear, LinearConfig,
        conv::Conv2d,
        transformer::{
            TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
            TransformerEncoderLayer,
        },
    },
    prelude::*,
    tensor::ops::TransactionOps,
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

fn repeat_interleave<B: Backend, const D: usize, const D2: usize>(
    tensor: Tensor<B, D>,
    repeats: usize,
    dim: usize,
) -> Tensor<B, D> {
    // 1. 指定した次元の直後に新しい次元を追加 (Unsqueeze)
    // 例: shape [3, 5], dim=0 -> [3, 1, 5]
    let x: Tensor<B, D2> = tensor.unsqueeze_dim(dim + 1);

    // 2. repeat用のシェイプを作成
    // 元の次元数(D) + 追加した1次元 = D + 1
    // 基本はすべて1倍で、追加した次元だけ repeats倍にする
    let mut repeat_shape = vec![1; D + 1];
    repeat_shape[dim + 1] = repeats;

    // 3. 追加した次元方向にリピートし、その後元の次元とマージ (Flatten)
    // [3, 1, 5] -> repeat -> [3, 2, 5] -> flatten(0, 1) -> [6, 5]
    x.repeat(&repeat_shape).flatten(dim, dim + 1)
}

#[derive(Config, Debug)]
pub struct VitConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub hidden_dim: usize,
    pub mlp_dim: usize,
    pub dropout: f64,
}

impl VitConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Vit<B> {
        let patch_embedding = LinearConfig::new(PATCH_VECTOR_LEN, EMBED_VECTOR_LEN);
        // Pram from_tensor ではなく， uninitializedを使えとの警告あり
        // paramIdがなにかわかってないので，ひとまずはこれで
        let cls: Param<Tensor<B, 3>> =
            Param::from_tensor(Tensor::zeros(Shape::new([1, 1, EMBED_VECTOR_LEN]), device));
        let position_embedding: Param<Tensor<B, 3>> = Param::from_tensor(Tensor::zeros(
            Shape::new([1, PATCH_TOTAL + 1, EMBED_VECTOR_LEN]),
            device,
        ));
        // let encoder_layer = TransformerEncoderLayer
        let transformer_encoder =
            TransformerEncoderConfig::new(EMBED_VECTOR_LEN, DIM_FEEDFORWARD, HEAD, LAYERS)
                .with_norm_first(true);
        let mlp_head = LinearConfig::new(EMBED_VECTOR_LEN, 10);

        Vit {
            patch_embedding: patch_embedding.init(device),
            cls,
            position_embedding,
            transformer_encoder: transformer_encoder.init(device),
            mlp_head: mlp_head.init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Vit<B: Backend> {
    patch_embedding: Linear<B>,
    cls: Param<Tensor<B, 3>>,
    position_embedding: Param<Tensor<B, 3>>,
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

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.patchfy(input);
        // 次元のインデックスは0スタート．2から4次元までを平坦化
        // 2つ次元が減るので，5から3次元になる
        let x: Tensor<B, 3> = x.flatten(2, 4);
        let x = self.patch_embedding.forward(x);
        let cls_token = repeat_interleave::<B, 3, 4>(self.cls.clone().val(), x.shape().dims[0], 0);
        let x = Tensor::cat(vec![cls_token, x], 1);
        let x = x + self.position_embedding.clone().val();
        let transformer_encoder_input = TransformerEncoderInput::new(x);
        let x = self.transformer_encoder.forward(transformer_encoder_input);

        // 1. サイズを取得
        let [batch_size, _seq_len, embed_dim] = x.dims();

        // 2. スライス: 0番目のトークン(CLS)だけを切り出す
        // Python: x[:, 0, :]
        // Burn: x.slice(...) -> Shapeは [batch, 1, embed] のまま維持される
        let cls_token = x.slice([0..batch_size, 0..1, 0..embed_dim]);

        // 3. 次元削除: [batch, 1, embed] -> [batch, embed]
        // 真ん中の「1」になっている次元(dim=1)をつぶす
        let cls_token: Tensor<B, 2> = cls_token.squeeze();

        // 4. MLP Headに通して分類結果を出す
        // Output: [batch, 10]
        let x = self.mlp_head.forward(cls_token);

        x
    }
}
