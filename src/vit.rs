// CIFAR-10 images are 32x32 pixels

use burn::{
    module::Param,
    nn::{
        Linear, LinearConfig,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::cifar10_batcher::Cifar10Batch;

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
    // pub num_layers: usize,
    // pub num_heads: usize,
    // pub hidden_dim: usize,
    // pub mlp_dim: usize,
    // pub dropout: f64,
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

        Tensor::cat(cols_patches, 1)
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

        self.mlp_head.forward(cls_token)
    }

    pub fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<Cifar10Batch<B>, ClassificationOutput<B>> for Vit<B> {
    fn step(&self, batch: Cifar10Batch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<Cifar10Batch<B>, ClassificationOutput<B>> for Vit<B> {
    fn step(&self, batch: Cifar10Batch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::*;
    use burn::tensor::{Tensor, Tolerance};

    // テスト用のバックエンド定義（プロジェクトの設定に合わせて変更してください）
    // 通常は burn::backend::ndarray::NdArray<f32> などを使います
    type TestBackend = burn::backend::ndarray::NdArray<f32>;

    #[test]
    fn test_repeat_interleave_1d() {
        // ケース1: 1次元テンソル [1, 2] を3回リピート -> [1, 1, 1, 2, 2, 2]
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_data([1.0, 2.0], &device);

        // D=1 なので D2=2 を指定する必要があります
        let output = repeat_interleave::<TestBackend, 1, 2>(tensor, 3, 0);

        let expected_data = TensorData::from([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        // 値の検証
        output
            .into_data()
            .assert_approx_eq::<f32>(&expected_data, Tolerance::default());
    }

    #[test]
    fn test_repeat_interleave_2d_dim0() {
        // ケース2: 2次元テンソル (dim=0, 行方向のリピート)
        // [[1, 2],
        //  [3, 4]]
        // ↓ repeats=2, dim=0
        // [[1, 2], [1, 2],
        //  [3, 4], [3, 4]]

        let device = Default::default();
        let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &device);

        // D=2 なので D2=3
        let output = repeat_interleave::<TestBackend, 2, 3>(tensor, 2, 0);

        let expected_data = TensorData::from([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]]);

        // シェイプの検証: [4, 2] になっているはず
        assert_eq!(output.dims(), [4, 2]);
        output
            .into_data()
            .assert_approx_eq::<f32>(&expected_data, Tolerance::default());
    }

    #[test]
    fn test_repeat_interleave_2d_dim1() {
        // ケース3: 2次元テンソル (dim=1, 列方向のリピート)
        // [[1, 2],
        //  [3, 4]]
        // ↓ repeats=2, dim=1
        // [[1, 1, 2, 2],
        //  [3, 3, 4, 4]]

        let device = Default::default();
        let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &device);

        // D=2 なので D2=3
        let output = repeat_interleave::<TestBackend, 2, 3>(tensor, 2, 1);

        let expected_data = TensorData::from([[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]]);

        // シェイプの検証: [2, 4] になっているはず
        assert_eq!(output.dims(), [2, 4]);
        output
            .into_data()
            .assert_approx_eq::<f32>(&expected_data, Tolerance::default());
    }
}
