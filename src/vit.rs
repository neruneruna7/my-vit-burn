// CIFAR-10 images are 32x32 pixels

use burn::{
    module::{Initializer, Param},
    nn::{
        LayerNorm, LayerNormConfig, Linear, LinearConfig,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    prelude::*,
    tensor::{Distribution, backend::AutodiffBackend},
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
use tracing::debug;

use crate::cifar10_batcher::Cifar10Batch;
use crate::config::*;

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
        let patch_embedding = Linear {
            weight: Initializer::Normal {
                mean: 0.0,
                std: (1.0 / PATCH_VECTOR_LEN as f64).sqrt(),
            }
            .init_with(
                [PATCH_VECTOR_LEN, EMBED_VECTOR_LEN],
                Some(PATCH_VECTOR_LEN),
                Some(EMBED_VECTOR_LEN),
                device,
            ),
            bias: Some(Initializer::Zeros.init([EMBED_VECTOR_LEN], device)),
        };

        // Pram from_tensor ではなく， uninitializedを使えとの警告あり
        // paramIdがなにかわかってないので，ひとまずはこれで
        let cls: Param<Tensor<B, 3>> =
            Param::from_tensor(Tensor::zeros(Shape::new([1, 1, EMBED_VECTOR_LEN]), device));
        let position_embedding: Param<Tensor<B, 3>> = Param::from_tensor(Tensor::random(
            Shape::new([1, PATCH_TOTAL + 1, EMBED_VECTOR_LEN]),
            Distribution::Normal(0.0, POSITION_EMBEDDING_STD),
            device,
        ));
        // let encoder_layer = TransformerEncoderLayer
        let transformer_encoder =
            TransformerEncoderConfig::new(EMBED_VECTOR_LEN, DIM_FEEDFORWARD, HEAD, LAYERS)
                .with_dropout(DROPOUT)
                .with_norm_first(true);
        let layer_norm = LayerNormConfig::new(EMBED_VECTOR_LEN).with_epsilon(LAYER_NORM_EPSILON);
        let mlp_head = LinearConfig::new(EMBED_VECTOR_LEN, 10).with_initializer(Initializer::Zeros);

        Vit {
            patch_embedding,
            cls,
            position_embedding,
            transformer_encoder: transformer_encoder.init(device),
            layer_norm: layer_norm.init(device),
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
    layer_norm: LayerNorm<B>,
    mlp_head: Linear<B>,
}

impl<B: Backend> Vit<B> {
    /// 画像をパッチ化する
    /// パッチサイズは8x8で、32x32の画像からは16個のパッチができる
    /// 入力: [Batch, Channel, Height, Width] = [B, 3, 32, 32]
    /// 分割されて，[Batch, Num_Patches, Channels, Patch_H, Patch_W] = [B, 16, 3, 8, 8]
    /// 出力: [Batch, Num_Patches, Patch_Vector_Len] = [B, 16, 3*8*8=192]
    fn patchfy(&self, tensor: Tensor<B, 4>, split_rows: usize, split_cols: usize) -> Tensor<B, 3> {
        let rows_chunked = tensor.chunk(split_rows, 2);
        let rows: Tensor<B, 5> = Tensor::stack(rows_chunked, 1);
        let cols_chunked = rows.chunk(split_cols, 4);
        let grid: Tensor<B, 6> = Tensor::stack(cols_chunked, 2);
        let patches: Tensor<B, 5> = grid.flatten(1, 2);

        // pytorchでは，patches = patches.flatten(2) をしたい
        // shape [Batch, Num_Patches, Channels, Patch_H, Patch_W]
        // -> [Batch, Num_Patches, Channels * Patch_H * Patch_W]
        let patches: Tensor<B, 3> = patches.flatten(2, 4);
        patches
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        debug!("Input shape: {:?}", input.dims());
        let x = self.patchfy(input, SPLIT_ROWS, SPLIT_COLS);
        debug!("After patchfy shape: {:?}", x.dims());
        // 次元のインデックスは0スタート．2から4次元までを平坦化
        // 2つ次元が減るので，5から3次元になる
        let x = self.patch_embedding.forward(x);
        debug!("After patch embedding shape: {:?}", x.dims());
        let batch_size = x.dims()[0];
        let cls_token = self
            .cls
            .clone()
            .val()
            .expand([batch_size, 1, EMBED_VECTOR_LEN]);

        debug!("cls token shape: {:?}", cls_token.dims());

        let x = Tensor::cat(vec![cls_token, x], 1);
        debug!("After concatenating CLS token shape: {:?}", x.dims());
        let x = x + self.position_embedding.clone().val();
        debug!("After adding position embedding shape: {:?}", x.dims());
        let transformer_encoder_input = TransformerEncoderInput::new(x);
        let x = self.transformer_encoder.forward(transformer_encoder_input);
        debug!("After transformer encoder shape: {:?}", x.dims());
        let x = self.layer_norm.forward(x);
        debug!("After layer norm shape: {:?}", x.dims());

        // 1. サイズを取得
        let [batch_size, _seq_len, embed_dim] = x.dims();

        // 2. スライス: 0番目のトークン(CLS)だけを切り出す
        // Python: x[:, 0, :]
        // Burn: x.slice(...) -> Shapeは [batch, 1, embed] のまま維持される
        // バッチのデータすべて，0番目のトークン，すべての埋め込み次元を取得
        let cls_token = x.slice([0..batch_size, 0..1, 0..embed_dim]);

        // 3. 次元削除: [batch, 1, embed] -> [batch, embed]
        // 真ん中の「1」になっている次元(dim=1)をつぶす
        let cls_token: Tensor<B, 2> = cls_token.squeeze();
        debug!("After squeezing CLS token shape: {:?}", cls_token.dims());
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

// お，v20から関連型を使うようになったんだ
// いいね，扱いやすくなった．
// 型を2回書かなくてよくなった
impl<B: AutodiffBackend> TrainStep for Vit<B> {
    type Input = Cifar10Batch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let item = self.forward_classification(batch.images, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

// おそらく，v20からValidStepだったものがInferenceStepに命名変更された
//
impl<B: Backend> InferenceStep for Vit<B> {
    type Input = Cifar10Batch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Self::Input) -> Self::Output {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, Tolerance};

    // テスト用のバックエンド定義（プロジェクトの設定に合わせて変更してください）
    // 通常は burn::backend::ndarray::NdArray<f32> などを使います
    type TestBackend = burn::backend::ndarray::NdArray<f32>;

    fn repeat_interleave<B: Backend, const D: usize, const D2: usize>(
        tensor: Tensor<B, D>,
        repeats: usize,
        dim: usize,
    ) -> Tensor<B, D> {
        assert!(dim < D, "dim must be less than the tensor rank");
        assert_eq!(D2, D + 1, "D2 must be D + 1");

        let dims = tensor.dims();
        let mut expanded_dims = [0; D2];
        let mut source_index = 0;

        for (expanded_index, expanded_dim) in expanded_dims.iter_mut().enumerate() {
            if expanded_index == dim + 1 {
                *expanded_dim = repeats;
            } else {
                *expanded_dim = dims[source_index];
                source_index += 1;
            }
        }

        tensor
            .unsqueeze_dim::<D2>(dim + 1)
            .expand(expanded_dims)
            .flatten(dim, dim + 1)
    }

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
