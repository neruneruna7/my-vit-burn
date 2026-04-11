use burn::{
    module::Param,
    nn::{Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::*,
    tensor::{Distribution, activation::softmax},
};

#[derive(Config, Debug)]
pub struct FeedForwardConfig {
    pub dim: usize,
    pub hidden_dim: usize,
    #[config(default = 0.0)]
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    norm: LayerNorm<B>,
    linear_in: Linear<B>,
    gelu: Gelu,
    dropout_in: Dropout,
    linear_out: Linear<B>,
    dropout_out: Dropout,
}

impl FeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        FeedForward {
            norm: LayerNormConfig::new(self.dim).init(device),
            linear_in: LinearConfig::new(self.dim, self.hidden_dim).init(device),
            gelu: Gelu::new(),
            dropout_in: DropoutConfig::new(self.dropout).init(),
            linear_out: LinearConfig::new(self.hidden_dim, self.dim).init(device),
            dropout_out: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.norm.forward(input);
        let x = self.linear_in.forward(x);
        let x = self.gelu.forward(x);
        let x = self.dropout_in.forward(x);
        let x = self.linear_out.forward(x);

        self.dropout_out.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct AttentionConfig {
    pub dim: usize,
    #[config(default = 8)]
    pub heads: usize,
    #[config(default = 64)]
    pub dim_head: usize,
    #[config(default = 0.0)]
    pub dropout: f64,
}

#[derive(Module, Debug)]
struct AttentionProjection<B: Backend> {
    linear: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> AttentionProjection<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear.forward(input);
        self.dropout.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    norm: LayerNorm<B>,
    to_qkv: Linear<B>,
    attend_dropout: Dropout,
    to_out: Option<AttentionProjection<B>>,
    heads: usize,
    dim_head: usize,
    scale: f32,
}

impl AttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Attention<B> {
        assert!(self.heads > 0, "heads must be greater than zero");
        assert!(self.dim_head > 0, "dim_head must be greater than zero");

        let inner_dim = self.heads * self.dim_head;
        let project_out = !(self.heads == 1 && self.dim_head == self.dim);

        Attention {
            norm: LayerNormConfig::new(self.dim).init(device),
            to_qkv: LinearConfig::new(self.dim, inner_dim * 3)
                .with_bias(false)
                .init(device),
            attend_dropout: DropoutConfig::new(self.dropout).init(),
            to_out: project_out.then(|| AttentionProjection {
                linear: LinearConfig::new(inner_dim, self.dim).init(device),
                dropout: DropoutConfig::new(self.dropout).init(),
            }),
            heads: self.heads,
            dim_head: self.dim_head,
            scale: 1.0 / (self.dim_head as f32).sqrt(),
        }
    }
}

impl<B: Backend> Attention<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _dim] = input.dims();

        let x = self.norm.forward(input);
        let qkv = self.to_qkv.forward(x).chunk(3, 2);
        let mut qkv = qkv.into_iter();
        let q = qkv
            .next()
            .expect("qkv projection must produce a query tensor");
        let k = qkv
            .next()
            .expect("qkv projection must produce a key tensor");
        let v = qkv
            .next()
            .expect("qkv projection must produce a value tensor");

        let q = q
            .reshape([batch_size, seq_len, self.heads, self.dim_head])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch_size, seq_len, self.heads, self.dim_head])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch_size, seq_len, self.heads, self.dim_head])
            .swap_dims(1, 2);

        let dots = q.matmul(k.transpose()).mul_scalar(self.scale);
        let attn = softmax(dots, 3);
        let attn = self.attend_dropout.forward(attn);

        let out = attn.matmul(v);
        let out = out
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, self.heads * self.dim_head]);

        match &self.to_out {
            Some(to_out) => to_out.forward(out),
            None => out,
        }
    }
}

#[derive(Module, Debug)]
struct TransformerLayer<B: Backend> {
    attention: Attention<B>,
    feed_forward: FeedForward<B>,
}

#[derive(Config, Debug)]
pub struct TransformerConfig {
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub dim_head: usize,
    pub mlp_dim: usize,
    #[config(default = 0.0)]
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    norm: LayerNorm<B>,
    layers: Vec<TransformerLayer<B>>,
}

impl TransformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Transformer<B> {
        let layers = (0..self.depth)
            .map(|_| TransformerLayer {
                attention: AttentionConfig::new(self.dim)
                    .with_heads(self.heads)
                    .with_dim_head(self.dim_head)
                    .with_dropout(self.dropout)
                    .init(device),
                feed_forward: FeedForwardConfig::new(self.dim, self.mlp_dim)
                    .with_dropout(self.dropout)
                    .init(device),
            })
            .collect();

        Transformer {
            norm: LayerNormConfig::new(self.dim).init(device),
            layers,
        }
    }
}

impl<B: Backend> Transformer<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = input;

        for layer in &self.layers {
            x = layer.attention.forward(x.clone()) + x;
            x = layer.feed_forward.forward(x.clone()) + x;
        }

        self.norm.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct PatchEmbeddingConfig {
    pub image_size: [usize; 2],
    pub patch_size: [usize; 2],
    pub dim: usize,
    #[config(default = 3)]
    pub channels: usize,
}

#[derive(Module, Debug)]
pub struct PatchEmbedding<B: Backend> {
    patch_norm: LayerNorm<B>,
    patch_to_embedding: Linear<B>,
    embedding_norm: LayerNorm<B>,
    image_height: usize,
    image_width: usize,
    patch_height: usize,
    patch_width: usize,
    channels: usize,
    num_patch_rows: usize,
    num_patch_cols: usize,
    num_patches: usize,
}

impl PatchEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PatchEmbedding<B> {
        let [image_height, image_width] = self.image_size;
        let [patch_height, patch_width] = self.patch_size;

        assert!(patch_height > 0, "patch height must be greater than zero");
        assert!(patch_width > 0, "patch width must be greater than zero");
        assert!(
            image_height % patch_height == 0 && image_width % patch_width == 0,
            "image dimensions must be divisible by patch size",
        );

        let num_patch_rows = image_height / patch_height;
        let num_patch_cols = image_width / patch_width;
        let patch_dim = self.channels * patch_height * patch_width;

        PatchEmbedding {
            patch_norm: LayerNormConfig::new(patch_dim).init(device),
            patch_to_embedding: LinearConfig::new(patch_dim, self.dim).init(device),
            embedding_norm: LayerNormConfig::new(self.dim).init(device),
            image_height,
            image_width,
            patch_height,
            patch_width,
            channels: self.channels,
            num_patch_rows,
            num_patch_cols,
            num_patches: num_patch_rows * num_patch_cols,
        }
    }
}

impl<B: Backend> PatchEmbedding<B> {
    pub fn num_patches(&self) -> usize {
        self.num_patches
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 3> {
        let [_batch_size, channels, image_height, image_width] = input.dims();

        assert_eq!(
            channels, self.channels,
            "input channel count must match the configured channels"
        );
        assert_eq!(
            image_height, self.image_height,
            "input height must match the configured image height"
        );
        assert_eq!(
            image_width, self.image_width,
            "input width must match the configured image width"
        );

        let rows_chunked = input.chunk(self.num_patch_rows, 2);
        let rows: Tensor<B, 5> = Tensor::stack(rows_chunked, 1);
        let cols_chunked = rows.chunk(self.num_patch_cols, 4);
        let grid: Tensor<B, 6> = Tensor::stack(cols_chunked, 2);
        let patches: Tensor<B, 5> = grid.flatten(1, 2);
        let patches: Tensor<B, 3> = patches.flatten(2, 4);

        let x = self.patch_norm.forward(patches);
        let x = self.patch_to_embedding.forward(x);
        self.embedding_norm.forward(x)
    }

    pub fn patch_size(&self) -> [usize; 2] {
        [self.patch_height, self.patch_width]
    }
}

#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum ViTPool {
    Cls,
    Mean,
}

#[derive(Config, Debug)]
pub struct ViTConfig {
    pub image_size: [usize; 2],
    pub patch_size: [usize; 2],
    pub num_classes: usize,
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub mlp_dim: usize,
    #[config(default = "ViTPool::Cls")]
    pub pool: ViTPool,
    #[config(default = 3)]
    pub channels: usize,
    #[config(default = 64)]
    pub dim_head: usize,
    #[config(default = 0.0)]
    pub dropout: f64,
    #[config(default = 0.0)]
    pub emb_dropout: f64,
}

#[derive(Debug, Clone)]
pub enum ViTOutput<B: Backend> {
    Logits(Tensor<B, 2>),
    Tokens(Tensor<B, 3>),
}

#[derive(Module, Debug)]
pub struct ViT<B: Backend> {
    patch_embedding: PatchEmbedding<B>,
    cls_token: Option<Param<Tensor<B, 3>>>,
    pos_embedding: Param<Tensor<B, 3>>,
    dropout: Dropout,
    transformer: Transformer<B>,
    mlp_head: Option<Linear<B>>,
    use_mean_pool: bool,
    dim: usize,
}

impl ViTConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ViT<B> {
        assert!(self.dim > 0, "dim must be greater than zero");
        assert!(self.heads > 0, "heads must be greater than zero");
        assert!(self.dim_head > 0, "dim_head must be greater than zero");

        let patch_embedding = PatchEmbeddingConfig::new(self.image_size, self.patch_size, self.dim)
            .with_channels(self.channels)
            .init(device);
        let use_cls_token = self.pool == ViTPool::Cls;
        let num_cls_tokens = usize::from(use_cls_token);
        let position_sequence_len = patch_embedding.num_patches() + num_cls_tokens;

        ViT {
            patch_embedding,
            cls_token: use_cls_token.then(|| {
                Param::from_tensor(Tensor::random(
                    [1, 1, self.dim],
                    Distribution::Normal(0.0, 1.0),
                    device,
                ))
            }),
            pos_embedding: Param::from_tensor(Tensor::random(
                [1, position_sequence_len, self.dim],
                Distribution::Normal(0.0, 1.0),
                device,
            )),
            dropout: DropoutConfig::new(self.emb_dropout).init(),
            transformer: TransformerConfig::new(
                self.dim,
                self.depth,
                self.heads,
                self.dim_head,
                self.mlp_dim,
            )
            .with_dropout(self.dropout)
            .init(device),
            mlp_head: (self.num_classes > 0)
                .then(|| LinearConfig::new(self.dim, self.num_classes).init(device)),
            use_mean_pool: self.pool == ViTPool::Mean,
            dim: self.dim,
        }
    }
}

impl<B: Backend> ViT<B> {
    pub fn forward_tokens(&self, input: Tensor<B, 4>) -> Tensor<B, 3> {
        let batch_size = input.dims()[0];
        let mut x = self.patch_embedding.forward(input);

        if let Some(cls_token) = &self.cls_token {
            let cls_token = cls_token.val().repeat_dim(0, batch_size);
            x = Tensor::cat(vec![cls_token, x], 1);
        }

        let seq_len = x.dims()[1];
        let pos_embedding = self
            .pos_embedding
            .val()
            .slice([0..1, 0..seq_len, 0..self.dim]);
        let x = x + pos_embedding;
        let x = self.dropout.forward(x);

        self.transformer.forward(x)
    }

    pub fn forward_logits(&self, input: Tensor<B, 4>) -> Option<Tensor<B, 2>> {
        let x = self.forward_tokens(input);
        let [batch_size, _seq_len, dim] = x.dims();

        let latent: Tensor<B, 2> = if self.use_mean_pool {
            x.mean_dim(1).squeeze()
        } else {
            x.slice([0..batch_size, 0..1, 0..dim]).squeeze()
        };

        self.mlp_head
            .as_ref()
            .map(|mlp_head| mlp_head.forward(latent))
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> ViTOutput<B> {
        match self.forward_logits(input.clone()) {
            Some(logits) => ViTOutput::Logits(logits),
            None => ViTOutput::Tokens(self.forward_tokens(input)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Shape;

    type TestBackend = burn::backend::ndarray::NdArray<f32>;

    #[test]
    fn vit_forward_cls_pool_returns_logits() {
        let device = Default::default();
        let model = ViTConfig::new([8, 8], [4, 4], 10, 16, 2, 2, 32).init::<TestBackend>(&device);
        let images = Tensor::<TestBackend, 4>::random(
            Shape::new([2, 3, 8, 8]),
            Distribution::Default,
            &device,
        );

        let output = model.forward(images);

        match output {
            ViTOutput::Logits(logits) => assert_eq!(logits.dims(), [2, 10]),
            ViTOutput::Tokens(_) => panic!("expected logits output"),
        }
    }

    #[test]
    fn vit_forward_mean_pool_returns_logits_without_cls_token() {
        let device = Default::default();
        let model = ViTConfig::new([8, 8], [4, 4], 10, 16, 2, 2, 32)
            .with_pool(ViTPool::Mean)
            .init::<TestBackend>(&device);
        let images = Tensor::<TestBackend, 4>::random(
            Shape::new([2, 3, 8, 8]),
            Distribution::Default,
            &device,
        );

        let logits = model
            .forward_logits(images)
            .expect("classification head should be present");

        assert_eq!(logits.dims(), [2, 10]);
    }

    #[test]
    fn vit_forward_without_head_returns_tokens() {
        let device = Default::default();
        let model = ViTConfig::new([8, 8], [4, 4], 0, 16, 2, 2, 32)
            .with_pool(ViTPool::Mean)
            .init::<TestBackend>(&device);
        let images = Tensor::<TestBackend, 4>::random(
            Shape::new([2, 3, 8, 8]),
            Distribution::Default,
            &device,
        );

        let output = model.forward(images);

        match output {
            ViTOutput::Tokens(tokens) => assert_eq!(tokens.dims(), [2, 4, 16]),
            ViTOutput::Logits(_) => panic!("expected token output"),
        }
    }
}
