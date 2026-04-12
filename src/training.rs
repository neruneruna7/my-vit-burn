use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::transform::MapperDataset;
use burn::data::dataset::{HuggingfaceDatasetLoader, SqliteDataset};
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::{Learner, SupervisedTraining};
use std::marker::PhantomData;

use crate::cifar10_batcher::Cifar10Batcher;
use crate::cifar10_item::{Cifar10ItemRaw, Cifar10Mapper};
use crate::config::WEIGHT_DECAY;
use crate::vit::{Vit, VitConfig}; // あなたのViTモデルをインポート

// ハイパーパラメータ設定
#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 16)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 1e-4)]
    pub learning_rate: f64,
    #[config(default = 42)]
    pub seed: u64,

    // ViT用のパラメータ (CIFAR-10は32x32なのでパッチサイズに注意)
    #[config(default = 128)]
    pub hidden_size: usize,
    #[config(default = 4)]
    pub patch_size: usize, // 32x32画像なら4x4パッチで8x8=64トークンになる
    #[config(default = 4)]
    pub num_layers: usize,
    #[config(default = 4)]
    pub num_heads: usize,
    #[config(default = 10)]
    pub num_classes: usize,
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    // コンフィグの初期化
    let config = TrainingConfig::new();
    B::seed(&device, config.seed);

    // バッチャーの作成
    let batcher_train = Cifar10Batcher::<B>::new(device.clone());
    let batcher_valid = Cifar10Batcher::<B::InnerBackend>::new(device.clone());

    // --- データセットの読み込み ---
    // Train
    let dataset_train_raw: SqliteDataset<Cifar10ItemRaw> = HuggingfaceDatasetLoader::new("cifar10")
        .dataset("train")
        .unwrap();
    let mapper_train = Cifar10Mapper::<B> {
        _phantom: PhantomData,
    };
    let dataset_train = MapperDataset::new(dataset_train_raw, mapper_train);

    // Test (Validationとして使用)
    let dataset_test_raw: SqliteDataset<Cifar10ItemRaw> = HuggingfaceDatasetLoader::new("cifar10")
        .dataset("test")
        .unwrap();
    let mapper_test = Cifar10Mapper::<B::InnerBackend> {
        _phantom: PhantomData,
    };
    let dataset_test = MapperDataset::new(dataset_test_raw, mapper_test);

    // --- DataLoaderの構築 ---
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_train);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_test);

    // --- モデルとオプティマイザの初期化 ---
    // ※ ここはあなたのViTの実装に合わせて初期化してください
    // 一般的には ViTConfig::new(...).init(&device) のような形になります
    let model: Vit<B> = VitConfig {}.init(&device);

    let optimizer = AdamWConfig::new()
        .with_weight_decay(WEIGHT_DECAY) // ViTにはWeight Decayが重要
        .init();

    // --- 学習実行 (Learner) ---
    let learner = Learner::new(model, optimizer, config.learning_rate);
    let result =
        SupervisedTraining::new("/tmp/burn-vit-cifar10", dataloader_train, dataloader_test)
            .metric_train_numeric(AccuracyMetric::new())
            .metric_valid_numeric(AccuracyMetric::new())
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .with_file_checkpointer(CompactRecorder::new())
            .num_epochs(config.num_epochs)
            .summary()
            .launch(learner);

    let _model_trained = result.model;

    println!("Training completed!");
}
