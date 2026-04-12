use burn::data::dataset::transform::Mapper; // Mapperトレイト
use burn::data::dataset::transform::MapperDataset;
use burn::data::dataset::{Dataset, HuggingfaceDatasetLoader, SqliteDataset};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use serde::Deserialize;
use serde::Serialize;
use std::marker::PhantomData;

// 生データ（変更なし）
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cifar10ItemRaw {
    pub img_bytes: Vec<u8>,
    pub label: usize,
}

#[derive(Clone, Debug)]
pub struct Cifar10Item<B: Backend> {
    pub image: Tensor<B, 3>,
    pub label: usize,
}

// 【修正3】Mapperも Backend <B> に依存するため、PhantomDataを持たせる
// これがないと "type parameter B is not constrained" エラーになる
#[derive(Clone, Debug)]
pub struct Cifar10Mapper<B: Backend> {
    pub _phantom: PhantomData<B>,
}

impl<B: Backend> Mapper<Cifar10ItemRaw, Cifar10Item<B>> for Cifar10Mapper<B> {
    fn map(&self, item: &Cifar10ItemRaw) -> Cifar10Item<B> {
        // 画像変換処理（imageクレートを使用）
        let img = image::load_from_memory(&item.img_bytes)
            .expect("Failed to load image")
            .to_rgb8();

        let width = img.width() as usize;
        let height = img.height() as usize;

        // torchvision.transforms.Normalize((0.5,), (0.5,)) と同じく [-1.0, 1.0] に正規化する。
        let raw_pixels: Vec<f32> = img
            .into_raw()
            .into_iter()
            .map(|x| (x as f32) / 255.0 * 2.0 - 1.0)
            .collect();

        // Tensor作成
        let tensor_data = TensorData::new(raw_pixels, vec![height, width, 3]);
        let image =
            Tensor::<B, 3>::from_data(tensor_data, &B::Device::default()).permute([2, 0, 1]); // [H, W, C] -> [C, H, W]

        Cifar10Item {
            image,
            label: item.label,
        }
    }
}

// データセット構築部分の例（main関数など）
pub fn build_dataset<B: Backend>() {
    let dataset_raw: SqliteDataset<Cifar10ItemRaw> = HuggingfaceDatasetLoader::new("cifar10")
        .dataset("train")
        .unwrap();

    // Mapperのインスタンス化（PhantomDataを含める）
    let mapper = Cifar10Mapper::<B> {
        _phantom: PhantomData,
    };

    // MapperDatasetの作成
    let dataset = MapperDataset::new(dataset_raw, mapper);

    println!("Dataset size: {}", dataset.len());
}
