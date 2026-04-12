use std::collections::HashMap;

use burn::{
    backend::Wgpu,
    data::dataset::{
        Dataset as _, HuggingfaceDatasetLoader, SqliteDataset, transform::MapperDataset,
    },
    prelude::Backend,
};
use image::{ColorType, save_buffer};
use my_vit_burn::cifar10_item::{Cifar10Item, Cifar10ItemRaw, Cifar10Mapper};
use serde_json::Value;

type MyBackend = Wgpu;

fn save_sample_image<B: Backend>(item: &Cifar10Item<B>, file_path: &str) {
    // 1. Tensorの形状を [C, H, W] から [H, W, C] に変更
    // Burnの画像は通常 [Channel, Height, Width] だが、
    // 保存時は [Height, Width, Channel] の順にピクセルが並んでいる必要がある
    let image_tensor = item.image.clone().permute([1, 2, 0]);

    // 2. データをCPU上の生データ(f32の配列)として取得
    let data = image_tensor.into_data();
    // iter::<f32>() でイテレータを取得し、Vec<f32> に変換します
    let floats: Vec<f32> = data.iter::<f32>().collect();

    // 3. [-1.0, 1.0] の f32 を 0-255 の u8 に戻す
    let bytes: Vec<u8> = floats
        .iter()
        .map(|&x| (((x + 1.0) * 0.5) * 255.0).clamp(0.0, 255.0) as u8)
        .collect();

    // 4. 画像として保存 (CIFAR-10は 32x32)
    save_buffer(file_path, &bytes, 32, 32, ColorType::Rgb8).expect("Failed to save image");

    println!("Saved image to: {}", file_path);
    println!("Label: {}", item.label);
}
fn main() {
    let dataset_debug: SqliteDataset<HashMap<String, Value>> =
        HuggingfaceDatasetLoader::new("cifar10")
            .dataset("train")
            .unwrap();

    // 最初のアイテムを取得してキー（カラム名）を表示
    if let Some(item) = dataset_debug.get(0) {
        println!("Available columns: {:?}", item.keys());
    }

    // データセットの構築 (前回のコードと同様)
    let dataset_raw: SqliteDataset<Cifar10ItemRaw> = HuggingfaceDatasetLoader::new("cifar10")
        .dataset("train")
        .unwrap();

    let mapper = Cifar10Mapper::<MyBackend> {
        _phantom: std::marker::PhantomData,
    };
    let dataset = MapperDataset::new(dataset_raw, mapper);

    // --- ここから保存テスト ---

    // インデックス0のデータを取得
    let item = dataset.get(0).unwrap();

    // 画像を保存
    save_sample_image(&item, "cifar10_sample_0.png");

    // 別の画像も保存してみる (例: インデックス10)
    if let Some(item_10) = dataset.get(10) {
        save_sample_image(&item_10, "cifar10_sample_10.png");
    }
}
