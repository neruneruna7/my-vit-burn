use crate::cifar10_item::Cifar10Item;
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

#[derive(Clone)]
pub struct Cifar10Batcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> Cifar10Batcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct Cifar10Batch<B: Backend> {
    pub images: Tensor<B, 4>,       // [Batch, Channel, Height, Width]
    pub targets: Tensor<B, 1, Int>, // [Batch]
}

impl<B: Backend> Batcher<B, Cifar10Item<B>, Cifar10Batch<B>> for Cifar10Batcher<B> {
    fn batch(
        &self,
        items: Vec<Cifar10Item<B>>,
        _device: &<B as Backend>::Device,
    ) -> Cifar10Batch<B> {
        // 画像をスタック: [C, H, W] -> [B, C, H, W]
        let images = items
            .iter()
            .map(|item| item.image.clone().unsqueeze()) // [1, C, H, W] に拡張
            .collect();

        let images = Tensor::cat(images, 0).to_device(&self.device);

        // ラベルをスタック: usize -> Tensor<B, 1, Int>
        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_ints([item.label as i32], &self.device))
            .collect();

        let targets = Tensor::cat(targets, 0);

        Cifar10Batch { images, targets }
    }
}
