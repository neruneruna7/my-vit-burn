use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
    tensor::{Int, Tensor, backend::AutodiffBackend},
};

#[derive(Debug, Clone, Default)]
pub struct Cifar10Batcher {}

pub struct Cifar10Batch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub labels: Tensor<B, 1, Int>,
}

// impl<B: Backend> Batcher<Cifar10Item, Cifar10Batch<B>> for Cifar10Batcher {
//     fn batch(&self, items: Vec<Cifar10Item>) -> Cifar10Batch<B> {}
// }
