use burn::{
    backend::{
        Autodiff, Wgpu,
        wgpu::{Metal, WgpuDevice},
    },
    data::dataset::{HuggingfaceDatasetLoader, SqliteDataset},
    optim::{AdamConfig, decay::WeightDecayConfig},
    prelude::Backend,
};
use my_vit_burn::cifar10_item::{Cifar10ItemRaw, build_dataset};

type MyBackend = Wgpu;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    println!("Hello, world!");
    let device = WgpuDevice::default();

    build_dataset::<MyBackend>();
}
