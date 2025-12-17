use burn::backend::{
        Autodiff, Wgpu,
        wgpu::WgpuDevice,
    };
use my_vit_burn::training;

type MyBackend = Wgpu;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    println!("Hello, world!");
    let device = WgpuDevice::default();

    training::run::<MyAutodiffBackend>(device);
}
