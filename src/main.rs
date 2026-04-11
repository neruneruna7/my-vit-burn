#![recursion_limit = "256"]
// なにやらコンパイルエラーがでるので，抑制のため再帰回数を指定

use burn::backend::{Autodiff, Wgpu, wgpu::WgpuDevice};
use my_vit_burn::training;
use tracing::info;

// type MyBackend = Wgpu;
type MyAutodiffBackend = Autodiff<Wgpu>;

fn main() {
    // tracingの初期化
    let subscriber = tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set global subscriber");

    info!("Hello, world!");

    let device = WgpuDevice::default();

    training::run::<MyAutodiffBackend>(device);
}
