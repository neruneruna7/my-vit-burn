// image params
pub const IMAGE_WH: usize = 32;
pub const CHANNEL: usize = 3;

// vit hyper prams
pub const PATCH_WH: usize = 8;
pub const SPLIT_ROWS: usize = IMAGE_WH / PATCH_WH;
pub const SPLIT_COLS: usize = IMAGE_WH / PATCH_WH;
pub const PATCH_TOTAL: usize = (IMAGE_WH / PATCH_WH).pow(2);
pub const PATCH_VECTOR_LEN: usize = CHANNEL * PATCH_WH.pow(2);
pub const EMBED_VECTOR_LEN: usize = PATCH_VECTOR_LEN / 2;

// transformer params
pub const HEAD: usize = 12;
pub const DIM_FEEDFORWARD: usize = EMBED_VECTOR_LEN;
pub const ACTIVATION: &str = "gelu";
pub const LAYERS: usize = 12;
pub const DROPOUT: f64 = 0.0;
pub const LAYER_NORM_EPSILON: f64 = 1e-6;
pub const POSITION_EMBEDDING_STD: f64 = 0.02;

// training params
pub const LEARNING_RATE: f64 = 1e-4;
pub const WEIGHT_DECAY: f32 = 1e-2;
pub const ADAM_EPSILON: f32 = 1e-8;
