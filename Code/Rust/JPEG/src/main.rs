mod colorspace_transforms;
mod dct;
mod downsampling;
mod quantization;
mod arithmetic_encoding;

use colorspace_transforms::{rgb_to_ycbcr,ycbcr_to_rgb};
use std::{env};

fn main() {
    println!("Hello, world!");
}
