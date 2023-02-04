mod colorspace_transforms;
mod dct;
mod downsampling;

use colorspace_transforms::{rgb_to_ycbcr,ycbcr_to_rgb};

fn main() {
    println!("Hello, world!");
    let (y,cb,cr) = rgb_to_ycbcr(120,56,198);
    println!("{} {} {}",y,cb,cr);
    let (r,g,b) = ycbcr_to_rgb(y,cb,cr);
    println!("{} {} {}",r,g,b);
}
