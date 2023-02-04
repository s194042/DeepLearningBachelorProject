

// rgb to ycbcr

fn rgb_to_y(R : usize, G : usize, B : usize) -> usize{
    ((0.299 * (R as f64)) + (0.587 * (G as f64)) + (0.114 * (B as f64))).round() as usize 
}

fn rgb_to_cb(R : usize, G : usize, B : usize) -> usize{
    (128.0 - (0.168736 * (R as f64)) - (0.331264 * (G as f64)) + (0.5 * (B as f64))).round() as usize 
}

fn rgb_to_cr(R : usize, G : usize, B : usize) -> usize{
    (128.0 + (0.5 * (R as f64)) - (0.418688 * (G as f64)) - (0.081312 * (B as f64))).round() as usize 
}

pub fn rgb_to_ycbcr(R : usize, G : usize, B : usize) -> (usize, usize, usize){
    (rgb_to_y(R,G,B),rgb_to_cb(R,G,B),rgb_to_cr(R,G,B))
}

// ycbcr to rgb

fn ycbcr_to_r(Y : usize, Cb : usize, Cr : usize) -> usize{
    (Y as f64 + 1.402 * (Cr - 128) as f64).round() as usize
}

fn ycbcr_to_g(Y : usize, Cb : usize, Cr : usize) -> usize{
    (Y as f64 - 0.344136 * (Cb - 128) as f64 -0.714136 * (Cr - 128) as f64).round() as usize
}

fn ycbcr_to_b(Y : usize, Cb : usize, Cr : usize) -> usize{
    (Y as f64 + 1.772 * (Cb - 128) as f64).round() as usize
}

pub fn ycbcr_to_rgb(Y : usize, Cb : usize, Cr : usize) -> (usize,usize,usize){
    (ycbcr_to_r(Y,Cb,Cr),ycbcr_to_g(Y,Cb,Cr),ycbcr_to_b(Y,Cb,Cr))
}
