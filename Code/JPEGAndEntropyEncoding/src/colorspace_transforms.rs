use crate::JPEGContainer;

// rgb to ycbcr

pub fn rgb_to_y(R : f64, G : f64, B : f64) -> f64{
    ((0.299 * (R )) + (0.587 * (G )) + (0.114 * (B ))).round() 
}

pub fn rgb_to_cb(R : f64, G : f64, B : f64) -> f64{
    (128.0 - (0.168736 * (R )) - (0.331264 * (G )) + (0.5 * (B ))).round()  
}

pub fn rgb_to_cr(R : f64, G : f64, B : f64) -> f64{
    (128.0 + (0.5 * (R )) - (0.418688 * (G )) - (0.081312 * (B ))).round()  
}

pub fn rgb_to_ycbcr(R : f64, G : f64, B : f64) -> (f64, f64, f64){
    (rgb_to_y(R,G,B),rgb_to_cb(R,G,B),rgb_to_cr(R,G,B))
}

// ycbcr to rgb

pub fn ycbcr_to_r(Y : f64, Cb : f64, Cr : f64) -> f64{
    (Y  + 1.402 * (Cr - 128.0) ).round() 
}

pub fn ycbcr_to_g(Y : f64, Cb : f64, Cr : f64) -> f64{
    (Y  - 0.344136 * (Cb - 128.0)  -0.714136 * (Cr - 128.0) ).round() 
}

pub fn ycbcr_to_b(Y : f64, Cb : f64, Cr : f64) -> f64{
    (Y  + 1.772 * (Cb - 128.0) ).round() 
}

pub fn ycbcr_to_rgb(Y : f64, Cb : f64, Cr : f64) -> (f64,f64,f64){
    (ycbcr_to_r(Y,Cb,Cr),ycbcr_to_g(Y,Cb,Cr),ycbcr_to_b(Y,Cb,Cr))
}


