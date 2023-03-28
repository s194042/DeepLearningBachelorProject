use pyo3::prelude::*;
mod colorspace_transforms;
mod dct;
mod quantization;
mod arithmetic_encoding;
mod JPEGSteps;
mod entropy_encoding_step;
mod file_io;
use std::{sync::{Arc,Mutex}};

pub struct JPEGContainer{
    y_channel : Vec<Vec<f64>>,
    cb_channel : Vec<Vec<f64>>,
    cr_channel : Vec<Vec<f64>>,
    original_size : (usize,usize),
    Qf : f64,
    sample_type : Sampling,
}

impl ToPyObject for JPEGContainer {
    fn to_object(&self, py: pyo3::Python<'_>) -> pyo3::Py<pyo3::PyAny> { 
        
        (self.y_channel.clone(),self.cb_channel.clone(),self.cr_channel.clone(),self.original_size.clone()).to_object(py)
        
    }
}

#[derive(PartialEq,Debug,Copy, Clone)]
pub enum Sampling {
    Down444,
    Down422,
    Down420
}


/// A Python module implemented in Rust.
#[pymodule]
fn JPEGAndEntropyEncoding(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(JPEGcompress_and_decompress,m)?)?;
    Ok(())
}

#[pyfunction]
fn JPEGcompress_and_decompress(mut image : Vec<Vec<Vec<f64>>>, Qf : f64) -> Py<PyAny>{
    
    let dct_image = JPEG_compress_to_blocks(image, Qf);
    let original_image = JPEG_decompress_from_blocks(dct_image, Qf);

    return Python::with_gil(|py| original_image.to_object(py));
}



#[pyfunction]
fn JPEG_compress_to_file(mut image : Vec<Vec<Vec<f64>>>, Qf : f64, path : &str){
    let container = JPEG_compress_to_blocks(image, Qf);
    let aux_data = file_io::AuxiliaryData{
        original_size : container.original_size,
        Qf : container.Qf,
        sample_type : container.sample_type,
    };
    let mut arith_encoder = JPEGSteps::entropy_encoding(container);
    file_io::arithmetic_encoding_to_file(&arith_encoder, aux_data, path);
}

#[pyfunction]
fn JPEG_decompress_from_file(path : &str){
    let  (aux_data,mut arith_encoder) = file_io::arithmetic_encoding_from_file(path);

}


fn JPEG_compress_to_blocks(mut image : Vec<Vec<Vec<f64>>>, Qf : f64) -> JPEGContainer{
    let mut result_vector = Arc::new(Mutex::new(vec![JPEGSteps::CHANNEL_PROCESSING_RESULT::Empty; 3]));
    let mut downsampled_image = JPEGSteps::color_transform_and_dowsample_image(image, Sampling::Down444, Qf);
    let mut dct_image = JPEGSteps::parallel_function_over_channels(downsampled_image, Arc::new(JPEGSteps::dct_and_quantization_over_channel), result_vector.clone());
    return dct_image;    
}

fn JPEG_decompress_from_blocks(mut dct_image : JPEGContainer, Qf : f64) -> Vec<Vec<Vec<usize>>>{
    let mut result_vector = Arc::new(Mutex::new(vec![JPEGSteps::CHANNEL_PROCESSING_RESULT::Empty; 3]));
    let mut inverse_dct_image = JPEGSteps::parallel_function_over_channels(dct_image, Arc::new(JPEGSteps::inverse_quantization_and_dct_over_channel), result_vector.clone());
    let mut original_image = JPEGSteps::upsample_and_inverse_color_transform_image(inverse_dct_image);
    return original_image;
}


