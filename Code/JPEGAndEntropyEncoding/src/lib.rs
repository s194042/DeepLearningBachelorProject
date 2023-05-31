use arithmetic_encoding::ArithEncoder;
use pyo3::prelude::*;
mod colorspace_transforms;
mod dct;
mod quantization;
mod arithmetic_encoding;
mod JPEGSteps;
mod runlength_encoding;
mod file_io;
use std::{sync::{Arc,Mutex}};

use crate::runlength_encoding::JPEGSymbol;

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
    m.add_function(wrap_pyfunction!(JPEG_compress_to_file,m)?)?;
    m.add_function(wrap_pyfunction!(JPEG_decompress_from_file,m)?)?;
    m.add_function(wrap_pyfunction!(arith_encode_latent_layer,m)?)?;
    m.add_function(wrap_pyfunction!(arith_decode_latent_layer,m)?)?;
    Ok(())
}


#[pyfunction]
fn arith_encode_latent_layer(mut latent_layer : Vec<i32>, path : &str) -> Py<PyAny>{
    let eof : i32 = 2i32.pow(20);
    let mut arith_encoder = arithmetic_encoding::ArithEncoder::new(latent_layer,eof);
    arith_encoder.encode();
    file_io::write_to_bin(&file_io::encode_AE_buffer(&arith_encoder.freq_count, &arith_encoder.encoded_message), path);
    return Python::with_gil(|py| arith_encoder.encoded_message.to_object(py));

}
#[pyfunction]
fn arith_decode_latent_layer(path : &str) -> Py<PyAny>{
    let eof = 2i32.pow(21);
    let (freq_vec, encoded_message) = file_io::decode_AE_buffer(file_io::load_from_bin(path));
    let mut arith_encoder = ArithEncoder::from_encoded_message(freq_vec, encoded_message, eof);
    arith_encoder.decode();
    return Python::with_gil(|py| arith_encoder.message.to_object(py));

}



#[pyfunction]
fn JPEGcompress_and_decompress(mut image : Vec<Vec<Vec<f64>>>, Qf : f64, sampling : &str) -> Py<PyAny>{
    

    let dct_image = JPEG_compress_to_blocks(image, Qf, sampling);
    let original_image = JPEG_decompress_from_blocks(dct_image);

    return Python::with_gil(|py| original_image.to_object(py));
}



#[pyfunction]
fn JPEG_compress_to_file(mut image : Vec<Vec<Vec<f64>>>, Qf : f64, sampling : &str, path : &str){
    let container = JPEG_compress_to_blocks(image, Qf, sampling);
    let aux_data = file_io::AuxiliaryData{
        original_size : container.original_size,
        Qf : container.Qf,
        sample_type : container.sample_type,
    };
    let mut arith_encoder = JPEGSteps::entropy_encoding(container);
    arith_encoder.encode();
    file_io::JPEG_arithmetic_encoding_to_file(&arith_encoder, aux_data, path);
}

#[pyfunction]
fn JPEG_decompress_from_file(path : &str) -> Vec<Vec<Vec<usize>>>{
    let (aux_data,mut arith_encoder) = file_io::JPEG_arithmetic_encoding_from_file(path);
    arith_encoder.decode();
    let container = JPEGSteps::entropy_decode(arith_encoder, Arc::new(aux_data));
    return JPEG_decompress_from_blocks(container);
}

fn JPEG_compress_to_blocks(mut image : Vec<Vec<Vec<f64>>>, Qf : f64, sampling : &str) -> JPEGContainer{
    let sample_type = match sampling{
        "Down444" => Sampling::Down444,
        "Down422" => Sampling::Down422,
        "Down420" => Sampling::Down420,
        _ => panic!("Not a valid sampling type"),
    };
    let mut result_vector = Arc::new(Mutex::new(vec![JPEGSteps::CHANNEL_PROCESSING_RESULT::Empty; 3]));
    let mut downsampled_image = JPEGSteps::color_transform_and_dowsample_image(image, sample_type, Qf);
    let mut dct_image = JPEGSteps::parallel_function_over_channels(downsampled_image, Arc::new(JPEGSteps::dct_and_quantization_over_channel), result_vector.clone());
    return dct_image;    
}

fn JPEG_decompress_from_blocks(mut dct_image : JPEGContainer) -> Vec<Vec<Vec<usize>>>{
    let mut result_vector = Arc::new(Mutex::new(vec![JPEGSteps::CHANNEL_PROCESSING_RESULT::Empty; 3]));
    let mut inverse_dct_image = JPEGSteps::parallel_function_over_channels(dct_image, Arc::new(JPEGSteps::inverse_quantization_and_dct_over_channel), result_vector.clone());
    let mut original_image = JPEGSteps::upsample_and_inverse_color_transform_image(inverse_dct_image);
    return original_image;
}


