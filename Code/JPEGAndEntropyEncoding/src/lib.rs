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

#[derive(PartialEq)]
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
    let mut result_vector = Arc::new(Mutex::new(vec![JPEGSteps::CHANNEL_PROCESSING_RESULT::Empty; 3]));

    let mut downsampled_image = JPEGSteps::color_transform_and_dowsample_image(image, Sampling::Down444, Qf);
    let mut dct_image = JPEGSteps::parallel_function_over_channels(downsampled_image, Arc::new(JPEGSteps::dct_and_quantization_over_channel), result_vector.clone());
    let mut inverse_dct_image = JPEGSteps::parallel_function_over_channels(dct_image, Arc::new(JPEGSteps::inverse_quantization_and_dct_over_channel), result_vector.clone());
    let mut original_image = JPEGSteps::upsample_and_inverse_color_transform_image(inverse_dct_image);
    return Python::with_gil(|py| original_image.to_object(py));
}


