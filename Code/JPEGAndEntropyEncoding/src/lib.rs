use downsampling::down420;
use pyo3::prelude::*;
mod colorspace_transforms;
mod dct;
mod downsampling;
mod quantization;
mod arithmetic_encoding;
mod JPEGSteps;


pub struct JPEGContainer{
    y_channel : Vec<Vec<f64>>,
    cb_channel : Vec<Vec<f64>>,
    cr_channel : Vec<Vec<f64>>,
    original_size : (usize,usize),
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


/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn JPEGAndEntropyEncoding(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(list_test,m)?)?;
    Ok(())
}

#[pyfunction]
fn list_test(mut image : Vec<Vec<Vec<f64>>>) -> Py<PyAny>{

    let downsampled_image = JPEGSteps::color_transform_and_dowsample_image(image, Sampling::Down444);

    return Python::with_gil(|py| downsampled_image.to_object(py));
}






#[cfg(test)]

mod test{

    
    #[test]
    fn test1(){
        assert!(1==1);
    }
}