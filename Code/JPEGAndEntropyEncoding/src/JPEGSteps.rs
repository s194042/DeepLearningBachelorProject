use crate::{JPEGContainer,Sampling::*,Sampling, dct,quantization,runlength_encoding::{*},arithmetic_encoding::{self, ArithEncoder}, file_io::AuxiliaryData};
use crate::colorspace_transforms::*;
use core::panic;
use std::{ops::DerefMut, io::Empty};
use std::thread;
use std::sync::mpsc::{self, channel};
use std::cmp::min;
use std::sync::{Arc,Mutex};

#[derive(Clone)]
pub enum CHANNEL_PROCESSING_RESULT{
    Empty,
    RUN_LENGTH_COMPRESS(Vec<JPEGSymbol>),
    RUN_LENGTH_DECOMPRESS(Vec<Vec<f64>>),
}


pub fn parallel_function_over_channels(mut image : JPEGContainer, func : Arc<dyn Fn(&mut Vec<Vec<f64>>,f64,bool) -> CHANNEL_PROCESSING_RESULT + Send + Sync>, result_vector : Arc<Mutex<Vec<CHANNEL_PROCESSING_RESULT>>> ) -> JPEGContainer{


    let mut senders = vec![];
    let mut receivers = vec![];

    for i in 0..6{
        let (tx,rx) = mpsc::channel();
        senders.push(tx);
        receivers.push(rx);
    }

    senders[0].send(image.y_channel).unwrap();
    senders[1].send(image.cb_channel).unwrap();
    senders[2].send(image.cr_channel).unwrap();

    let mut threads = vec![];

    for channel in 0..3{
        let receiver = receivers.remove(0);
        let sender = senders.remove(3);
        let f = func.clone();
        let r = result_vector.clone();
        threads.push(
            thread::spawn(move ||{
                let mut image_channel = receiver.recv().unwrap();
                let tmp_res =  f(&mut image_channel, image.Qf, channel > 0);
                r.lock().unwrap()[channel] = tmp_res;

                sender.send(image_channel).unwrap();
            })
        )
    }

    image.y_channel = receivers[0].recv().unwrap();
    image.cb_channel = receivers[1].recv().unwrap();
    image.cr_channel = receivers[2].recv().unwrap();
    for handle in threads{
        handle.join().unwrap();
    }
    return image;


}

pub fn color_transform_and_dowsample_image(image : Vec<Vec<Vec<f64>>>, sample_type : Sampling, Qf : f64) -> JPEGContainer{

    let (height,width) = (8 * (image.len() as f64 / 8.0).ceil() as usize,8 * (image[0].len() as f64 / 8.0).ceil() as usize);

    let (subsample_height,subsample_width) = match sample_type {
        Down444 => (height,width),
        Down422 => (height,((width / 8) as f64 / 2.0).ceil() as usize * 8),
        Down420 => (((height / 8) as f64 / 2.0).ceil() as usize * 8,((width / 8) as f64 / 2.0).ceil() as usize * 8),
    };

    let mut result = JPEGContainer{
        y_channel : vec![vec![0.0;width];height],
        cb_channel : vec![vec![0.0;subsample_width];subsample_height],
        cr_channel : vec![vec![0.0;subsample_width];subsample_height],
        original_size : (image.len(),image[0].len()),
        Qf,
        sample_type,
    };

    for i in 0..image.len(){
        for j in 0..image[0].len(){
            result.y_channel[i][j] = -128.0 + rgb_to_y(image[i][j][0], image[i][j][1], image[i][j][2])
        }
    }
    let down_h = if result.sample_type == Down420 {2} else {1};
    let down_w = if result.sample_type != Down444 {2} else {1};

    for i in 0..image.len() / down_h {
        for j in 0..image[0].len() / down_w {
            result.cb_channel[i][j] = -128.0 + rgb_to_cb(image[i* down_h][j * down_w][0], image[i* down_h][j * down_w][1], image[i* down_h][j * down_w][2]);
            result.cr_channel[i][j] = -128.0 + rgb_to_cr(image[i* down_h][j * down_w][0], image[i* down_h][j * down_w][1], image[i* down_h][j * down_w][2]);

        }
    }

    return result;

}

pub fn dct_and_quantization_over_channel(channel : &mut Vec<Vec<f64>>, Qf : f64, chroma : bool) -> CHANNEL_PROCESSING_RESULT{

    let quantization_matrix = if chroma {quantization::gen_quantize_chroma_matrix(Qf)} else {quantization::gen_quantize_lumi_matrix(Qf)};

    for i in 0..channel.len() / 8{
        for j in 0..channel[0].len() / 8{
            let dct_block_coefficients = dct::dct_block(&channel,i*8, j*8);
            for block_i in 0..8{
                for block_j in 0..8{
                    channel[i*8+block_i][j*8+block_j] = dct_block_coefficients[block_i][block_j];
                }
            }
            quantization::quantization_block(channel, i*8, j*8, &quantization_matrix);
        }
    }
    CHANNEL_PROCESSING_RESULT::Empty
}

pub fn runlength_encoding_over_channel(mut channel : &mut Vec<Vec<f64>>, Qf : f64, chroma : bool) -> CHANNEL_PROCESSING_RESULT{
    let mut result = vec![];
    let zigzag_sequence = generate_zigzag_sequence(8);

    for i in 0..channel.len() / 8{
        for j in 0..channel[0].len() / 8{
            result.extend(run_length_encoding_block(channel, i * 8, j * 8, &zigzag_sequence))
        }
    }
    result.push(JPEGSymbol::CHANNEL_MARKER);
    return CHANNEL_PROCESSING_RESULT::RUN_LENGTH_COMPRESS(result);
}

pub fn entropy_encoding(mut image : JPEGContainer) -> ArithEncoder<JPEGSymbol>{
    let results_vector = Arc::new(Mutex::new(vec![CHANNEL_PROCESSING_RESULT::Empty;3]));

    parallel_function_over_channels(image, Arc::new(runlength_encoding_over_channel), results_vector.clone());

    let mut total_run_length_message = vec![];
    for _ in 0..3{total_run_length_message.extend(match results_vector.lock().unwrap().pop().unwrap(){
        CHANNEL_PROCESSING_RESULT::RUN_LENGTH_COMPRESS(coded_message) => coded_message,
        _ => panic!("Wrong processing result during entropy encoding")
    
    })} 
    total_run_length_message.push(JPEGSymbol::EOF);
    let arith_encoder = arithmetic_encoding::ArithEncoder::new(total_run_length_message, JPEGSymbol::EOF);
    return arith_encoder;

}

pub fn entropy_decode(arith_encoder : ArithEncoder<JPEGSymbol>, aux_data : Arc<AuxiliaryData>) -> JPEGContainer{

    let results_vector = Arc::new(Mutex::new(vec![CHANNEL_PROCESSING_RESULT::Empty;3]));
    let mut message_index = 0;
    let mut threads = vec![];

    for i in 0..3{

        let mut channel_encoding = vec![];
        loop{
            if arith_encoder.message[message_index] == JPEGSymbol::CHANNEL_MARKER{
                message_index += 1;
                let a = aux_data.clone();
                let r = results_vector.clone();
                threads.push(thread::spawn(move || {
                    let channel_encoding = channel_encoding;
                    r.lock().unwrap()[i] = channel_from_runlength_encoding(&channel_encoding, a, i > 0);
                }));
                break;
            }else{
                channel_encoding.push(arith_encoder.message[message_index]);
                message_index += 1;
            }
        }
    }
    for thread in threads{
        thread.join().unwrap();
    }



    let mut results = results_vector.lock().unwrap();
    let y_channel = match results.pop().unwrap(){
        CHANNEL_PROCESSING_RESULT::RUN_LENGTH_DECOMPRESS(x) => x,
        _ => panic!("Wrong run_length decoding result")
    };

    let cb_channel = match results.pop().unwrap(){
        CHANNEL_PROCESSING_RESULT::RUN_LENGTH_DECOMPRESS(x) => x,
        _ => panic!("Wrong run_length decoding result")
    };

    let cr_channel = match results.pop().unwrap(){
        CHANNEL_PROCESSING_RESULT::RUN_LENGTH_DECOMPRESS(x) => x,
        _ => panic!("Wrong run_length decoding result")
    };

    let containter = JPEGContainer{
        y_channel,
        cb_channel,
        cr_channel,
        Qf : aux_data.Qf,
        original_size : aux_data.original_size,
        sample_type : aux_data.sample_type
    };

    return containter


}


pub fn channel_from_runlength_encoding(channel_encoding : &Vec<JPEGSymbol>, aux_data : Arc<AuxiliaryData>, chroma : bool) -> CHANNEL_PROCESSING_RESULT{
    let (height,width) = (8 * (aux_data.original_size.0 as f64 / 8.0).ceil() as usize,8 * (aux_data.original_size.1 as f64 / 8.0).ceil() as usize);
    let (subsample_height,subsample_width) = match aux_data.sample_type {
        Down444 => (height,width),
        Down422 => (height,((width / 8) as f64 / 2.0).ceil() as usize * 8),
        Down420 => (((height / 8) as f64 / 2.0).ceil() as usize * 8,((width / 8) as f64 / 2.0).ceil() as usize * 8),
    };
    let mut result = vec![vec![0.0; if chroma {subsample_width} else {width}];if chroma {subsample_height} else {height}];


    let zigzag_sequence = generate_zigzag_sequence(8);
    let mut run_length_index = 0;

    for i in 0..height / 8{
        for j in 0..width / 8{
            run_length_decoding_block(&mut result, i * 8, j * 8, &zigzag_sequence, channel_encoding, &mut run_length_index);
        }
    }
    return CHANNEL_PROCESSING_RESULT::RUN_LENGTH_DECOMPRESS(result);


}

pub fn inverse_quantization_and_dct_over_channel(channel : &mut Vec<Vec<f64>>, Qf : f64, chroma : bool) -> CHANNEL_PROCESSING_RESULT{
    let quantization_matrix = if chroma {quantization::gen_quantize_chroma_matrix(Qf)} else {quantization::gen_quantize_lumi_matrix(Qf)};

    for i in 0..channel.len() / 8{
        for j in 0..channel[0].len() / 8{
            quantization::inverse_quantization_block(channel, i*8, j*8, &quantization_matrix);
            let inverse_dct_block_coefficients = dct::inv_dct_block(&channel,i*8, j*8);
            for block_i in 0..8{
                for block_j in 0..8{
                    channel[i*8+block_i][j*8+block_j] = inverse_dct_block_coefficients[block_i][block_j].round();
                }
            }
        }
    }

    CHANNEL_PROCESSING_RESULT::Empty
}

pub fn upsample_and_inverse_color_transform_image(image : JPEGContainer) -> Vec<Vec<Vec<usize>>>{
    let mut result = vec![vec![vec![0;3];image.original_size.1];image.original_size.0];

    let (up_h,up_w) = match image.sample_type {
        Down444 => (1,1),
        Down422 => (1,2),
        Down420 => (2,2),
    };

    let mut cb = 0.0;
    let mut cr = 0.0;

    for i in 0..image.original_size.0{
        for j in 0..image.original_size.1{
            let y = image.y_channel[i][j] + 128.0;
            if i % up_h == 0 && j % up_w == 0{
                cb = image.cb_channel[i/up_h][j / up_w] + 128.0;
                cr = image.cr_channel[i/up_h][j / up_w] + 128.0;
            }
            let (r,g,b) = ycbcr_to_rgb(y, cb, cr);
            result[i][j][0] = min(255,r as usize);
            result[i][j][1] = min(255,g as usize);
            result[i][j][2] = min(255,b as usize);

        }
    }


    return result;
}


