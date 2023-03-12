use crate::{JPEGContainer,Sampling::*,Sampling, dct,quantization};
use crate::colorspace_transforms::*;
use std::thread;
use std::sync::mpsc;



pub fn color_transform_and_dowsample_image(image : Vec<Vec<Vec<f64>>>, sample_type : Sampling) -> JPEGContainer{

    let (height,width) = (8 * (image[0].len() as f64 / 8.0).ceil() as usize,8 * (image[0][0].len() as f64 / 8.0).ceil() as usize);

    let (subsample_height,subsample_width) = match sample_type {
        Down444 => (height,width),
        Down422 => (height,((width / 8) as f64 / 2.0).ceil() as usize * 8),
        Down420 => (((height / 8) as f64 / 2.0).ceil() as usize * 8,((width / 8) as f64 / 2.0).ceil() as usize * 8),
    };

    let mut result = JPEGContainer{
        y_channel : vec![vec![0.0;width];height],
        cb_channel : vec![vec![0.0;subsample_width];subsample_height],
        cr_channel : vec![vec![0.0;subsample_width];subsample_height],
        original_size : (image[0].len(),image[0][0].len()),
        sample_type,
    };

    for i in 0..image[0].len(){
        for j in 0..image[0][0].len(){
            result.y_channel[i][j] = -128.0 + rgb_to_y(image[0][i][j], image[1][i][j], image[2][i][j])
        }
    }
    let down_h = if result.sample_type == Down420 {2} else {1};
    let down_w = if result.sample_type != Down444 {2} else {1};

    for i in 0..image[0].len() / down_h {
        for j in 0..image[0][0].len() / down_w {
            result.cb_channel[i][j] = -128.0 + rgb_to_cb(image[0][i* down_h][j * down_w], image[1][i* down_h][j * down_w], image[2][i* down_h][j * down_w]);
            result.cr_channel[i][j] = -128.0 + rgb_to_cr(image[0][i* down_h][j * down_w], image[1][i* down_h][j * down_w], image[2][i* down_h][j * down_w]);

        }
    }

    return result;

}

pub fn dct_and_quantize_image(mut image : JPEGContainer, Qf : f64) -> JPEGContainer{


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
        threads.push(
            thread::spawn(move ||{
                let mut image_channel = receiver.recv().unwrap();
                dct_and_quantization_over_channel(&mut image_channel, Qf, channel > 0);
                sender.send(image_channel).unwrap();
            })
        )
    }

    image.y_channel = receivers[0].recv().unwrap();
    image.cb_channel = receivers[1].recv().unwrap();
    image.cr_channel = receivers[2].recv().unwrap();
    return image;
}

fn dct_and_quantization_over_channel(channel : &mut Vec<Vec<f64>>, Qf : f64, chroma : bool){

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
}





#[cfg(test)]
mod test{
    use super::*;
    use crate::{JPEGContainer,Sampling::*,Sampling};
    use crate::colorspace_transforms::*;

    #[test]
    fn test_color_transform_and_downsample_step(){


        let test_case = vec![vec![vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0];8];3];
        let compare_target = test_case.clone();

        let result1 = color_transform_and_dowsample_image(test_case.clone(), Down444);

        for i in 0..8{
            for j in 0..8{
                assert_eq!(result1.y_channel[i][j],rgb_to_y(test_case[0][i][j], test_case[1][i][j], test_case[2][i][j]));
                assert_eq!(result1.cb_channel[i][j],rgb_to_cb(test_case[0][i][j], test_case[1][i][j], test_case[2][i][j]));
                assert_eq!(result1.cr_channel[i][j],rgb_to_cr(test_case[0][i][j], test_case[1][i][j], test_case[2][i][j]));

            }
        }

        let result2 = color_transform_and_dowsample_image(test_case.clone(), Down422);


        for i in 0..8{
            for j in 0..8{
                assert_eq!(result2.y_channel[i][j],rgb_to_y(test_case[0][i][j], test_case[1][i][j], test_case[2][i][j]));
            }
        }

        for i in 0..8{
            for j in 0..4{
                assert_eq!(result2.cb_channel[i][j],rgb_to_cb(test_case[0][i][j * 2], test_case[1][i][j * 2], test_case[2][i][j * 2]));
                assert_eq!(result2.cr_channel[i][j],rgb_to_cb(test_case[0][i][j * 2], test_case[1][i][j * 2], test_case[2][i][j * 2]));
            }
        }

        let result3 = color_transform_and_dowsample_image(test_case.clone(), Down422);


        for i in 0..8{
            for j in 0..8{
                assert_eq!(result2.y_channel[i][j],rgb_to_y(test_case[0][i][j], test_case[1][i][j], test_case[2][i][j]));
            }
        }

        for i in 0..4{
            for j in 0..4{
                assert_eq!(result3.cb_channel[i][j],rgb_to_cb(test_case[0][i * 2][j * 2], test_case[1][i * 2][j * 2], test_case[2][i * 2][j * 2]));
                assert_eq!(result3.cr_channel[i][j],rgb_to_cb(test_case[0][i * 2][j * 2], test_case[1 * 2][i][j * 2], test_case[2][i * 2][j * 2]));
            }
        }

    }



}