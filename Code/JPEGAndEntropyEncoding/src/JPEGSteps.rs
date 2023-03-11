use crate::{JPEGContainer,Sampling::*,Sampling};
use crate::colorspace_transforms::*;



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
            result.y_channel[i][j] = rgb_to_y(image[0][i][j], image[1][i][j], image[2][i][j])
        }
    }
    let down_h = if result.sample_type == Down420 {2} else {1};
    let down_w = if result.sample_type != Down444 {2} else {1};

    for i in 0..image[0].len() / down_h {
        for j in 0..image[0][0].len() / down_w {
            result.cb_channel[i][j] = rgb_to_cb(image[0][i* down_h][j * down_w], image[1][i* down_h][j * down_w], image[2][i* down_h][j * down_w]);
            result.cr_channel[i][j] = rgb_to_cr(image[0][i* down_h][j * down_w], image[1][i* down_h][j * down_w], image[2][i* down_h][j * down_w]);

        }
    }

    return result;

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