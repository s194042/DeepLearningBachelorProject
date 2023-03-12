




pub fn quantization_block(block : &mut Vec<Vec<f64>>, row : usize, column : usize, q_matrix : &Vec<Vec<isize>>) {

    for i in 0..8{
        for j in 0..8{
            block[row + i][column + j] = (block[row + i][column + j] / (q_matrix[i][j] as f64)).round();
        }
    }

}

pub fn inverse_quantization_block(block : &mut Vec<Vec<f64>>, row : usize, column : usize, q_matrix : &Vec<Vec<isize>>) {
    for i in 0..8{
        for j in 0..8{
            block[row + i][column + j] = block[row + i][column + j] * (q_matrix[i][j] as f64);
        }
    }
}


pub fn gen_quantize_lumi_matrix(Qf : f64) -> Vec<Vec<isize>>{
    let mut q_lumi = vec![vec![16, 11, 10, 16, 24, 40, 51, 61],
                          vec![12, 12, 14, 19, 26, 58, 60, 55],
                          vec![14, 13, 16, 24, 40, 57, 69, 56],
                          vec![14, 17, 22, 29, 51, 87, 80, 62],
                          vec![18, 22, 37, 56, 68, 109, 103, 77],
                          vec![24, 35, 55, 64, 81, 104, 113, 92],
                          vec![49, 64, 78, 87, 103, 121, 120, 101],
                          vec![72, 92, 95, 98, 112, 100, 103, 99]];

    let S = if Qf >= 50.0 {200.0 - 2.0 * Qf} else {5000.0 / Qf};

    for i in 0..8{
        for j in 0..8{
            q_lumi[i][j] = ((S * (q_lumi[i][j] as f64) + 50.0) / 100.0).floor() as isize;
            if q_lumi[i][j] == 0{
                q_lumi[i][j] = 1;
            }
        }
    }

    return q_lumi;

}


pub fn gen_quantize_chroma_matrix(Qf : f64) -> Vec<Vec<isize>>{
    let mut q_chroma =    vec![vec![17, 18, 24, 47, 99, 99, 99, 99],
                               vec![18, 21, 26, 66, 99, 99, 99, 99],
                               vec![24, 26, 56, 99, 99, 99, 99, 99],
                               vec![47, 66, 99, 99, 99, 99, 99, 99],
                               vec![99, 99, 99, 99, 99, 99, 99, 99],
                               vec![99, 99, 99, 99, 99, 99, 99, 99],
                               vec![99, 99, 99, 99, 99, 99, 99, 99],
                               vec![99, 99, 99, 99, 99, 99, 99, 99]];

    let S = if Qf >= 50.0 {200.0 - 2.0 * Qf} else {5000.0 / Qf};

    for i in 0..8{
        for j in 0..8{
            q_chroma[i][j] = ((S * (q_chroma[i][j] as f64) + 50.0) / 100.0).floor() as isize;
            if q_chroma[i][j] == 0{
                q_chroma[i][j] = 1;
            }
        }
    }

    return q_chroma;
}



#[cfg(test)]

mod test{

    use super::*;
    use crate::dct::dct_block;

    #[test]
    fn test_quantize_matrix(){
        let mut q_chroma =    vec![vec![17, 18, 24, 47, 99, 99, 99, 99],
                               vec![18, 21, 26, 66, 99, 99, 99, 99],
                               vec![24, 26, 56, 99, 99, 99, 99, 99],
                               vec![47, 66, 99, 99, 99, 99, 99, 99],
                               vec![99, 99, 99, 99, 99, 99, 99, 99],
                               vec![99, 99, 99, 99, 99, 99, 99, 99],
                               vec![99, 99, 99, 99, 99, 99, 99, 99],
                               vec![99, 99, 99, 99, 99, 99, 99, 99]];
        let mut q_lumi = vec![vec![16, 11, 10, 16, 24, 40, 51, 61],
                          vec![12, 12, 14, 19, 26, 58, 60, 55],
                          vec![14, 13, 16, 24, 40, 57, 69, 56],
                          vec![14, 17, 22, 29, 51, 87, 80, 62],
                          vec![18, 22, 37, 56, 68, 109, 103, 77],
                          vec![24, 35, 55, 64, 81, 104, 113, 92],
                          vec![49, 64, 78, 87, 103, 121, 120, 101],
                          vec![72, 92, 95, 98, 112, 100, 103, 99]];

        assert_eq!(q_chroma,gen_quantize_chroma_matrix(50.0));
        assert_eq!(q_lumi,gen_quantize_lumi_matrix(50.0));
        assert_ne!(q_chroma,gen_quantize_chroma_matrix(25.0));
        assert_ne!(q_lumi,gen_quantize_lumi_matrix(25.0));
    }

    #[test]
    fn test_quantization_block(){
        let mut t = vec![vec![-76.0, -73.0, -67.0, -62.0, -58.0, -67.0, -64.0, -55.0],
                     vec![-65.0, -69.0, -73.0, -38.0, -19.0, -43.0, -59.0, -56.0],
                     vec![-66.0, -69.0, -60.0, -15.0,  16.0, -24.0, -62.0, -55.0],
                     vec![-65.0, -70.0, -57.0,  -6.0,  26.0, -22.0, -58.0, -59.0],
                     vec![-61.0, -67.0, -60.0, -24.0,  -2.0, -40.0, -60.0, -58.0],
                     vec![-49.0, -63.0, -68.0, -58.0, -51.0, -60.0, -70.0, -53.0],
                     vec![-43.0, -57.0, -64.0, -69.0, -73.0, -67.0, -63.0, -45.0],
                     vec![-41.0, -49.0, -59.0, -60.0, -63.0, -52.0, -50.0, -34.0]];
        t = dct_block(&t,0,0);

        let target = vec![vec![-26.0, -3.0, -6.0, 2.0, 2.0, -1.0, 0.0, 0.0], 
                          vec![0.0, -2.0, -4.0, 1.0, 1.0, 0.0, 0.0, 0.0], 
                          vec![-3.0, 1.0, 5.0, -1.0, -1.0, 0.0, 0.0, 0.0], 
                          vec![-3.0, 1.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0], 
                          vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                          vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                          vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                          vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
                                
        quantization_block(&mut t, 0, 0, &gen_quantize_lumi_matrix(50.0));

        assert_eq!(t,target);
    }

}