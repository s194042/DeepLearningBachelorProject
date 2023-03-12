

fn alpha(i : f64) -> f64{
    if i <= 0.0 {1.0/(2.0_f64).sqrt()} else {1.0}
}


fn dct_coeff(block : &Vec<Vec<f64>>, u : f64, v : f64, row : usize, column : usize) -> f64{
    let mut res = 0.0;
    for i in row..row+8{
        for j in column..column+8{
            res += (block[i][j]) * ((2*i + 1) as f64 * u * std::f64::consts::PI / 16.0 ).cos() * ((2*j + 1) as f64 * v * std::f64::consts::PI / 16.0 ).cos();
        }
    }

    return 1.0 / 4.0 * alpha(u) * alpha(v) * res;
}

pub fn dct_block(block : &Vec<Vec<f64>>, row : usize, column : usize) -> Vec<Vec<f64>>{
    let mut res = vec![vec![0.0;8];8];
    for i in 0..8{
        for j in 0..8{
            res[i][j] = dct_coeff(block, i as f64, j as f64, row,column);
        }
    }

    return res;
}

fn inv_dct_coeff(block : &Vec<Vec<f64>>, x : usize, y : usize, row : usize, column : usize) -> f64{
    let mut res = 0.0;
    for i in row..row+8{
        for j in column..column+8{
            res += 1.0 / 4.0 * alpha(i as f64) * alpha(j as f64) * (block[i][j]) * ((2*x + 1) as f64 * (i as f64) * std::f64::consts::PI / 16.0 ).cos() * ((2*y + 1) as f64 * (j as f64) * std::f64::consts::PI / 16.0 ).cos();
        }
    }

    return res;
}

fn inv_dct_block(block : &Vec<Vec<f64>>, row : usize, column : usize) -> Vec<Vec<f64>>{
    let mut res = vec![vec![0.0;8];8];
    for i in 0..8{
        for j in 0..8{
            res[i][j] = inv_dct_coeff(block, i, j, row, column);
        }
    }

    return res;
}



#[cfg(test)]

mod tests{

    use super::*;

    #[test]
    fn test_dct_coeff(){
        let t = vec![vec![-76.0, -73.0, -67.0, -62.0, -58.0, -67.0, -64.0, -55.0],
                     vec![-65.0, -69.0, -73.0, -38.0, -19.0, -43.0, -59.0, -56.0],
                     vec![-66.0, -69.0, -60.0, -15.0,  16.0, -24.0, -62.0, -55.0],
                     vec![-65.0, -70.0, -57.0,  -6.0,  26.0, -22.0, -58.0, -59.0],
                     vec![-61.0, -67.0, -60.0, -24.0,  -2.0, -40.0, -60.0, -58.0],
                     vec![-49.0, -63.0, -68.0, -58.0, -51.0, -60.0, -70.0, -53.0],
                     vec![-43.0, -57.0, -64.0, -69.0, -73.0, -67.0, -63.0, -45.0],
                     vec![-41.0, -49.0, -59.0, -60.0, -63.0, -52.0, -50.0, -34.0]];


        assert_eq!(dct_coeff(&t,0.0,0.0,0,0).round() as isize, -415);
    }

    #[test]
    #[ignore]
    fn test_dct_block(){
        let t = vec![vec![-76.0, -73.0, -67.0, -62.0, -58.0, -67.0, -64.0, -55.0],
                     vec![-65.0, -69.0, -73.0, -38.0, -19.0, -43.0, -59.0, -56.0],
                     vec![-66.0, -69.0, -60.0, -15.0,  16.0, -24.0, -62.0, -55.0],
                     vec![-65.0, -70.0, -57.0,  -6.0,  26.0, -22.0, -58.0, -59.0],
                     vec![-61.0, -67.0, -60.0, -24.0,  -2.0, -40.0, -60.0, -58.0],
                     vec![-49.0, -63.0, -68.0, -58.0, -51.0, -60.0, -70.0, -53.0],
                     vec![-43.0, -57.0, -64.0, -69.0, -73.0, -67.0, -63.0, -45.0],
                     vec![-41.0, -49.0, -59.0, -60.0, -63.0, -52.0, -50.0, -34.0]];

        let after = dct_block(&t,0,0);
        for i in 0..8{
            println!("{:?}",after[i]);  
        }

        assert!(false)
    }

    #[test]
    fn test_inv_dct_coeff(){
        let t = vec![vec![-76.0, -73.0, -67.0, -62.0, -58.0, -67.0, -64.0, -55.0],
                     vec![-65.0, -69.0, -73.0, -38.0, -19.0, -43.0, -59.0, -56.0],
                     vec![-66.0, -69.0, -60.0, -15.0,  16.0, -24.0, -62.0, -55.0],
                     vec![-65.0, -70.0, -57.0,  -6.0,  26.0, -22.0, -58.0, -59.0],
                     vec![-61.0, -67.0, -60.0, -24.0,  -2.0, -40.0, -60.0, -58.0],
                     vec![-49.0, -63.0, -68.0, -58.0, -51.0, -60.0, -70.0, -53.0],
                     vec![-43.0, -57.0, -64.0, -69.0, -73.0, -67.0, -63.0, -45.0],
                     vec![-41.0, -49.0, -59.0, -60.0, -63.0, -52.0, -50.0, -34.0]];

        assert_eq!(t[0][0] as isize, inv_dct_coeff(&dct_block(&t,0,0),0,0,0,0).round() as isize);
    }


    #[test]
    fn test_inv_dct_block(){
        let t = vec![vec![-76.0, -73.0, -67.0, -62.0, -58.0, -67.0, -64.0, -55.0],
                     vec![-65.0, -69.0, -73.0, -38.0, -19.0, -43.0, -59.0, -56.0],
                     vec![-66.0, -69.0, -60.0, -15.0,  16.0, -24.0, -62.0, -55.0],
                     vec![-65.0, -70.0, -57.0,  -6.0,  26.0, -22.0, -58.0, -59.0],
                     vec![-61.0, -67.0, -60.0, -24.0,  -2.0, -40.0, -60.0, -58.0],
                     vec![-49.0, -63.0, -68.0, -58.0, -51.0, -60.0, -70.0, -53.0],
                     vec![-43.0, -57.0, -64.0, -69.0, -73.0, -67.0, -63.0, -45.0],
                     vec![-41.0, -49.0, -59.0, -60.0, -63.0, -52.0, -50.0, -34.0]];

        let after = inv_dct_block(&dct_block(&t,0,0),0,0);
        for i in 0..8{
            for j in 0..8{
                assert_eq!(t[i][j] as isize, after[i][j].round() as isize);
            }
        }
    }

}