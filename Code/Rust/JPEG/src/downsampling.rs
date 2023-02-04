

pub fn down444(image : &Vec<Vec<Vec<usize>>>) -> Vec<Vec<Vec<f64>>>{
    let (h,w) = (image[0].len(),image[0][0].len());
    let mut res = vec![vec![vec![0.0;w];h];3];
    
    for i in 0..3{
        for j in 0..h{
            for k in 0..w{
                res[i][j][k] = image[i][j][k] as f64;
            }
        }
    }

    return res;
}

pub fn down422(image : &Vec<Vec<Vec<usize>>>) -> Vec<Vec<Vec<f64>>>{
    let (h,w) = (image[0].len(),image[0][0].len());
    let mut res = Vec::new();
    res.push(vec![vec![0.0;w];h]);
    res.push(vec![vec![0.0;w/2];h]);
    res.push(vec![vec![0.0;w/2];h]);

    // copy Y component
    for i in 0..h{
        for j in 0..w{
            res[0][i][j] = image[0][i][j] as f64;
        }
    }
    // subsample Cb and Cr components
    for i in 1..3{
        for j in 0..h{
            for k in 0..(w/2){
                res[i][j][k] = image[i][j][k * 2] as f64;
            }
        }        
    }



    return res;

}

pub fn down420(image : &Vec<Vec<Vec<usize>>>) -> Vec<Vec<Vec<f64>>>{
    let (h,w) = (image[0].len(),image[0][0].len());
    let mut res = Vec::new();
    res.push(vec![vec![0.0;w];h]);
    res.push(vec![vec![0.0;w/2];h/2]);
    res.push(vec![vec![0.0;w/2];h/2]);

    // copy Y component
    for i in 0..h{
        for j in 0..w{
            res[0][i][j] = image[0][i][j] as f64;
        }
    }
    // subsample Cb and Cr components
    for i in 1..3{
        for j in 0..(h/2){
            for k in 0..(w/2){
                res[i][j][k] = image[i][j * 2][k * 2] as f64;
            }
        }        
    }

    return res;
}



pub fn up444(image : &Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<usize>>>{
    let (h,w) = (image[0].len(),image[0][0].len());
    let mut res = vec![vec![vec![0;w];h];3];
    
    for i in 0..3{
        for j in 0..h{
            for k in 0..w{
                res[i][j][k] = image[i][j][k] as usize;
            }
        }
    }

    return res;

}

pub fn up422(image : &Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<usize>>>{
    let (h,w) = (image[0].len(), image[0][0].len());
    let mut res = vec![vec![vec![0;w];h];3];

    for i in 0..h{
        for j in 0..w{
            res[0][i][j] = image[0][i][j] as usize;
        }
    }

    for i in 1..3{
        for j in 0..h{
            for k in 0..w{
                res[i][j][k] = image[i][j][k / 2] as usize;
            }
        }
    }

    return res;
}


pub fn up420(image : &Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<usize>>>{
    let (h,w) = (image[0].len() , image[0][0].len());
    let mut res = vec![vec![vec![0;w];h];3];

    for i in 0..h{
        for j in 0..w{
            res[0][i][j] = image[0][i][j] as usize;
        }
    }

    for i in 1..3{
        for j in 0..h{
            for k in 0..w{
                res[i][j][k] = image[i][j / 2][k / 2] as usize;
            }
        }
    }

    return res;

}




#[cfg(test)]
mod tests{

    use super::*;

    #[test]
    fn test_down444(){
        let t = vec![vec![vec![1;4];4];3];
        let mut after = down444(&t);
        assert_eq!(vec![vec![vec![1.0;4];4];3],after);
    }

    #[test]
    fn test_down422(){
        let t = vec![vec![vec![1,2,3,4];4];3];
        let mut after = down422(&t);
        assert_eq!(vec![vec![1.0,2.0,3.0,4.0];4],after[0]);
        assert_eq!(vec![vec![1.0,3.0];4],after[1]);
        assert_eq!(vec![vec![1.0,3.0];4],after[2]);

    }


    #[test]
    fn test_down420(){
        let t = vec![vec![vec![1,2,3,4];4];3];
        let mut after = down420(&t);
        assert_eq!(vec![vec![1.0,2.0,3.0,4.0];4],after[0]);
        assert_eq!(vec![vec![1.0,3.0];2],after[1]);
        assert_eq!(vec![vec![1.0,3.0];2],after[2]);

    } 

    #[test]
    fn test_up444(){
        let t = vec![vec![vec![1.0;4];4];3];
        let mut after = up444(&t);
        assert_eq!(vec![vec![vec![1;4];4];3],after);
    }

    #[test]
    fn test_up422(){
        let t = vec![vec![vec![1.0,2.0,3.0,4.0];4],vec![vec![1.0,3.0];4],vec![vec![1.0,3.0];4]];
        let mut after = up422(&t);
        assert_eq!(vec![vec![vec![1,2,3,4];4],vec![vec![1,1,3,3];4],vec![vec![1,1,3,3];4]],after);
    }

    #[test]
    fn test_up420(){
        let t = vec![vec![vec![1.0,2.0,3.0,4.0];4],vec![vec![1.0,3.0];2],vec![vec![1.0,3.0];2]];
        let mut after = up420(&t);
        assert_eq!(vec![vec![vec![1,2,3,4];4],vec![vec![1,1,3,3];4],vec![vec![1,1,3,3];4]],after);
    }

    #[test]
    fn test_all(){
        let t = vec![vec![vec![1,3,4,2,7,5];6];3];
        assert_eq!(up444(&down444(&t)),t, "444");
        assert_eq!(up422(&down422(&t)),vec![vec![vec![1,3,4,2,7,5];6],vec![vec![1,1,4,4,7,7];6],vec![vec![1,1,4,4,7,7];6]], "422");
        assert_eq!(up420(&down420(&t)),vec![vec![vec![1,3,4,2,7,5];6],vec![vec![1,1,4,4,7,7];6],vec![vec![1,1,4,4,7,7];6]], "420");
    }

}