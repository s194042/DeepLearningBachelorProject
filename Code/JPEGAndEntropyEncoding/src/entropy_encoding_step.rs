/*
pub enum JPEGSymbol {
    Count(
}

// todo! fix DC components
pub fn run_length_encoding_block(block : Vec<Vec<f64>>, column : usize, row : usize){
    let zigzag_sequence = generate_zigzag_sequence(8);
    let mut result = vec![];





}
*/

fn generate_zigzag_sequence(size : usize) -> Vec<(usize,usize)>{
    let mut zigzag_sequence = vec![];
    let (mut column, mut row) = (0,0);
    for i in 0..8{
        zigzag_sequence.push((row,column));
        for j in 0..i{
            if i % 2 == 0{
                column += 1;
                row -= 1;
            }else{
                column -= 1;
                row += 1;
            }
            zigzag_sequence.push((row,column))
        }
        row += if i % 2 != 0 {1} else {0};
        column += if i % 2 != 0 {0} else {1};
    }
        row = 7;
    for i in (0..7).rev(){
        row += if i % 2 != 0 {1} else {0};
        column += if i % 2 != 0 {0} else {1};
        zigzag_sequence.push((row,column));
        for j in 0..i{
            if i % 2 == 0{
                column += 1;
                row -= 1;
            }else{
                column -= 1;
                row += 1;
            }
            zigzag_sequence.push((row,column))
        }
    }
    return zigzag_sequence;
}


#[cfg(test)]

mod test{
    use super::*;

    #[test]
    fn test_zigzag(){
        let mut test = vec![vec![0;8];8];
        let seq = generate_zigzag_sequence(8);

        let target = vec![vec![1, 2, 6, 7, 15, 16, 28, 29],
        vec![3, 5, 8, 14, 17, 27, 30, 43],
        vec![4, 9, 13, 18, 26, 31, 42, 44],
        vec![10, 12, 19, 25, 32, 41, 45, 54],
        vec![11, 20, 24, 33, 40, 46, 53, 55],
        vec![21, 23, 34, 39, 47, 52, 56, 61],
        vec![22, 35, 38, 48, 51, 57, 60, 62],
        vec![36, 37, 49, 50, 58, 59, 63, 64]];
        

        let mut counter = 1;
        for (i,j) in seq.iter(){
            test[*i][*j] = counter;
            counter += 1; 
        }

        for i in 0..8{
            println!("{:?}",test[i]);
        }
        assert_eq!(test,target);
    }
}