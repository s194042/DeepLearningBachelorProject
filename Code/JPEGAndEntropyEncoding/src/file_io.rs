use crate::entropy_encoding_step::JPEGSymbol;



pub fn freq_vec_to_bin(freq_vec : Vec<(JPEGSymbol,u64)>) {
    let mut bin_result : Vec<u8> = vec![];
    let mut buffer : u8 = 0;
    let mut buffer_size = 0;
    for (symbol,freq) in freq_vec.iter(){
        match *symbol{
            JPEGSymbol::Zeros(x) => {buffer = 0;push_to_buffer(&mut bin_result, 3, &mut buffer, &mut buffer_size);
                                         buffer = x;push_to_buffer(&mut bin_result, 8, &mut buffer, &mut buffer_size)},
            JPEGSymbol::Symbol(x) => {buffer = 1;push_to_buffer(&mut bin_result, 3, &mut buffer, &mut buffer_size)},
            JPEGSymbol::EOB => {buffer = 2;push_to_buffer(&mut bin_result, 3, &mut buffer, &mut buffer_size)},
            JPEGSymbol::CHANNEL_MARKER => {buffer = 3; push_to_buffer(&mut bin_result, 3, &mut buffer, &mut buffer_size)},
            JPEGSymbol::EOF => {buffer = 4; push_to_buffer(&mut bin_result, 3, &mut buffer, &mut buffer_size)},
        }
    }
}


pub fn push_to_buffer(bin_result : &mut Vec<u8>, count : usize, buffer : &mut u8, buffer_size : &mut usize){
    if bin_result.is_empty(){
        bin_result.push(0);
    }
    if count >= *buffer_size {
        
    }

}