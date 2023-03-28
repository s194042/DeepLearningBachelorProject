use std::{fs::File,io::prelude::*,env};

use crate::{entropy_encoding_step::JPEGSymbol, arithmetic_encoding::ArithEncoder};



pub struct BinaryBuffer{
    buffer : Vec<u8>,
    write_index : usize,
    read_index : usize,
}


pub fn arithmetic_encoding_to_file(arith_encoder : &ArithEncoder<JPEGSymbol>, original_size : (usize,usize), path : &str){

    let buffer = encode_buffer(&arith_encoder.model_to_freq_vec(), &arith_encoder.encoded_message,original_size);
    write_to_bin(&buffer, path)

}

pub fn arithmetic_encoding_from_file(path : &str) -> ((i32,i32),ArithEncoder<JPEGSymbol>){
    let mut buffer = load_from_bin(path);
    let ((h,w),freq_vec,encoded_message) = decode_buffer(&mut buffer);

    return ((h,w),ArithEncoder::from_encoded_message(freq_vec, encoded_message, JPEGSymbol::EOF));
}


pub fn write_to_bin(buffer : &BinaryBuffer, path : &str){
    let mut file = File::create(path).unwrap();
    file.write_all(&buffer.buffer).unwrap();
}


pub fn load_from_bin(path : &str) -> BinaryBuffer{
    let mut file = File::open(path).unwrap();
    // read the same file back into a Vec of bytes
    let mut buffer = Vec::<u8>::new();
    file.read_to_end(&mut buffer).unwrap();
    return BinaryBuffer{
        buffer,
        write_index : 0,
        read_index : 0,
    }
}



pub fn encode_buffer(freq_vec : &Vec<(JPEGSymbol,i32)>, encoded_message : &Vec<u8>, original_size : (usize,usize)) -> BinaryBuffer{

    let mut buffer_result = BinaryBuffer{
        buffer : vec![],
        write_index : 0,
        read_index : 0
    };

    let buffer = &mut buffer_result;

    push_i32_to_buffer(buffer, original_size.0 as i32);
    push_i32_to_buffer(buffer, original_size.1 as i32);

    for (symbol,freq) in freq_vec.iter(){
        match *symbol{
            JPEGSymbol::Zeros(x) => {push_to_buffer(buffer, 3,  0);
                                         push_to_buffer(buffer, 8,  x)},
            JPEGSymbol::Symbol(x) => {push_to_buffer(buffer, 3,  1);
                                           push_i32_to_buffer(buffer, x)},
            JPEGSymbol::EOB => {push_to_buffer(buffer, 3,  2)},
            JPEGSymbol::CHANNEL_MARKER => { push_to_buffer(buffer, 3,  3)},
            JPEGSymbol::EOF => {push_to_buffer(buffer, 3,  4)},
        }
        push_i32_to_buffer(buffer, *freq);
    }
    for bit in encoded_message{
        push_to_buffer(buffer, 1, *bit);
    }
    return buffer_result;
}


pub fn decode_buffer(buffer : &mut BinaryBuffer) -> ((i32,i32),Vec<(JPEGSymbol,i32)>,Vec<u8>){
    let mut freq_vec = vec![];

    let original_size = (read_from_buffer(buffer, 32) as i32,read_from_buffer(buffer, 32) as i32);

    loop{
        let sym = read_from_buffer(buffer, 3);
        match sym{
            0 => freq_vec.push((JPEGSymbol::Zeros(read_from_buffer(buffer, 8) as u8),read_from_buffer(buffer, 32) as i32)),
            1 => freq_vec.push((JPEGSymbol::Symbol(read_from_buffer(buffer, 32) as i32),read_from_buffer(buffer, 32) as i32)),
            2 => freq_vec.push((JPEGSymbol::CHANNEL_MARKER,read_from_buffer(buffer, 32) as i32)),
            3 => freq_vec.push((JPEGSymbol::EOB,read_from_buffer(buffer, 32) as i32)),
            4 => {freq_vec.push((JPEGSymbol::EOF,read_from_buffer(buffer, 32) as i32)); break},
            _ => panic!("Unrecognized symbol when decoding buffer"),
        }
    }

    let mut encoded_message = vec![];

    while buffer.read_index < buffer.buffer.len() * 8{
        encoded_message.push(read_from_buffer(buffer, 1) as u8);
    }
    return (original_size,freq_vec,encoded_message);
}

pub fn read_from_buffer(buffer : &mut BinaryBuffer, count : usize) -> u64{
    let mut result : u64 = 0;

    for i in 0..count{
        result <<= 1;
        result += if buffer.buffer[buffer.read_index / 8] & (1 << (7 - (buffer.read_index % 8))) >= 1 {1} else {0};
        buffer.read_index += 1;
    }

    return result;
}

pub fn push_i32_to_buffer(buffer : &mut BinaryBuffer, to_push32 : i32){
    let mut to_push :u8 = 0;
    for i in 0..4{
        to_push += (to_push32 >> (32 - 8 * (i+1))) as u8;
        push_to_buffer(buffer, 8, to_push);
        to_push = 0;
    }
}


pub fn push_to_buffer(buffer : &mut BinaryBuffer, mut count : usize, to_push : u8){
    let mut to_push = to_push;
    if buffer.buffer.is_empty() || buffer.write_index == 8{
        buffer.buffer.push(0);
        buffer.write_index = 0;
    }
    let mut last_elem = buffer.buffer.len() - 1;
    if count > 8 - buffer.write_index {
        let buffer_shift = count - (8 - buffer.write_index);
        buffer.buffer[last_elem] += (to_push & !((2_u32.pow(buffer_shift as u32) - 1) as u8) ) >> buffer_shift;
        count -= 8 - buffer.write_index;
        buffer.buffer.push(0);
        last_elem += 1;
        to_push &= 2_u8.pow(count as u32) - 1;
        buffer.write_index = 0;
    }
    buffer.buffer[last_elem] += to_push << (8 - (buffer.write_index + count));
    buffer.write_index += count;

}



#[cfg(test)]
mod test{
    use std::result;

    use super::*;


    #[test]
    fn test_push_to_buffer(){
        let mut buffer = BinaryBuffer { buffer: vec![], write_index: 0, read_index: 0 };
        let target : Vec<u8> = vec![0b10010010,0b01001001,0b00100100,0b10010011,0b01000000];
        
        for i in 0..10{
            push_to_buffer(&mut buffer, 3, 4);
        }
        push_to_buffer(&mut buffer, 4, 13);


        assert_eq!(buffer.buffer,target);

    }

    #[test]
    fn test_push_i32_to_buffer(){
        let mut buffer = BinaryBuffer { buffer: vec![], write_index: 0, read_index: 0 };
        let test_case = 0b01001101_01001110_10101010;
        let target = vec![0b00000000,0b01001101,0b01001110,0b10101010];

        push_i32_to_buffer(&mut buffer, test_case);

        assert_eq!(buffer.buffer,target)

    }

    #[test]
    fn test_read_from_buffer(){
        let target = 0b01001101_01001110_10101010;

        let mut buffer = BinaryBuffer{
            buffer : vec![0b00000000,0b01001101,0b01001110,0b10101010],
            write_index : 0,
            read_index :0,
        };

        let mut buffer_index = 0;
        let result = read_from_buffer(&mut buffer, 32);
        println!("{:b}",result);
        assert_eq!(target,result);
    }

    #[test]
    fn test_encode_decode_buffer(){
        use JPEGSymbol::*;
        let freq_vec = vec![(Zeros(8),20),(Symbol(-400),30),(EOB,2),(EOF,1)];
        let encoded_message = vec![1,1,1,0,0,0,1,1,0,0,1,0,1,0,1,1];
        let mut buffer = encode_buffer(&freq_vec, &encoded_message,(212,234));
        let ((h,w),freq_vec_decoded,encoded_message_decoded) = decode_buffer(&mut buffer);
        assert_eq!(freq_vec,freq_vec_decoded);
        assert_eq!(encoded_message,encoded_message_decoded);
        assert_eq!((h,w),(212,234));
    }

}