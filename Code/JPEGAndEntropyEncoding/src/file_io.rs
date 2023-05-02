use std::{fs::File,io::prelude::*,env};

use pyo3::buffer;

use crate::{entropy_encoding_step::JPEGSymbol, arithmetic_encoding::ArithEncoder,Sampling};



pub struct BinaryBuffer{
    buffer : Vec<u8>,
    write_index : usize,
    read_index : usize,
}

pub struct AuxiliaryData{
    pub original_size : (usize,usize),
    pub Qf : f64,
    pub sample_type : Sampling
}


pub fn arithmetic_encoding_to_file(arith_encoder : &ArithEncoder<JPEGSymbol>,aux_data : AuxiliaryData , path : &str){

    let buffer = encode_JPEG_buffer(&arith_encoder.freq_count, &arith_encoder.encoded_message,&aux_data);
    write_to_bin(&buffer, path)

}

pub fn arithmetic_encoding_from_file(path : &str) -> (AuxiliaryData,ArithEncoder<JPEGSymbol>){
    let mut buffer = load_from_bin(path);
    let (aux_data,freq_vec,encoded_message) = decode_JPEG_buffer(&mut buffer);
    return (aux_data,ArithEncoder::from_encoded_message(freq_vec, encoded_message, JPEGSymbol::EOF));
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



pub fn encode_JPEG_buffer(freq_vec : &Vec<(JPEGSymbol,i32)>, encoded_message : &Vec<u8>, aux_data : &AuxiliaryData) -> BinaryBuffer{

    let mut buffer_result = BinaryBuffer{
        buffer : vec![],
        write_index : 0,
        read_index : 0
    };

    let buffer = &mut buffer_result;

    push_i32_to_buffer(buffer, aux_data.original_size.0 as i32);
    push_i32_to_buffer(buffer, aux_data.original_size.1 as i32);
    push_f64_to_buffer(buffer, aux_data.Qf);
    push_to_buffer(buffer, 2, match aux_data.sample_type {
        Sampling::Down420 => 0,
        Sampling::Down422 => 1,
        Sampling::Down444 => 2,
    });


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
    push_to_buffer(buffer, 3, 5);
    for bit in encoded_message{
        push_to_buffer(buffer, 1, *bit);
    }
    return buffer_result;
}


pub fn decode_JPEG_buffer(buffer : &mut BinaryBuffer) -> (AuxiliaryData,Vec<(JPEGSymbol,i32)>,Vec<u8>){
    let mut freq_vec = vec![];

    let aux_data = AuxiliaryData{
        original_size : (read_from_buffer(buffer, 32) as usize,read_from_buffer(buffer, 32) as usize),
        Qf : f64::from_bits(read_from_buffer(buffer, 64)),
        sample_type : match read_from_buffer(buffer, 2) {
            0 => Sampling::Down420,
            1 => Sampling::Down422,
            2 => Sampling::Down444,
            _ => panic!("Not a valid sample type")
        }
    };

    loop{
        let sym = read_from_buffer(buffer, 3);
        match sym{
            0 => freq_vec.push((JPEGSymbol::Zeros(read_from_buffer(buffer, 8) as u8),read_from_buffer(buffer, 32) as i32)),
            1 => freq_vec.push((JPEGSymbol::Symbol(read_from_buffer(buffer, 32) as i32),read_from_buffer(buffer, 32) as i32)),
            2 => freq_vec.push((JPEGSymbol::EOB,read_from_buffer(buffer, 32) as i32)),
            3 => freq_vec.push((JPEGSymbol::CHANNEL_MARKER,read_from_buffer(buffer, 32) as i32)),
            4 => freq_vec.push((JPEGSymbol::EOF,read_from_buffer(buffer, 32) as i32)),
            5 => {break},
            _ => panic!("Unrecognized symbol when decoding buffer"),
        }
    }

    let mut encoded_message = vec![];

    while buffer.read_index < buffer.buffer.len() * 8{
        encoded_message.push(read_from_buffer(buffer, 1) as u8);
    }
    return (aux_data,freq_vec,encoded_message);
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

pub fn push_f64_to_buffer(buffer : &mut BinaryBuffer, to_push64 : f64){
    let mut to_push : u8 = 0;
    let to_push64 = to_push64.to_bits();
    for i in 0..8{
        to_push += (to_push64 >> (64 - 8 * (i+1))) as u8;
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
        let aux_data = AuxiliaryData{
            Qf : 0.5,
            original_size : (212,345),
            sample_type : Sampling::Down420
        };
        let mut buffer = encode_JPEG_buffer(&freq_vec, &encoded_message,&aux_data);
        let (aux_data_decoded,freq_vec_decoded,encoded_message_decoded) = decode_JPEG_buffer(&mut buffer);
        assert_eq!(freq_vec,freq_vec_decoded);
        assert_eq!(encoded_message,encoded_message_decoded[0..encoded_message.len()]);
        assert_eq!(aux_data.original_size,aux_data_decoded.original_size);
        assert_eq!(aux_data.sample_type,aux_data_decoded.sample_type);
        assert_eq!(aux_data.Qf,aux_data_decoded.Qf);
    }

}