const PRECISION : u64 = 32;
const WHOLE : u64 = (2u64).pow(PRECISION as u32);
const HALF : u64 = WHOLE / 2;
const QUARTER : u64 = WHOLE / 4;


use std::{collections::{HashMap},hash::Hash, fmt::Debug,fs::File,io::prelude::*,env};
use rand::Rng;



pub struct ArithEncoder<T : Eq + Hash + Copy + Debug>{
    pub message : Vec<T>,
    pub model : Model<T>,
    pub encoded_message : Vec<u8>,
}


impl<T : Eq + Hash + Copy + Debug> ArithEncoder<T> {
    pub fn new(message : Vec<T>, eof : T) -> ArithEncoder<T>{
        let freq_count = message.clone().into_iter().fold(&mut HashMap::new(), |s,x| if s.contains_key(&x) {s.insert(x, s.get(&x).unwrap() + 1); s} else{s.insert(x, 1); s}).into_iter().map(|(k,v)| (*k,*v)).collect();
        let model = Model::new(&freq_count, eof);
        ArithEncoder { message, model, encoded_message: vec![] }
    }

    pub fn set_new_message(&mut self, message : Vec<T>, eof : T) {
        let freq_count = message.clone().into_iter().fold(&mut HashMap::new(), |s,x| if s.contains_key(&x) {s.insert(x, s.get(&x).unwrap() + 1); s} else{s.insert(x, 1); s}).into_iter().map(|(k,v)| (*k,*v)).collect();
        let model = Model::new(&freq_count, eof);
        self.message = message;
        self.model = model;
        self.encoded_message = vec![];
    }

    pub fn encode(&mut self){
        self.encoded_message = encode(&self.model, &self.message)
    }

    pub fn decode(&mut self) -> Vec<T>{
        let res = decode(&self.encoded_message, &self.model);
        assert_eq!(res,self.message);
        res
    }

    pub fn write_to_bin(&mut self, path : &str){
        let mut file = File::create(path).unwrap();
        file.write_all(&self.message_to_bytes()).unwrap();
    }


    pub fn load_from_bin(&mut self, path : &str){
        let mut file = File::open(path).unwrap();
        // read the same file back into a Vec of bytes
        let mut buffer = Vec::<u8>::new();
        file.read_to_end(&mut buffer).unwrap();
        self.encoded_message = self.message_from_bytes(&buffer);
    }

    pub fn model_to_freq_vec(&self) -> Vec<(T,u64)>{
        let mut result = vec![];
        for key in self.model.ranges.keys(){
            let (lower,upper,freq) = self.model.get_prob(key);
            result.push((*key,upper - lower));
        }
        return result;
    }


    pub fn message_to_bytes(&mut self) -> Vec<u8>{
        let l = (self.encoded_message.len() as f64 / 8.0).ceil() as usize * 8;
        let mut res : Vec<u8> = vec![0;(l as f64 / 8.0).ceil() as usize];
        for i in 0..l{
            res[i / 8] <<= 1;
            if i < self.encoded_message.len() {res[i / 8] += self.encoded_message[i]};
        } 
        res
    }

    pub fn message_from_bytes(&mut self, bytes : &Vec<u8>) -> Vec<u8>{
        let l = bytes.len();
        let mut res : Vec<u8> = vec![0; l * 8];
        for i in 0..l{
            let mut current = bytes[i];
            for j in 0..8{
                res[i * 8 + 7 - j] = current & 1;
                current >>= 1;
            }
        }

        res
    }


}


pub struct Model<T : Eq + Hash + Copy + Debug>{
    pub ranges : HashMap<T,(u64,u64,u64)>,
    pub freq : u64,
    pub end : T,
}


impl<T : Eq + Hash + Copy + Debug> Model<T>{
    pub fn new(freq_vec : &Vec<(T,u64)>, end : T) -> Model<T>{
        let mut res : HashMap<T,(u64,u64,u64)> = HashMap::new();
        let mut s = 0;
        let denom = freq_vec.iter().fold(0,|s,(x,y)|  s + y);
        for (sym,freq) in freq_vec{
            res.insert(*sym,(s,s+freq,denom));
            s += freq;
        }

        let model = Model{
            ranges : res,
            freq : denom,
            end : end,
        };
        return model;
    }

    pub fn get_prob(&self, sym : &T) -> (u64,u64,u64){
        return *(self.ranges.get(sym).unwrap());
    }

    pub fn get_sym(&self, low : u64, range : u64, value : u64) -> T{
        for (sym,(lower,upper,_)) in &self.ranges{
            let low_0 = low + ((*lower) * range) / self.freq;
            let high_0 = low + ((*upper) * range) / self.freq;
            if low_0 <= value && value < high_0{
                return *sym;
            }
        } 
        panic!("No symbol matches the ranges");
    }
}




pub fn encode<T : Eq + Hash + Copy + Debug>(model : &Model<T>, message : &Vec<T>) -> Vec<u8>{
    let mut high : u64 = WHOLE;
    let mut low : u64 = 0;
    let mut res : Vec<u8> = vec![];
    let mut pending = 0;


    for sym in message{
        let range = high - low;
        let (lower,upper,denom) = model.get_prob(sym);
        high = low + (upper * range) / denom;
        low = low + (lower * range) / denom;
        while low > HALF || high < HALF{
            if low >= HALF{
                high = 2 * (high - HALF);
                low = 2 * (low - HALF);
                output_bits_and_pending(&mut res, 1, pending);
                pending = 0;
            }else if high < HALF{
                high *= 2;
                low *= 2;
                output_bits_and_pending(&mut res, 0, pending);
                pending = 0;
            }
        }
        while low > QUARTER && high < 3 * QUARTER{
            pending += 1;
            low = 2 * (low - QUARTER);
            high = 2 * (high - QUARTER);
        }
    }

    pending += 1;
    if low <= QUARTER{
        output_bits_and_pending(&mut res, 0, pending)
    }else{
        output_bits_and_pending(&mut res, 1, pending)
    }

    return res;
}

fn output_bits_and_pending(output : &mut Vec<u8>, bit : u8, pending : u32) {
    output.push(bit);
    let mut pending = pending;
    while pending > 0{
        pending -= 1;
        output.push(if bit == 1 {0} else {1});
    }
}


pub fn decode<T : Eq + Hash + Copy + Debug>(message : &Vec<u8>, model : &Model<T>) -> Vec<T>{

    let mut high : u64 = WHOLE;
    let mut value : u64 = 0;
    let mut low : u64 = 0;
    let mut index = PRECISION as usize;
    let mut res = vec![];

    for i in 0..32{
        value <<= 1;
        if i < message.len(){
            value += message[i] as u64;
        }
    }


    loop{
        let range = high - low;
        let sym = model.get_sym(low,range,value);
        let (lower,upper,denom) = model.get_prob(&sym);
        high = low + (upper * range) / denom;
        low = low + (lower * range) / denom;
        res.push(sym);
        if sym == model.end {break}

        while low > HALF || high < HALF{
            if low > HALF{
                high = 2 * (high - HALF);
                low = 2 * (low - HALF);
                value = 2 * (value - HALF);
            }else if high < HALF{
                high *= 2;
                low *= 2;
                value *= 2;
            }
            if index < message.len() && message[index] == 1{
                value += 1;
            }
            index += 1;
        }
        while low > QUARTER && high < 3 * QUARTER{
            low = 2 * (low - QUARTER);
            high = 2 * (high - QUARTER);
            value = 2 * (value - QUARTER);
            if index < message.len() && message[index] == 1{
                value += 1;
            }
            index += 1;
        }
    }


    return res;

}



pub fn calculate_entropy<T : Hash + Eq + Copy + Debug>(message : &Vec<T>, model : &Model<T>) -> f64{
    let mut H = 0.0;

    for symbol in model.ranges.keys(){
        let (low,high,denom) = model.get_prob(symbol);
        let p = (high - low) as f64 / denom as f64;
        H -= p * p.log2();
    }

    return H * message.len() as f64;
}


#[cfg(test)]
mod test{
    use super::*;

    #[test]
    fn test_create_model(){
        let res = Model::new(&vec![((1,3),10),((3,1),4),((7,1),20),((21,7),4)],(21,7));
        println!("{:?}",res.ranges);
        assert_eq!(HashMap::from([((1,3),(0,10,38)),
                                    ((3,1),(10,14,38)),
                                    ((7,1),(14,34,38)),
                                    ((21,7),(34,38,38))]),res.ranges);
    }

    #[test]
    fn test_encode(){
        let message = vec![5,1,5,5,5,5,5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,5,5,5,5,1,1,1,1,1,1,1,1,5,5,5,5,5,5,5,5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,5,5,5,5,5,5,5,5,5,1,1,1,1,5,5,5,1,1,1,5,5,5,5,5,5,5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,7];
        let model = Model::new(&vec![(1,(&message).into_iter().fold(0, |s,x| if *x == 1 {s + 1} else {s})),(5,(&message).into_iter().fold(0, |s,x| if *x == 5 {s + 1} else {s})),(7,(&message).into_iter().fold(0, |s,x| if *x == 7 {s + 1} else {s}))],7);
        let example = vec![1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1];
        let res = encode(&model,&message);
        println!("{:?} {:?}",example,res);
        let decoded = decode(&res, &model);
        println!("{:?}",decoded);
        println!("{} {}",res.len(),decoded.len() * 2);
        assert_eq!(decoded,message);

    }

    #[test]
    fn test_arith_encoder(){
        let len = rand::thread_rng().gen_range(10000..20000);
        let symbols : i32 = rand::thread_rng().gen_range(5..11);
        let mut message : Vec<i32> = vec![0;len].into_iter().map(|x| rand::thread_rng().gen_range(0..symbols)).collect();
        message.push(100);
        let mut ae = ArithEncoder::new(message.clone(), 100);
        ae.encode();
        assert_eq!(ae.decode(),ae.message);
        ae.write_to_bin("TestingFiles/test_compression");
        ae.load_from_bin("TestingFiles/test_compression");
        assert_eq!(ae.decode(),message);


    }   

}