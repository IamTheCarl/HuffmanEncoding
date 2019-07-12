extern crate byteorder;

pub mod encoding {
    use std::collections::HashMap;
    use std::collections::vec_deque::VecDeque;
    use std::fmt;
    use std::fmt::Formatter;
    use byteorder::{NetworkEndian, ReadBytesExt, WriteBytesExt};
    use std::io::Cursor;
    use std::io;

    trait BitReaderIMPL {
        fn get_vec(&self) -> &Vec<u8>;
        fn get_byte_index(&mut self) -> &mut usize;
        fn get_bit_index(&mut self) -> &mut usize;

        fn get_byte_index_imm(&self) -> usize;
        fn get_bit_index_imm(&self) -> usize;

        fn set_position_bits(self: &mut Self, position: usize) {
            assert_eq!(position <= 7, true); // Do not set your bits out of range.
            *self.get_bit_index() = position;
        }

        fn set_position_bytes(self: &mut Self, position: usize) {
            *self.get_byte_index() = position;
        }

        fn read_bit(self: &mut Self) -> bool {
            let mut byte_index = self.get_byte_index_imm();
            let mut bit_index = self.get_bit_index_imm();

            let val = (*self.get_vec())[byte_index];

            let result = val & (0x80 >> bit_index) != 0;

            bit_index += 1;
            if bit_index > 7 {
                bit_index = 0;
                byte_index += 1;
            }

            *self.get_byte_index() = byte_index;
            *self.get_bit_index() = bit_index;

            result
        }

        /// Indicates that we have reached the end of the vec, down to its very last bit.
        fn is_at_end(&self) -> bool {
            self.get_byte_index_imm() == self.get_vec().len() - 1 && self.get_bit_index_imm() == 8
        }
    }

    struct BitReader<'a> {
        vec: &'a Vec<u8>,
        byte_index: usize,
        bit_index: usize
    }

    impl<'a> BitReader<'a> {
        pub fn new(vec: &'a Vec<u8>) -> BitReader {
            let banger = BitReader {
                vec,
                byte_index: 0,
                bit_index: 0
            };

            banger
        }
    }

    impl<'a> BitReaderIMPL for BitReader<'a> {
        fn get_vec(&self) -> &Vec<u8> {
            &self.vec
        }

        fn get_byte_index(&mut self) -> &mut usize {
            &mut self.byte_index
        }

        fn get_bit_index(&mut self) -> &mut usize {
            &mut self.bit_index
        }

        fn get_byte_index_imm(&self) -> usize {
            self.byte_index
        }

        fn get_bit_index_imm(&self) -> usize {
            self.bit_index
        }
    }

    impl<'a> fmt::Display for BitReader<'a> {
        fn fmt(&self, f: &mut Formatter) -> fmt::Result {
            for b in self.get_vec().iter() {
                write!(f, "{:01$X} ", b, 2)?;
            }

            Ok(())
        }
    }

    struct BitWriter<'a> {
        vec: &'a mut Vec<u8>,
        byte_index: usize,
        bit_index: usize
    }

    impl<'a> BitWriter<'a> {
        pub fn new(vec: &'a mut Vec<u8>) -> BitWriter {
            let banger = BitWriter {
                vec,
                byte_index: 0,
                bit_index: 0
            };

            banger
        }

        pub fn write_bit(self: &mut Self, bit_set: bool) {
            let mut byte_index = self.byte_index;
            let mut bit_index = self.bit_index;

            let new_val = self.vec.get(byte_index);
            let mut new_val = match new_val {
                Some(byte) => *byte,
                None => {
                    // Data does not exist. Make it exist.
                    self.vec.insert(byte_index, 0);
                    0
                },
            };

            new_val |= match bit_set { true => 0x80, false => 0x00, } >> bit_index;
            self.vec[byte_index] = new_val;
            bit_index += 1;
            if bit_index > 7 {
                bit_index = 0;
                byte_index += 1;
            }

            self.byte_index = byte_index;
            self.bit_index = bit_index;
        }
    }

    impl<'a> BitReaderIMPL for BitWriter<'a> {
        fn get_vec(&self) -> &Vec<u8> {
            &self.vec
        }

        fn get_byte_index(&mut self) -> &mut usize {
            &mut self.byte_index
        }

        fn get_bit_index(&mut self) -> &mut usize {
            &mut self.bit_index
        }

        fn get_byte_index_imm(&self) -> usize {
            self.byte_index
        }

        fn get_bit_index_imm(&self) -> usize {
            self.bit_index
        }
    }

    impl<'a> fmt::Display for BitWriter<'a> {
        fn fmt(&self, f: &mut Formatter) -> fmt::Result {
            for b in self.get_vec().iter() {
                write!(f, "{:01$X} ", b, 2)?;
            }

            Ok(())
        }
    }

    pub trait Coder {
        fn encode(self: &Self, data: &Vec<u8>) -> Result<Vec<u8>, &'static str>;
        fn decode(self: &Self, data: &Vec<u8>) -> Result<Vec<u8>, &'static str>;
    }

    pub struct HuffmanCoder {

    }

    #[derive(Clone)]
    enum Node {
        Branch(Box<Node>, Box<Node>),
        Leaf(u8, u32),
        EOF
    }

    impl fmt::Display for Node {
        fn fmt(&self, f: &mut Formatter) -> fmt::Result {
            fn visualize_tree(root: &Node) -> String {
                match root {
                    Node::Branch(lesser_node, greater_node) => {
                        format!("({}, {})",
                                visualize_tree(lesser_node),
                                visualize_tree(greater_node)
                        )
                    },
                    Node::Leaf(val, count) => {
                        format!("[{}, {}]", val, count)
                    },
                    Node::EOF => String::from("EOF")
                }
            }

            write!(f, "{}", visualize_tree(self))
        }
    }

    impl HuffmanCoder {

        pub fn new() -> HuffmanCoder {
            HuffmanCoder{}
        }

        fn build_weighted_list(data: &Vec<u8>) -> HashMap<u8, u32> {
            let mut weighted_list: HashMap<u8, u32> = HashMap::new();

            for byte in data {
                let value = weighted_list.entry(byte.clone()).or_insert(0);
                *value += 1;
            }

            weighted_list
        }

        fn is_left_heavier(me: &(u8, u32), them: &(u8, u32)) -> bool {
            me.1 > them.1 || (me.1 == them.1 && me.0 > them.0)
        }

        fn get_weighted_list_top(weighted_list: &mut HashMap<u8, u32>) -> (u8, u32) {

            let mut max = (0, 0);

            for pair in weighted_list.iter() {

                let deref_pair = (*pair.0, *pair.1);

                // If we have a greater count, we are top.
                // If we have an equal count, go for the bigger key.
                if Self::is_left_heavier(&deref_pair, &max) {
                    max = deref_pair;
                }
            }

            // Remove the key since we just consumed it.
            weighted_list.remove(&max.0);

            max
        }

        fn get_node_count(node: &Node) -> u32 {
            match node {
                Node::Leaf(_val, count) => *count,
                Node::Branch(l, g) =>
                    HuffmanCoder::get_node_count(l.as_ref())
                        + HuffmanCoder::get_node_count(g.as_ref()),
                Node::EOF => 1
            }
        }

        fn build_decode_tree(weighted_list: &mut HashMap<u8, u32>) -> Node {

            let mut queue = VecDeque::new();

            // Build our list first.
            while !weighted_list.is_empty() {
                let (val, count) = Self::get_weighted_list_top(weighted_list);
                queue.push_front(Box::new(Node::Leaf(val, count)));
            }

            queue.push_front(Box::new(Node::EOF));

            // Now we turn it into a tree.
            // Once we're down to one node, we are finished.
            loop {

                /*print!("<");
                for e in &queue {
                    print!("{}, ", Self::visualize_tree(&e));
                }
                println!(">");*/

                if queue.len() <= 1 {
                    break;
                }

                // Get the two smallest nodes first, removing them from the queue.
                let smallest = queue.pop_front()
                    .expect("Element is missing from queue.");
                let second_smallest = queue.pop_front()
                    .expect("Element is missing from queue.");

                // Merge them together.
                let new_node =  Node::Branch(smallest, second_smallest);

                // Now we need to insert sort this.
                let our_count = Self::get_node_count(&new_node);

                let mut index = 0;
                for e in queue.iter() {
                    let their_count = Self::get_node_count(e.as_ref());

                    if their_count >= our_count {
                        break;
                    }

                    index += 1;
                }

                queue.insert(index, Box::new(new_node));
            }

            *queue.front().expect("Queue was empty.").clone()
        }

        fn build_encoder_map(tree: &Node, map: &mut HashMap<u8, Vec<bool>>) {

            fn get_byte_encoding_reversed(byte: u8, bit_vec: &mut Vec<bool>, tree: &Node) -> bool{
                match tree {
                    Node::EOF => false,
                    Node::Branch(lesser, greater) => {
                        let encoded = get_byte_encoding_reversed(byte, bit_vec, lesser);
                        if encoded {
                            bit_vec.push(false);
                            true
                        } else {
                            let encoded = get_byte_encoding_reversed(byte, bit_vec,
                                                                     greater);
                            if encoded {
                                bit_vec.push(true);
                                true
                            } else {
                                false
                            }
                        }
                    },
                    Node::Leaf(val, _count) => {
                        *val == byte
                    }
                }
            }

            fn list_all_syms(syms: &mut Vec<u8>, node: &Node) {
                match node {
                    Node::Branch(lesser, greater) => {
                        list_all_syms(syms, lesser);
                        list_all_syms(syms, greater);
                    },
                    Node::Leaf(val, _count) => {
                        syms.push(*val);
                    },
                    Node::EOF => {},
                }
            }

            let mut syms = Vec::new();

            list_all_syms(&mut syms, tree);

            for byte in syms.iter() {
                let mut vec = Vec::new();
                let found = get_byte_encoding_reversed(*byte, &mut vec, tree);
                if found {
                    vec.reverse();
                    map.entry(*byte).or_insert(vec);
                }
            }
        }

        fn get_eof_signature(tree: &Node, signature: &mut Vec<bool>) {
            fn get_byte_encoding_reversed(signature: &mut Vec<bool>, tree: &Node) -> bool {
                match tree {
                    Node::EOF => {
                        true
                    },
                    Node::Branch(lesser, greater) => {
                        let found = get_byte_encoding_reversed(signature, lesser);
                        if found {
                            signature.push(false);
                            true
                        } else {
                            let found = get_byte_encoding_reversed(signature, greater);
                            if found {
                                signature.push(true);
                                true
                            } else {
                                false
                            }
                        }
                    },
                    Node::Leaf(_val, _count) => {
                        // Not what we are interested in.
                        false
                    }
                }
            }

            get_byte_encoding_reversed(signature, tree);
            signature.reverse();
        }

        fn store_weighted_list(list: &HashMap<u8, u32>, data: &mut Vec<u8>) {
            for pair in list.iter() {

                let deref_pair = (*pair.0, *pair.1);

                data.write_u8(deref_pair.0).unwrap();
                data.write_u32::<NetworkEndian>(deref_pair.1).unwrap();
            }

            // Five zeros indicate the end of the list. The first zero doesn't have to be a zero.
            // It's the count of zero that actually indicates the end of the list.
            for _n in 0..5 {
                data.write_u8(0).unwrap();
            }
        }

        fn load_weighted_list(data: &mut Cursor<&Vec<u8>>) -> Result<HashMap<u8, u32>, io::Error> {
            let mut map = HashMap::new();

            loop {
                let key = data.read_u8()?;
                let count = data.read_u32::<NetworkEndian>()?;

                // A count of zero indicates an end of file.
                if count == 0 {
                    break;
                }

                map.entry(key).or_insert(count);
            }

            Ok(map)
        }

        /// Decodes a single byte and leaves the supplying BitBanger at the last point of read.
        /// The result is ether a decoded byte or None. In the case of None, this means you have
        /// reached the end of the data stream.
        ///
        /// Will return an error if the datastream runs out early.
        fn decode_byte(data: &mut BitReader, tree: &Node) -> Result<Option<u8>, &'static str> {

            fn find_leaf<'a>(data: &mut BitReader, node: &'a Node)
                -> Result<&'a Node, &'static str> {

                match node {
                    Node::Branch(lesser, greater) => {

                        if data.is_at_end() {
                            return Err("Unexpected end of data.");
                        }

                        let branch =
                            match data.read_bit() {
                                false => lesser,
                                true => greater
                            };

                        find_leaf(data, branch)
                    },
                    _ => {
                        Ok(node)
                    }
                }
            }

            //let start_byte = data.get_byte_index_imm();
            //
            // let start_bit = data.get_bit_index_imm();

            // Read one bit at a time until we find a code.
            let result = find_leaf(data, tree);

            match result {
                Ok(leaf) => {
                    match leaf {
                        Node::Leaf(val, _count) => {
                            Ok(Some(*val))
                        },
                        Node::EOF => {
                            Ok(None)
                        },
                        _ => {
                            panic!("Somehow got a branch. Shouldn't have been possible.")
                        }
                    }
                },
                Err(err) => {
                    /*data.set_position_bytes(start_byte);
                    data.set_position_bits(start_bit);

                    print!("READ: ");
                    while !data.is_at_end() {
                        print!("{}", match data.read_bit() { true => "1", false => "0" } );
                    }
                    println!();
                    print!("EOF: ");

                    let mut eof = Vec::new();
                    Self::get_eof_signature(tree, &mut eof);

                    for bit in eof {
                        print!("{}", match bit { true => "1", false => "0" } );
                    }
                    println!();*/

                    Err(err)
                }
            }
        }
    }

    impl Coder for HuffmanCoder {
        fn encode(self: &Self, data: &Vec<u8>) -> Result<Vec<u8>, &'static str> {

            let mut enc_bytes = Vec::new();

            let mut weighted_list = Self::build_weighted_list(data);
            Self::store_weighted_list(&weighted_list, &mut enc_bytes);

            let header_len = enc_bytes.len();

            let mut enc_bits = BitWriter::new(&mut enc_bytes);
            enc_bits.set_position_bytes(header_len); // Seek to byte after the header.
            enc_bits.set_position_bits(0); // First bit of that byte.

            let decode_tree = Self::build_decode_tree(&mut weighted_list);
            let mut encoder_map = HashMap::new();
            Self::build_encoder_map(&decode_tree, &mut encoder_map);

            for byte in data.iter() {
                let encoding = encoder_map.get(byte);

                match encoding {
                    Some(encoding) => {
                        for bit in encoding.iter() {
                            enc_bits.write_bit(*bit);
                        }
                    },
                    None => {
                        return Err("Attempted to encode byte that was not in tree.")
                    }
                }
            }

            let mut eof_signature = Vec::new();
            Self::get_eof_signature(&decode_tree, &mut eof_signature);

            for bit in eof_signature.iter() {
                enc_bits.write_bit(*bit);
            }

            Ok(enc_bytes)
        }

        fn decode(self: &Self, data: &Vec<u8>) -> Result<Vec<u8>, &'static str> {
            let mut data_cursor = Cursor::new(data.as_ref());

            let mut weighted_list =
                Self::load_weighted_list(&mut data_cursor).unwrap();

            let tree = Self::build_decode_tree(&mut weighted_list);

            let mut banger = BitReader::new(data);
            banger.set_position_bytes(data_cursor.position() as usize);

            let mut decoded = Vec::new();

            loop {
                let byte = Self::decode_byte(&mut banger, &tree)?;
                match byte {
                    Some(byte) => {
                        decoded.push(byte)
                    },
                    None => break // End of file.
                };
            }

            Ok(decoded)
        }
    }

    #[cfg(test)]
    mod tests {

        use super::*;
        use std::fs::File;
        use std::io::BufReader;

        #[cfg(test)]
        mod weighted_lists {
            use super::*;

            #[test]
            fn build_weighted_list() {
                let data: Vec<u8> = vec![5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 2, 2, 1, 1];

                let map = HuffmanCoder::build_weighted_list(&data);

                assert_eq!(*map.get(&5).expect("Could not find value 5"), 5,
                           "Did not count 5 correctly.");

                assert_eq!(*map.get(&4).expect("Could not find value 5"), 4,
                           "Did not count 4 correctly.");

                assert_eq!(*map.get(&3).expect("Could not find value 3"), 2,
                           "Did not count 3 correctly.");

                assert_eq!(*map.get(&2).expect("Could not find value 2"), 2,
                           "Did not count 2 correctly.");

                assert_eq!(*map.get(&1).expect("Could not find value 1"), 2,
                           "Did not count 1 correctly.");
            }

            #[test]
            fn get_weighted_list_top() {

                // First we need to build a weighted list.
                let data: Vec<u8> = vec![5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 3, 3, 2, 2, 1, 1];
                let mut list = HuffmanCoder::build_weighted_list(&data);

                // Now we can actually test.
                assert_eq!(HuffmanCoder::get_weighted_list_top(&mut list).0, 4);
                assert_eq!(HuffmanCoder::get_weighted_list_top(&mut list).0, 5);
                assert_eq!(HuffmanCoder::get_weighted_list_top(&mut list).0, 3);
                assert_eq!(HuffmanCoder::get_weighted_list_top(&mut list).0, 2);
                assert_eq!(HuffmanCoder::get_weighted_list_top(&mut list).0, 1);
                assert_eq!(HuffmanCoder::get_weighted_list_top(&mut list).1, 0);
            }

            #[test]
            fn find_eof_singature_in_tree() {
                let tree = Node::Branch(
                    Box::new(Node::Branch(
                        Box::new(Node::Branch(
                            Box::new(Node::EOF),
                            Box::new(Node::Leaf(1, 1))
                        )),
                        Box::new(Node::Leaf(2, 1))
                    )),
                    Box::new(Node::Leaf(3, 2))
                );

                let mut signature = Vec::new();
                HuffmanCoder::get_eof_signature(&tree, &mut signature);

                assert_eq!(signature.len(), 3);
                assert_eq!(signature[0], false);
                assert_eq!(signature[1], false);
                assert_eq!(signature[2], false);
            }

            /// This function is used to check the correctness of a stored weighted list. A weighted
            /// list is stored as a HashMap when loaded in memory, which means the order we iterate it
            /// in is not consistent in any way. Now this is fine for our usage, but it does mean that
            /// our check for correctness in the byte list needs to keep that in mind.
            fn check_weighted_list_index(data: &Vec<u8>, correct: &mut HashMap<u8, Vec<u8>>, index: usize) {
                // Check if it was stored correctly.

                let key = data[index];
                let array = &correct[&key];

                let expect = &[key, array[0], array[1], array[2], array[3]];

                let equal = data[index..index+5].iter().zip(expect)
                    .all(|(a,b)| *a == *b);

                /*print!("READ: ");
                for b in data[index..index+5].iter() {
                    print!("{:X}", b);
                }
                println!();

                print!("EXPECT: ");
                for b in expect.iter() {
                    print!("{:X}", b);
                }
                println!();*/

                assert_eq!(equal, true, "Data was not stored correctly.");
            }

            #[test]
            fn store_weighted_list_one_element() {
                let mut list: HashMap<u8, u32> = HashMap::new();
                list.entry(0x55).or_insert(0xA3A33A3A);
                let mut data = Vec::new();

                let correct: Vec<u8>  = vec![
                    0x55, 0xA3, 0xA3, 0x3A, 0x3A,
                    0x00, 0x00, 0x00, 0x00, 0x00
                ];

                HuffmanCoder::store_weighted_list(&list, &mut data);

                // Check if it was stored correctly. We don't need the fancy checks because its only
                // one element, so we don't need to worry about order.
                let equal = (data.len() == correct.len()) &&  // zip stops at the shortest
                    data.iter().zip(correct).all(|(a,b)| *a == b);

                assert_eq!(equal, true, "Data was not stored correctly.");
            }

            #[test]
            fn store_weighted_list_two_element() {
                let mut list: HashMap<u8, u32> = HashMap::new();
                list.entry(0x55).or_insert(0xA3A33A3A);
                list.entry(0xAA).or_insert(0x53533535);
                let mut data = Vec::new();

                let mut correct: HashMap<u8, Vec<u8>> = HashMap::new();

                correct.entry(0xAA).or_insert(vec![0x53, 0x53, 0x35, 0x35]);
                correct.entry(0x55).or_insert(vec![0xA3, 0xA3, 0x3A, 0x3A]);

                HuffmanCoder::store_weighted_list(&list, &mut data);

                // Check if they are stored correctly.
                check_weighted_list_index(&data, &mut correct, 0);
                check_weighted_list_index(&data, &mut correct, 5);
            }

            #[test]
            fn store_weighted_list_fore_element() {
                let mut list: HashMap<u8, u32> = HashMap::new();
                list.entry(0x55).or_insert(0xA3A33A3A);
                list.entry(0xAA).or_insert(0x53533535);
                list.entry(0x01).or_insert(0x10101010);
                list.entry(0x02).or_insert(0x20202020);
                let mut data = Vec::new();

                let mut correct: HashMap<u8, Vec<u8>> = HashMap::new();

                correct.entry(0xAA).or_insert(vec![0x53, 0x53, 0x35, 0x35]);
                correct.entry(0x01).or_insert(vec![0x10, 0x10, 0x10, 0x10]);
                correct.entry(0x02).or_insert(vec![0x20, 0x20, 0x20, 0x20]);
                correct.entry(0x55).or_insert(vec![0xA3, 0xA3, 0x3A, 0x3A]);

                HuffmanCoder::store_weighted_list(&list, &mut data);

                // Check if they are stored correctly.
                check_weighted_list_index(&data, &mut correct, 0);
                check_weighted_list_index(&data, &mut correct, 5);
                check_weighted_list_index(&data, &mut correct, 10);
                check_weighted_list_index(&data, &mut correct, 15);
            }

            fn assert_maps_eq(map_a: HashMap<u8, u32>, map_b: HashMap<u8, u32>) {
                // Are they of at least equal length?
                if map_a.len() == map_b.len() {
                    // Guess we gotta get into the details then.

                    for (key_a, value_a) in map_a {

                        println!("{}", key_a);

                        let optional = map_b.get(&key_a);

                        match optional {
                            Some(value_b) => {
                                if value_a != *value_b {
                                    panic!("Maps have non-matching value.");
                                }
                            },
                            None => {
                                panic!("Maps have non-matching key: {}.", key_a);
                            }
                        }
                    }
                } else {
                    // Not equal length? No way they're equal then.
                    panic!("Maps are not of equal length.");
                }
            }

            #[test]
            fn load_one_element_weigthed_list_from_storage() {

                let mut list: HashMap<u8, u32> = HashMap::new();
                list.entry(0x55).or_insert(0xA3A33A3A);

                let data: Vec<u8>  = vec![
                    0x55, 0xA3, 0xA3, 0x3A, 0x3A,
                    0x00, 0x00, 0x00, 0x00, 0x00
                ];
                let mut data_cursor = Cursor::new(&data);

                let weighted_list =
                    HuffmanCoder::load_weighted_list(&mut data_cursor)
                        .unwrap();

                assert_maps_eq(list, weighted_list);
            }

            #[test]
            fn load_two_element_weigthed_list_from_storage() {
                let mut list: HashMap<u8, u32> = HashMap::new();
                list.entry(0x55).or_insert(0xA3A33A3A);
                list.entry(0xAA).or_insert(0x53533535);

                let data: Vec<u8>  = vec![
                    0xAA, 0x53, 0x53, 0x35, 0x35,
                    0x55, 0xA3, 0xA3, 0x3A, 0x3A,
                    0x00, 0x00, 0x00, 0x00, 0x00
                ];
                let mut data_cursor = Cursor::new(&data);

                let weighted_list =
                    HuffmanCoder::load_weighted_list(&mut data_cursor)
                        .unwrap();

                assert_maps_eq(list, weighted_list);
            }

            #[test]
            fn load_fore_element_weigthed_list_from_storage() {
                let mut list: HashMap<u8, u32> = HashMap::new();
                list.entry(0x55).or_insert(0xA3A33A3A);
                list.entry(0xAA).or_insert(0x53533535);
                list.entry(0x01).or_insert(0x10101010);
                list.entry(0x02).or_insert(0x20202020);

                let data: Vec<u8>  = vec![
                    0xAA, 0x53, 0x53, 0x35, 0x35,
                    0x01, 0x10, 0x10, 0x10, 0x10,
                    0x02, 0x20, 0x20, 0x20, 0x20,
                    0x55, 0xA3, 0xA3, 0x3A, 0x3A,
                    0x00, 0x00, 0x00, 0x00, 0x00
                ];
                let mut data_cursor = Cursor::new(&data);

                let weighted_list =
                    HuffmanCoder::load_weighted_list(&mut data_cursor)
                        .unwrap();

                assert_maps_eq(list, weighted_list);
            }

            #[test]
            fn build_weighted_list_of_all_symbols_store_then_load() {
                // I'm trying really hard to break this.

                let mut data = Vec::new();
                for value in 0..255 as u8 {
                    data.push(value);
                }

                let list = HuffmanCoder::build_weighted_list(&data);
                let mut stored = Vec::new();
                HuffmanCoder::store_weighted_list(&list, &mut stored);

                let mut data_cursor = Cursor::new(&stored);
                data_cursor.set_position(0);
                let loaded =
                    HuffmanCoder::load_weighted_list(&mut data_cursor).unwrap();

                assert_maps_eq(list, loaded);
            }
        }

        #[cfg(test)]
        mod trees {
            use super::*;

            fn get_branch(node: &Node, expected_weight: u32) -> (&Box<Node>, &Box<Node>) {
                match node {
                    Node::Branch(lesser_node, greater_node) =>
                        {
                            let weight = HuffmanCoder::get_node_count(node);
                            assert_eq!(weight, expected_weight,
                                       "Branch should have had weight of {}. Was {}",
                                       expected_weight, weight);
                            (lesser_node, greater_node)
                        },
                    Node::Leaf(_val, _count) =>
                        panic!("Didn't get a branch when expected."),
                    Node::EOF =>
                        panic!("Didn't get a branch when expected."),
                }
            }

            fn get_leaf(node: &Node, expected_value: u8, expected_count: u32) -> (&u8, &u32) {
                match node {
                    Node::Branch(_lesser_node, _greater_node) =>
                        panic!("Didn't get a leaf when expected."),
                    Node::EOF =>
                        panic!("Didn't get a leaf when expected."),
                    Node::Leaf(val, count) =>
                        {
                            assert_eq!(*val, expected_value,
                                       "Did not get expected value for leaf. Expected: {} Got: {}",
                                       expected_value, val);

                            assert_eq!(*count, expected_count,
                                       "Did not get expected count for leaf. Expected: {} Got: {}",
                                       expected_count, count);

                            (val, count)
                        },
                }
            }

            fn get_eof(node: &Node) {
                match node {
                    Node::Branch(_lesser_node, _greater_node) =>
                        panic!("Didn't get EOF when expected."),
                    Node::Leaf(_val, _count) =>
                        panic!("Didn't get EOF when expected."),
                    Node::EOF => {},
                }
            }

            #[test]
            fn node_count() {

                let node = Node::Branch(
                    Box::new(Node::Branch(
                        Box::new(Node::Leaf(1, 1)),
                        Box::new(Node::Leaf(2, 2))
                    )),
                    Box::new(Node::Leaf(3, 3))
                );

                assert_eq!(HuffmanCoder::get_node_count(&node), 6);
            }

            #[test]
            fn build_empty_tree() {
                let data: Vec<u8> = vec![];
                let mut list = HuffmanCoder::build_weighted_list(&data);
                let tree = HuffmanCoder::build_decode_tree(&mut list);

                //println!("{}", tree);

                get_eof(&tree);
            }

            #[test]
            fn build_single_leaf_tree() {
                let data: Vec<u8> = vec![1];
                let mut list = HuffmanCoder::build_weighted_list(&data);
                let tree = HuffmanCoder::build_decode_tree(&mut list);

                //println!("{}", tree);

                let branch = get_branch(&tree, 2);

                get_eof(branch.0);
                get_leaf(branch.1, 1, 1);
            }

            #[test]
            fn build_two_leaf_tree() {
                let data: Vec<u8> = vec![1, 2];

                let mut list = HuffmanCoder::build_weighted_list(&data);
                let tree = HuffmanCoder::build_decode_tree(&mut list);

                /*
                Expected tree.

                    12
                   /  \
                  2   E1
                     / \
                    E  1
                */

                //println!("{}", tree);

                let root = get_branch(&tree, 3);
                get_leaf(&root.0, 2, 1);

                let branch = get_branch(&root.1, 2);
                get_eof(branch.0);
                get_leaf(branch.1, 1, 1);
            }

            #[test]
            fn build_two_leaf_reversed_tree() {
                let data: Vec<u8> = vec![2, 1];

                let mut list = HuffmanCoder::build_weighted_list(&data);
                let tree = HuffmanCoder::build_decode_tree(&mut list);

                /*
                Expected tree.

                    12
                   /  \
                  2   E1
                     / \
                    E  1
                */

                //println!("{}", tree);

                let root = get_branch(&tree, 3);

                get_leaf(&root.0, 2, 1);
                let branch = get_branch(&root.1, 2);

                get_eof(branch.0);
                get_leaf(branch.1, 1, 1);
            }

            #[test]
            fn build_two_leaf_biased_tree() {
                let data: Vec<u8> = vec![2, 1, 1, 1];

                let mut list = HuffmanCoder::build_weighted_list(&data);
                let tree = HuffmanCoder::build_decode_tree(&mut list);

                /*
                Expected tree

                     E12
                    /  \
                   E2  1
                  /  \
                 E   2
                */

                //println!("{}", tree);

                let root = get_branch(&tree, 5);

                let branch = get_branch(&root.0, 2);
                get_leaf(&root.1, 1, 3);

                get_eof(branch.0);
                get_leaf(branch.1, 2, 1);
            }

            #[test]
            fn build_leaf_branch_tree() {
                let data: Vec<u8> = vec![5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3];
                let mut list = HuffmanCoder::build_weighted_list(&data);
                let tree = HuffmanCoder::build_decode_tree(&mut list);

                /*
                Expected tree.
                       E345
                      /    \
                     5    E34
                         /  \
                        E3  4
                       /  \
                      E   3
                */
                //println!("{}", tree);

                let root = get_branch(&tree, 12);
                get_leaf(root.0, 5, 5);
                let lesser_branch = get_branch(root.1, 7);

                let lesser_lesser_branch =
                    get_branch(lesser_branch.0, 3);
                get_leaf(lesser_branch.1, 4, 4);

                get_eof(lesser_lesser_branch.0);
                get_leaf(lesser_lesser_branch.1, 3, 2);

            }

            #[test]
            fn build_full_tree() {
                let data: Vec<u8> = vec![5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 2, 2, 1, 1];

                let mut list = HuffmanCoder::build_weighted_list(&data);
                let tree = HuffmanCoder::build_decode_tree(&mut list);

                /*
                The tree structure produced should look like this.

                          E12345
                         /     \
                       E123    45
                       /   \   / \
                      E1   23 4  5
                     / \  / \
                    E  1 2  3
                */
                //println!("{}", tree);

                let root = get_branch(&tree, 16);

                let root_lesser = get_branch(root.0, 7);
                let root_greater = get_branch(root.1, 9);

                let root_lesser_lesser =
                    get_branch(root_lesser.0, 3);
                let root_lesser_greater =
                    get_branch(root_lesser.1, 4);

                get_eof(root_lesser_lesser.0);
                get_leaf(root_lesser_lesser.1, 1, 2);

                get_leaf(root_lesser_greater.0, 2, 2);
                get_leaf(root_lesser_greater.1, 3, 2);

                get_leaf(root_greater.0, 4, 4);
                get_leaf(root_greater.1, 5, 5);
            }
        }

        #[cfg(test)]
        mod bit_banging {
            use super::*;

            #[test]
            fn bit_write_one_bit() {
                let mut vec: Vec<u8> = Vec::new();
                let mut bang = BitWriter::new(&mut vec);

                bang.write_bit(true);
                assert_eq!(vec[0], 0x80);
            }

            #[test]
            fn bit_write_one_byte() {
                let mut vec: Vec<u8> = Vec::new();
                let mut bang = BitWriter::new(&mut vec);

                bang.write_bit(true);
                bang.write_bit(false);
                bang.write_bit(true);
                bang.write_bit(false);

                bang.write_bit(true);
                bang.write_bit(false);
                bang.write_bit(true);
                bang.write_bit(false);

                assert_eq!(vec[0], 0xAA);
            }

            #[test]
            fn bit_write_two_byte() {
                let mut vec: Vec<u8> = Vec::new();
                let mut bang = BitWriter::new(&mut vec);

                bang.write_bit(true);
                bang.write_bit(false);
                bang.write_bit(true);
                bang.write_bit(false);

                bang.write_bit(true);
                bang.write_bit(false);
                bang.write_bit(true);
                bang.write_bit(false);

                bang.write_bit(true);
                bang.write_bit(false);
                bang.write_bit(true);
                bang.write_bit(false);

                bang.write_bit(true);
                bang.write_bit(false);
                bang.write_bit(true);
                bang.write_bit(false);

                assert_eq!(vec[0], 0xAA);
                assert_eq!(vec[1], 0xAA);
            }

            #[test]
            fn bit_read_one_bit() {
                let mut vec = vec![0x80];
                let mut bang = BitReader::new(&mut vec);

                assert_eq!(bang.read_bit(), true);
            }

            #[test]
            fn bit_read_one_byte() {
                let mut vec = vec![0xAA];
                let mut bang = BitReader::new(&mut vec);

                assert_eq!(bang.read_bit(), true);
                assert_eq!(bang.read_bit(), false);
                assert_eq!(bang.read_bit(), true);
                assert_eq!(bang.read_bit(), false);

                assert_eq!(bang.read_bit(), true);
                assert_eq!(bang.read_bit(), false);
                assert_eq!(bang.read_bit(), true);
                assert_eq!(bang.read_bit(), false);
            }

            #[test]
            fn bit_read_two_byte() {
                let mut vec = vec![0xAA, 0xAA];
                let mut bang = BitReader::new(&mut vec);

                assert_eq!(bang.read_bit(), true);
                assert_eq!(bang.read_bit(), false);
                assert_eq!(bang.read_bit(), true);
                assert_eq!(bang.read_bit(), false);

                assert_eq!(bang.read_bit(), true);
                assert_eq!(bang.read_bit(), false);
                assert_eq!(bang.read_bit(), true);
                assert_eq!(bang.read_bit(), false);

                assert_eq!(bang.read_bit(), true);
                assert_eq!(bang.read_bit(), false);
                assert_eq!(bang.read_bit(), true);
                assert_eq!(bang.read_bit(), false);

                assert_eq!(bang.read_bit(), true);
                assert_eq!(bang.read_bit(), false);
                assert_eq!(bang.read_bit(), true);
                assert_eq!(bang.read_bit(), false);
            }
        }

        #[cfg(test)]
        mod encoding {
            use super::*;

            #[test]
            fn build_encoder_map() {
                let mut encoder_map = HashMap::new();

                let tree = Node::Branch(
                    Box::new(Node::Branch(
                        Box::new(Node::Leaf(1, 1)),
                        Box::new(Node::Leaf(2, 2))
                    )),
                    Box::new(Node::Leaf(3, 3))
                );

                HuffmanCoder::build_encoder_map(&tree, &mut encoder_map);

                assert_eq!(encoder_map.len(), 3);

                let gen = encoder_map.get(&1).unwrap();
                let correct = vec![false, false];
                let equal = (gen.len() == correct.len()) &&  // zip stops at the shortest
                    gen.iter().zip(correct).all(|(a,b)| *a == b);
                assert_eq!(equal, true);

                let gen = encoder_map.get(&2).unwrap();
                let correct = vec![false, true];
                let equal = (gen.len() == correct.len()) &&  // zip stops at the shortest
                    gen.iter().zip(correct).all(|(a,b)| *a == b);
                assert_eq!(equal, true);

                let gen = encoder_map.get(&3).unwrap();
                let correct = vec![true];
                let equal = (gen.len() == correct.len()) &&  // zip stops at the shortest
                    gen.iter().zip(correct).all(|(a,b)| *a == b);
                assert_eq!(equal, true);
            }

            #[test]
            fn byte_encode_single_leaf() {
                let tree = Node::Branch(
                    Box::new(Node::EOF),
                    Box::new(Node::Leaf(1, 1))
                );

                let mut bit_vec = HashMap::new();
                HuffmanCoder::build_encoder_map(&tree, &mut bit_vec);

                let byte = bit_vec.get(&1).unwrap();
                assert_eq!(byte.len(), 1);
                assert_eq!(byte[0], true);
            }

            #[test]
            fn byte_encode_two_leaf() {
                let tree = Node::Branch(
                    Box::new(Node::Branch(
                        Box::new(Node::EOF),
                        Box::new(Node::Leaf(1, 1))
                    )),
                    Box::new(Node::Leaf(2, 1))
                );

                let mut bit_vec = HashMap::new();
                HuffmanCoder::build_encoder_map(&tree, &mut bit_vec);

                let byte = bit_vec.get(&1).unwrap();
                assert_eq!(byte.len(), 2);
                assert_eq!(byte[0], false);
                assert_eq!(byte[1], true);

                let byte = bit_vec.get(&2).unwrap();
                assert_eq!(byte.len(), 1);
                assert_eq!(byte[0], true);
            }

            #[test]
            fn byte_encode_three_leaf() {
                let tree = Node::Branch(
                    Box::new(Node::Branch(
                        Box::new(Node::Branch(
                            Box::new(Node::EOF),
                            Box::new(Node::Leaf(1, 1))
                        )),
                        Box::new(Node::Leaf(2, 1))
                    )),
                    Box::new(Node::Leaf(3, 2))
                );

                let mut bit_vec = HashMap::new();
                HuffmanCoder::build_encoder_map(&tree, &mut bit_vec);

                let byte = bit_vec.get(&1).unwrap();
                assert_eq!(byte.len(), 3);
                assert_eq!(byte[0], false);
                assert_eq!(byte[1], false);
                assert_eq!(byte[2], true);

                let byte = bit_vec.get(&2).unwrap();
                assert_eq!(byte.len(), 2);
                assert_eq!(byte[0], false);
                assert_eq!(byte[1], true);

                let byte = bit_vec.get(&3).unwrap();
                assert_eq!(byte.len(), 1);
                assert_eq!(byte[0], true);
            }

            #[test]
            fn encode_single() {
                let data: Vec<u8> = vec![5];

                let encoder = HuffmanCoder::new();

                let encoded = encoder.encode(&data).expect("Failed to encode.");

                assert_eq!(encoded.len(), 11); // Length of data.

                assert_eq!(encoded[0], 0x05); // First symbol is a 5.
                assert_eq!(encoded[1], 0x00); // There is one instance of it.
                assert_eq!(encoded[2], 0x00);
                assert_eq!(encoded[3], 0x00);
                assert_eq!(encoded[4], 0x01);

                assert_eq!(encoded[5], 0x00); // Second symbol is the end of header marker.
                assert_eq!(encoded[6], 0x00);
                assert_eq!(encoded[7], 0x00);
                assert_eq!(encoded[8], 0x00);
                assert_eq!(encoded[9], 0x00);

                assert_eq!(encoded[10], 0x80); // Actually encoded data. Ends with EOF.
            }

            #[test]
            fn encode_several_of_single() {
                let data: Vec<u8> = vec![5, 5, 5, 5, 5, 5, 5];

                let encoder = HuffmanCoder::new();

                let encoded = encoder.encode(&data).expect("Failed to encode.");

                assert_eq!(encoded.len(), 11); // Length of data.

                assert_eq!(encoded[0], 0x05); // First symbol is a 5.
                assert_eq!(encoded[1], 0x00); // There are 7 instances of it.
                assert_eq!(encoded[2], 0x00);
                assert_eq!(encoded[3], 0x00);
                assert_eq!(encoded[4], 0x07);

                assert_eq!(encoded[5], 0x00); // Mark end of header.
                assert_eq!(encoded[6], 0x00);
                assert_eq!(encoded[7], 0x00);
                assert_eq!(encoded[8], 0x00);
                assert_eq!(encoded[9], 0x00);

                assert_eq!(encoded[10], 0xFE); // Actually encoded data. Ends with EOF.
            }

            #[test]
            fn encode_several_of_two() {
                let data: Vec<u8> = vec![5, 5, 5, 5, 5, 4, 4, 4, 4];

                let encoder = HuffmanCoder::new();

                let encoded = encoder.encode(&data).expect("Failed to encode.");

                assert_eq!(encoded.len(), 17); // Length of data.

                let first = encoded[0];

                match first {
                    0x05 => {
                        assert_eq!(encoded[1], 0x00); // There are 5 instances of the 5.
                        assert_eq!(encoded[2], 0x00);
                        assert_eq!(encoded[3], 0x00);
                        assert_eq!(encoded[4], 0x05);
                    },
                    0x04 => {
                        assert_eq!(encoded[1], 0x00); // There are 4 instances of the 4.
                        assert_eq!(encoded[2], 0x00);
                        assert_eq!(encoded[3], 0x00);
                        assert_eq!(encoded[4], 0x04);
                    },
                    _ => panic!("Unexpected symbol for first.")
                }

                let second = encoded[5];

                // First and second symbol must be for different things, being 4 or 5.
                assert_ne!(second, first);

                match second {
                    0x05 => {
                        assert_eq!(encoded[6], 0x00); // There are 5 instances of the 5.
                        assert_eq!(encoded[7], 0x00);
                        assert_eq!(encoded[8], 0x00);
                        assert_eq!(encoded[9], 0x05);
                    },
                    0x04 => {
                        assert_eq!(encoded[6], 0x00); // There are 4 instances of the 4.
                        assert_eq!(encoded[7], 0x00);
                        assert_eq!(encoded[8], 0x00);
                        assert_eq!(encoded[9], 0x04);
                    },
                    _ => panic!("Unexpected symbol for first.")
                }

                assert_eq!(encoded[10], 0x00); // Mark end of header.
                assert_eq!(encoded[11], 0x00);
                assert_eq!(encoded[12], 0x00);
                assert_eq!(encoded[13], 0x00);
                assert_eq!(encoded[14], 0x00);

                assert_eq!(encoded[15], 0xFA); // Actually encoded data. Ends with EOF.
                assert_eq!(encoded[16], 0xA8);
            }
        }

        #[cfg(test)]
        mod decoding {
            use super::*;

            #[test]
            fn decode_single_byte_spoon_fed() {
                // Only thing we're testing here is our ability to decode a single byte.

                let tree = Node::Branch(
                    Box::new(Node::Branch(
                        Box::new(Node::Leaf(2, 3)),
                        Box::new(Node::EOF)
                    )),
                    Box::new(Node::Leaf(1, 2))
                );

                let data = vec![0x80];
                let mut banger = BitReader::new(&data);
                let byte = HuffmanCoder::decode_byte(&mut banger, &tree)
                    .unwrap().unwrap();

                assert_eq!(byte, 1);
            }

            #[test]
            fn decode_single_eof_spoon_fed() {
                // Only thing we're testing here is our ability to decode a single byte.

                let tree = Node::Branch(
                    Box::new(Node::Branch(
                        Box::new(Node::Leaf(2, 3)),
                        Box::new(Node::EOF)
                    )),
                    Box::new(Node::Leaf(1, 2))
                );

                let data = vec![0x40];
                let mut banger = BitReader::new(&data);
                let byte = HuffmanCoder::decode_byte(&mut banger, &tree)
                    .unwrap();

                match byte {
                    Some(byte) => panic!("Got a byte when expecting EOF. Byte: {}", byte),
                    None => {}, // That's a pass.
                }
            }

            #[test]
            fn decode_several_of_two() {
                let data: Vec<u8> = vec![
                    0x05, 0x00, 0x00, 0x00, 0x05, // 5 instances of 5.
                    0x04, 0x00, 0x00, 0x00, 0x04, // 4 instances of 4.
                    0x00, 0x00, 0x00, 0x00, 0x00, // End of list.
                    0xFA, 0xA8 // Encoded data.
                ];

                let encoder = HuffmanCoder::new();
                let decoded = encoder.decode(&data).unwrap();

                let correct_decode: Vec<u8> = vec![5, 5, 5, 5, 5, 4, 4, 4, 4];

                let equal = (correct_decode.len() == decoded.len()) &&
                    decoded.iter().zip(correct_decode).all(|(a,b)| *a == b);

                assert_eq!(equal, true);
            }

            #[test]
            fn decode_auto_encoded() {
                let data: Vec<u8> = vec![5, 5, 5, 5, 5, 4, 4, 4, 4];

                let encoder = HuffmanCoder::new();

                let encoded = encoder.encode(&data).unwrap();
                let decoded = encoder.decode(&encoded).unwrap();

                let equal = (data.len() == decoded.len()) &&
                    decoded.iter().zip(data).all(|(a,b)| *a == b);

                assert_eq!(equal, true);
            }
        }

        #[cfg(test)]
        mod big_tests {
            use super::*;
            use std::io::Read;

            #[test]
            fn big_weighted_list_build() {

                // This is more of a benchmark. Just needs to return in a reasonable amount of time.

                let file =  File::open("./firstNephi")
                    .expect("Failed to find the book of Nephi.");
                let mut reader = BufReader::new(file);

                let mut data: Vec<u8> = Vec::new();
                reader.read_to_end(&mut data).unwrap();

                let _list = HuffmanCoder::build_weighted_list(&data);
            }

            #[test]
            fn big_decode_tree_build() {

                // This is more of a benchmark. Just needs to return in a reasonable amount of time.

                let file =  File::open("./firstNephi")
                    .expect("Failed to find the book of Nephi.");
                let mut reader = BufReader::new(file);

                let mut data: Vec<u8> = Vec::new();
                reader.read_to_end(&mut data).unwrap();

                let mut list = HuffmanCoder::build_weighted_list(&data);
                let _tree = HuffmanCoder::build_decode_tree(&mut list);
            }

            #[test]
            fn big_encode_map_build() {

                // This is more of a benchmark. Just needs to return in a reasonable amount of time.

                let file =  File::open("./firstNephi")
                    .expect("Failed to find the book of Nephi.");
                let mut reader = BufReader::new(file);

                let mut data: Vec<u8> = Vec::new();
                reader.read_to_end(&mut data).unwrap();

                let mut list = HuffmanCoder::build_weighted_list(&data);
                let decode_tree = HuffmanCoder::build_decode_tree(&mut list);
                let mut encoder_map = HashMap::new();
                HuffmanCoder::build_encoder_map(&decode_tree, &mut encoder_map);
            }

            #[test]
            fn big_encode_full_test() {

                // We aren't testing if the encoding is implemented perfectly.
                // We are only testing if the data coming out is smaller than the data going in.
                // This is also a nice way to benchmark when all the features are working together.

                let file =  File::open("./firstNephi")
                    .expect("Failed to find the book of Nephi.");
                let mut reader = BufReader::new(file);

                let mut data: Vec<u8> = Vec::new();
                reader.read_to_end(&mut data).unwrap();

                let encoder = HuffmanCoder::new();

                let encoded = encoder.encode(&data).expect("Failed to encode.");

                assert_eq!(encoded.len() < data.len(), true, "Data was not compressed.");
                println!("Compressed from {} bytes to {} bytes. That's a {}% compression.", data.len(),
                         encoded.len(), (((encoded.len() as f32) / (data.len() as f32)) * 100.0) as f32);
            }

            #[test]
            fn fing_eof_signature() {
                // See if we can find the EOF for a big file.

                let file =  File::open("./firstNephi")
                    .expect("Failed to find the book of Nephi.");
                let mut reader = BufReader::new(file);

                let mut data: Vec<u8> = Vec::new();
                reader.read_to_end(&mut data).unwrap();

                let mut list = HuffmanCoder::build_weighted_list(&data);
                let tree = HuffmanCoder::build_decode_tree(&mut list);

                let mut eof = Vec::new();
                HuffmanCoder::get_eof_signature(&tree, &mut eof);

                assert_eq!(eof.len() > 0, true);
            }

            #[test]
            fn big_decode_all() {
                // Test everything all together now! Load a file, build a tree, encode the file, then
                // decode the file and see if you got everything back.

                let file =  File::open("./firstNephi")
                    .expect("Failed to find the book of Nephi.");
                let mut reader = BufReader::new(file);

                let mut data: Vec<u8> = Vec::new();
                reader.read_to_end(&mut data).unwrap();

                let encoder = HuffmanCoder::new();

                let encoded = encoder.encode(&data).unwrap();

                let decoded = encoder.decode(&encoded).unwrap();

                // Make sure we got our data back.
                let equal = (decoded.len() == data.len()) &&  // zip stops at the shortest
                    decoded.iter().zip(data).all(|(a,b)| *a == b);

                assert_eq!(equal, true, "Decompressed data does not match source data.");
            }
        }
    }
}

fn main() {
    println!("Hello, world!");
}
