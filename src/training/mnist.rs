/// MNIST is an annotated dataset of handwritten digits.
/// http://yann.lecun.com/exdb/mnist/
///
/// This module trains the network using this dataset.

use mnist::{Mnist, MnistBuilder};
use crate::algorithm;
use crate::network;
use crate::ut;

const IMG_SIZE: usize = 28;  // Handwritten digits, 28x28
const N_CLASSES: usize = 10;  // 0-9

fn mnist_load(training_set_length: usize, test_set_length: usize) -> Mnist {
    use std::env::current_dir;

    let mut path_base = current_dir().unwrap();
    path_base.push("data");
    let path_base_str = path_base.as_path().to_str().unwrap();

    MnistBuilder::new().label_format_digit()
        .training_set_length(training_set_length.try_into().unwrap())
        .test_set_length(test_set_length.try_into().unwrap())
        .base_path(path_base_str)
        .finalize()
}

/// Waterprobing. An attempt to load and unpack MNIST dataset
#[cfg(test)]
mod test_mnist_load {
    use super::{Mnist, MnistBuilder};
    use std::{env::current_dir};
    const IMG_SIZE: usize = 28 * 28;
    const TRAINING_SET_LEN: usize = 100;
    const TEST_SET_LEN: usize = 10;

    #[test]
    fn build() {
        let mut path_base = current_dir().unwrap();
        path_base.push("data");
        let path_base_str = path_base.as_path().to_str().unwrap();
        println!("Current {}", path_base_str);
        let Mnist {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
            ..
        } = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(TRAINING_SET_LEN.try_into().unwrap())
            .test_set_length(TEST_SET_LEN.try_into().unwrap())
            .base_path(path_base_str)
            .finalize();
        assert!(trn_img.len() == IMG_SIZE * TRAINING_SET_LEN);
        assert!(trn_lbl.len() == TRAINING_SET_LEN);
        assert!(tst_img.len() == IMG_SIZE * TEST_SET_LEN);
        assert!(tst_lbl.len() == TEST_SET_LEN);
    }
}

fn train(net: &mut network::Network, mnist: &Mnist, ibegin_training_image: usize) {
}

fn test(net: &mut network::Network, mnist: &Mnist) {
}

pub fn main() {
    use std::env::args;

    const TRAINING_SET_LEN: usize = 2000;
    const TEST_SET_LEN: usize = 100;
    const NETWORK_GEOMETRY: [usize; 4] = [IMG_SIZE, 16, 8, 10];
    const NETWORK_FILE: &str = "network.bin";
    let args: Vec<String> = args().collect();
    let mut ibegin_img = 0i32;
    let mut network = {
        if args.len() == 1 {
            network::Network::from_geometry(&NETWORK_GEOMETRY.into())
        } else {
            ibegin_img = args[1].parse::<i32>().unwrap();
            ut::network_deserialize_from_file(NETWORK_FILE).unwrap()
        }
    };
    let mnist = mnist_load(TRAINING_SET_LEN, TEST_SET_LEN);

    if ibegin_img < 0 {
        test(&mut network, &mnist);
    } else {
        train(&mut network, &mnist, ibegin_img as usize);
    }
}
