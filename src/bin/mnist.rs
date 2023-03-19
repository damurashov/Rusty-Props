/// MNIST is an annotated dataset of handwritten digits.
/// http://yann.lecun.com/exdb/mnist/
///
/// This module trains the network using this dataset.

use mnist::{Mnist, MnistBuilder};
use rusty_props::algorithm;
use rusty_props::network;
use rusty_props::ut;

const IMG_SIZE_BYTES: usize = 28 * 28;  // Handwritten digits, 28x28
const OUTPUT_NEURONS_NUMBER: usize = 10;
const NETWORK_GEOMETRY: [usize; 4] = [IMG_SIZE_BYTES, 16, 8, OUTPUT_NEURONS_NUMBER];
const MNIST_OUTPUT_LAYER_SIZE: usize = 10;  // Mnist is a handwritten digits annotated database, 10 digits

/// Encapsulates traininig state, so it can be resumed later
struct MnistTrainingState<'a> {
    dataset: &'a mnist::Mnist,
    base_offset: usize,
}

impl MnistTrainingState<'_> {
    fn offset_as_relative(&self, offset: usize) -> usize {
        return offset + self.base_offset;
    }
}

/// Implements signal initialization for MNIST dataset.
///
/// MNIST dataset is an annotated dataset of handwritten digits.
/// More on that here: https://en.wikipedia.org/wiki/MNIST_database
impl ut::data::Dataset for MnistTrainingState<'_> {
    fn copy_training_input_signal(&self, image_index: usize,
            signal: &mut ut::data::Signal) {
        let start_position = (self.offset_as_relative(image_index)) * IMG_SIZE_BYTES;
        <[u8] as ut::data::CopyConvertIntoSignal>::copy_convert_into_signal(
            &self.dataset.trn_img[start_position..start_position + IMG_SIZE_BYTES],
            signal
        );
    }

    fn copy_training_output_signal(&self, image_index: usize,
            signal: &mut ut::data::Signal) {
        let position = self.dataset.trn_lbl[self.offset_as_relative(image_index)] as usize;
        signal.reserve_exact(MNIST_OUTPUT_LAYER_SIZE);
        signal.resize(MNIST_OUTPUT_LAYER_SIZE, 0.0f32);
        signal[position] = 1.0;
    }

    fn length(&self) -> usize {
        self.dataset.trn_lbl.len() - self.base_offset
    }
}

const NETWORK_FILE: &str = "network.bin";

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
    use super::IMG_SIZE_BYTES;
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
        assert!(trn_img.len() == IMG_SIZE_BYTES * TRAINING_SET_LEN);
        assert!(trn_lbl.len() == TRAINING_SET_LEN);
        assert!(tst_img.len() == IMG_SIZE_BYTES * TEST_SET_LEN);
        assert!(tst_lbl.len() == TEST_SET_LEN);
    }
}

/// Trains network using back propagation algorithm. It can resume previously
/// started training session, if ibegin_training_image > 0
fn train_network(net: &mut network::Network, mnist: &Mnist, ibegin_training_image: usize) {
    let mut mnist_training_state = MnistTrainingState{
        dataset: mnist,
        base_offset: ibegin_training_image,
    };
    let activation_function_family = algorithm::ActivationFunctionFamily::StepFunction;
    let cost_function = algorithm::func::cost_mse_d;
    let training_rate = 0.001f32;
    algorithm::train_network_back_propagation(net, activation_function_family,
        cost_function, training_rate, &mnist_training_state,
        |iteration_number| {
            println!("{} of {}",
                iteration_number,
                (&mnist_training_state as &dyn ut::data::Dataset).length());
        });
}

/// Runs forward propagation on a network, measures its performance.
fn test_network(net: &mut network::Network, mnist: &Mnist) {
}

fn make_network() -> network::Network {
    match ut::network_deserialize_from_file(NETWORK_FILE) {
        Err(_) => network::Network::from_geometry(&NETWORK_GEOMETRY.into()),
        Ok(net) => net,
    }
}

pub fn main() {
    use std::env::args;
    const TRAINING_SET_LEN: usize = 2000;
    const TEST_SET_LEN: usize = 100;

    let args: Vec<String> = args().collect();
    let mnist = mnist_load(TRAINING_SET_LEN, TEST_SET_LEN);
    let starting_image_index = args[1].parse::<i32>().unwrap();  // If true, there will be attempt to load an existing network
    let should_run_recognition = starting_image_index < 0;
    let mut network = make_network();

    if should_run_recognition {
        test_network(&mut network, &mnist);
    } else {
        train_network(&mut network, &mnist, starting_image_index as usize);
    }
}
