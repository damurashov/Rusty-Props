pub use mnist;
use core::ops::Index;

pub type Signal = std::vec::Vec<f32>;

const MNIST_IMAGE_SIZE: usize = 28 * 28;
const MNIST_OUTPUT_LAYER_SIZE: usize = 10;  // Mnist is a handwritten digits annotated database, 10 digits

/// `M` - metadatata (label) type, annotation associated w/ the payload
///
/// No boundary check functionality is implied.
pub trait Dataset
{
    fn copy_training_input_signal(&self, image_index: usize,
        signal: &mut Signal);
    fn copy_training_output_signal(&self, image_index: usize,
        signal: &mut Signal);

    fn copy_testing_input_signal(&self, image_index: usize,
            signal: &mut Signal) {
        self.copy_training_input_signal(image_index, signal)
    }

    fn copy_testing_output_signal(&self, image_index: usize,
            signal: &mut Signal) {
        self.copy_training_output_signal(image_index, signal);
    }
}

/// Helper trait for quickly initializing `Signal` instances from various types'
/// instances
trait CopyConvertIntoSignal {
    fn copy_convert_into_signal(&self, signal: &mut Signal);
}

/// Helper trait
trait Length {
    fn length(&self) -> usize;
}

/// Implementation for array-like objects (slices, vectors, arrays)
impl<T> CopyConvertIntoSignal for T
where
    T: Index<usize> + Length + ?Sized,
    <T as Index<usize>>::Output: Into<f32> + Copy
{
    fn copy_convert_into_signal(&self, signal: &mut Signal) {
        signal.reserve_exact(self.length());
        signal.resize(self.length(), f32::NAN);

        for i in 0..self.length() {
            signal[i] = self[i].into();
        }
    }
}

/// Implements length acquisition for slices
impl<T> Length for [T] {
    fn length(&self) -> usize {
        self.len()
    }
}

/// Implements signal initialization for MNIST dataset.
///
/// MNIST dataset is an annotated dataset of handwritten digits.
/// More on that here: https://en.wikipedia.org/wiki/MNIST_database
impl Dataset for mnist::Mnist {
    fn copy_training_input_signal(&self, image_index: usize,
            signal: &mut Signal) {
        let start_position = image_index * MNIST_IMAGE_SIZE;
        self.trn_img[start_position..start_position + MNIST_IMAGE_SIZE]
            .copy_convert_into_signal(signal);
    }

    fn copy_training_output_signal(&self, image_index: usize,
            signal: &mut Signal) {
        let position = self.trn_lbl[image_index] as usize;
        signal.reserve_exact(MNIST_OUTPUT_LAYER_SIZE);
        signal.resize(MNIST_OUTPUT_LAYER_SIZE, 0.0f32);
        signal[position] = 1.0;
    }
}
