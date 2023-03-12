pub use mnist;

const MNIST_IMAGE_SIZE: usize = 28 * 28;

/// `D` - data type, payload that is used for training
/// `M` - metadatat type, annotation associated w/ the payload
///
/// No boundary check functionality is implied.
pub trait Dataset<'a, T, M>
{
    fn training_data(&'a self, at: usize) -> T;
    fn training_metadata(&self, at: usize) -> M;
}

impl<'a> Dataset<'a, &'a [u8], u8> for mnist::Mnist
{
    fn training_data(&self, at: usize) -> &[u8] {
        let start_position = at * MNIST_IMAGE_SIZE;

        &self.trn_img[start_position..start_position + MNIST_IMAGE_SIZE]
    }

    fn training_metadata(&self, at: usize) -> u8 {
        self.trn_lbl[at]
    }
}
