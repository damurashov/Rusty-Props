use core::ops::Index;

pub type Signal = std::vec::Vec<f32>;

/// `M` - metadatata (label) type, annotation associated w/ the payload
///
/// No boundary check functionality is implied.
pub trait Dataset
{
    fn copy_training_input_signal(&self, image_index: usize,
        signal: &mut Signal);
    fn copy_training_output_signal(&self, image_index: usize,
        signal: &mut Signal);
    fn nimages(&self) -> usize;
}

/// Helper trait for quickly initializing `Signal` instances from various types'
/// instances
pub trait CopyConvertIntoSignal {
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
