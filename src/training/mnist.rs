/// MNIST is an annotated dataset of handwritten digits.
/// http://yann.lecun.com/exdb/mnist/
///
/// This module trains the network using this dataset.

use mnist::{Mnist, MnistBuilder};

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
