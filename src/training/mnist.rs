use mnist::{Mnist, MnistBuilder};

/// Waterprobing. An attempt to load and unpack MNIST dataset
#[cfg(test)]
mod test_mnist_load {
	use super::{Mnist, MnistBuilder};
	use std::{env::current_dir, thread::current};

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
			.training_set_length(100)
			.test_set_length(10)
			.base_path(path_base_str)
			.finalize();
	}
}
