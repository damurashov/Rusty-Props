use std::{error::Error, env::current_dir, path::Path, fs::File, fs::create_dir, io::Write, process::Command};
use tokio;
use reqwest;

mod dataset {
	use super::*;

	static MNIST_URLS: &[&str] = &[
		"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
		"http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
		"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
		"http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
	];

	pub async fn fetch() -> Result<(), Box<dyn Error>> {
		let path_base = Path::new(current_dir().unwrap().to_str().unwrap()).join("data");  // `current_dir/data`

		for url in MNIST_URLS {
			let file_name = Path::new(url).file_name().unwrap();
			let mut file_path = path_base.clone();
			create_dir(&file_path);
			file_path.push(&file_name);
			let mut file = File::create(&file_path)?;
			let response = reqwest::get(*url).await?;
			let content = response.bytes().await?;
			file.write_all(&content)?;
		}

		Ok(())
	}

}

fn main() -> Result<(), Box<dyn Error>> {
	let mut rt = tokio::runtime::Runtime::new().unwrap();
	rt.block_on(async {dataset::fetch().await})?;

	Ok(())
}
