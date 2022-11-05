use std::{error::Error,
	env::current_dir,
	path::{
		Path,
		PathBuf
	},
	fs:: {
		File,
		create_dir,
	},
	io::{
		Write,
		BufReader,
		BufWriter,
		copy
	},
	process::Command
};
use tokio;
use reqwest;
use lazy_static::lazy_static;
use libflate;

mod dataset {
	use super::*;

	static MNIST_URLS: &[&str] = &[
		"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
		"http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
		"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
		"http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
	];

	static MNIST_FILES: &[&str] = &[
		"train-images-idx3-ubyte",
		"train-labels-idx1-ubyte",
		"t10k-images-idx3-ubyte",
		"t10k-labels-idx1-ubyte",
	];

	lazy_static! {
		static ref PATH_BASE: PathBuf = Path::new(current_dir().unwrap().to_str().unwrap()).join("data");
	}

	/// Fetches the dataset
	pub async fn fetch() -> Result<(), Box<dyn Error>> {
		// The `mnist` package "expects" the dataset to reside in `data/` directory in the project's root
		// https://docs.rs/mnist/latest/mnist/#setup
		for url in MNIST_URLS {
			let file_name = Path::new(url).file_name().unwrap();
			let mut file_path = PATH_BASE.clone();
			create_dir(&file_path);
			file_path.push(&file_name);
			let mut file = File::create(&file_path)?;
			let response = reqwest::get(*url).await?;
			let content = response.bytes().await?;
			file.write_all(&content)?;
		}

		Ok(())
	}

	/// Deflates `gzip`s
	///
	/// Pre: `data/` directory must exist
	pub fn unpack() {
		for fname in MNIST_FILES {
			let mut path_out= PATH_BASE.clone();
			path_out.push(fname);
			let mut path_in = path_out.clone();
			path_in.set_extension("gz");
			let mut file_in = File::open(&path_in).unwrap();
			let mut file_out = File::create(&path_out).unwrap();
			let mut stream_in = BufReader::new(&file_in);
			let mut stream_out = BufWriter::new(&mut file_out);
			let mut gz_decoder = libflate::gzip::Decoder::new(&mut stream_in).unwrap();
			copy(&mut gz_decoder, &mut stream_out);
		}
	}
}

fn main() -> Result<(), Box<dyn Error>> {
	let mut rt = tokio::runtime::Runtime::new().unwrap();
	rt.block_on(async {dataset::fetch().await})?;
	dataset::unpack();

	Ok(())
}
