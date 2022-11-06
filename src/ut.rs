use crate::algorithm::Signal;
use crate::network::Network;
use std::{
	vec::Vec,
	fs::File,
	io::{
		Write,
		BufWriter
	},
	path::Path,
};
use rand::distributions::{Distribution, Uniform};
use bincode;

fn signal_stub_from_network(network: &Network, ilayer: usize) -> Signal {
	let len = network.layer_len(ilayer);
	let mut signal = Signal::new();
	signal.reserve_exact(len);
	signal.resize(len, f32::NAN);

	signal
}

#[inline]
pub fn signal_stub_from_network_input(network: &Network) -> Signal {
	signal_stub_from_network(network, 0)
}

#[inline]
pub fn signal_stub_from_network_output(network: &Network) -> Signal {
	signal_stub_from_network(network, network.n_layers() - 1)
}

pub fn vec_init_random<T: rand::distributions::uniform::SampleUniform>(vec: &mut Vec<T>, from: T, to: T) {
	let mut rng = rand::thread_rng();
	let gen = Uniform::from(from..to);

	for s in vec {
		*s = gen.sample(&mut rng);
	}
}

pub fn network_serialize_into_file(network: &Network, fname: &str) -> Result<(), std::io::Error> {
	let path_out = Path::new(fname);
	let mut file_out = File::create(&path_out)?;
	let stream_out = BufWriter::new(&mut file_out);
	let layer_tuple_vec = network.as_layer_tuple_vec();

	match bincode::serialize_into(stream_out, &layer_tuple_vec) {
		Ok(_) => Ok(()),
		Err(e) => Err(std::io::Error::new(std::io::ErrorKind::Other, e))
	}
}
