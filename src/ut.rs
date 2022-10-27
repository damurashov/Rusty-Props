use crate::algorithm::Signal;
use crate::network::Network;
use std::vec::Vec;
use rand::distributions::{Distribution, Uniform};

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
