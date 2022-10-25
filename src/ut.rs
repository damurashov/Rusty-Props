use crate::algorithm::Signal;
use crate::network::Network;
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
