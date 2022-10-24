use crate::algorithm::Signal;
use crate::network::Network;
use rand::distributions::{Distribution, Uniform};

fn signal_layer_stub(network: &Network, ilayer: usize) -> Signal {
	let len = network.layer_len(ilayer);
	let mut signal = Signal::new();
	signal.reserve_exact(len);
	signal.resize(len, f32::NAN);

	signal
}

#[inline]
pub fn signal_input_stub(network: &Network) -> Signal {
	signal_layer_stub(network, 0)
}

#[inline]
pub fn signal_output_stub(network: &Network) -> Signal {
	signal_layer_stub(network, network.n_layers() - 1)
}
