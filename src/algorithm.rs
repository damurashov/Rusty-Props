/// Neural network activation and training algorithms.
///

use crate::network;
use std::assert;

pub type Signal = std::vec::Vec<f32>;

pub struct Prop {
	activate: fn(f32) -> f32,
}

impl Prop {
	/// Forward propagation between adjacent layers
	fn network_update_layer_propagate(&self, net: &mut network::Network, ilayer: usize) {
		assert!(ilayer > 0);

		for ito in 1..net.layer_len(ilayer) {
			let mut sum = 0.0f32;

			for ifrom in 1..net.layer_len(ilayer - 1) {
				let edge = net.edge(ilayer, ifrom, ito);
				sum += net.a(ilayer - 1, ifrom) * edge.w + edge.b;
			}

			net.set_z(ilayer, ito, sum);
		}
	}

	pub fn run(&self, net: &mut network::Network, input: &Signal) {
		net.init_input_layer(input);

		for ilayer in 1..net.n_layers() {
			self.network_update_layer_propagate(net, ilayer);
		}
	}
}

#[cfg(test)]
mod tests {
//	#[test]
//	fn
}
