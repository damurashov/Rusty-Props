/// Neural network activation and training algorithms.
///

use crate::network;
use std::assert;

pub type Signal = std::vec::Vec<f32>;

pub struct ForwardPropagation {
	activate: fn(f32) -> f32,
}

impl ForwardPropagation {
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

	fn network_update_layer_activate(&self, net: &mut network::Network, ilayer: usize) {
		assert!(ilayer > 0);

		for i in 1..net.layer_len(ilayer) {
			let z = net.z(ilayer, i);
			let a = (self.activate)(z);
			net.set_a(ilayer, i, a);
		}
	}

	pub fn run(&self, net: &mut network::Network, input: &Signal) {
		net.init_input_layer(input);

		for ilayer in 1..net.n_layers() {
			self.network_update_layer_propagate(net, ilayer);
			self.network_update_layer_activate(net, ilayer);
		}
	}
}

#[cfg(test)]
mod tests {
//	#[test]
//	fn
}
