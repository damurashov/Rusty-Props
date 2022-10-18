/// Neural network activation and training algorithms.
///

use network;
use std::assert;

pub type Signal = std::vec::Vec<f32>;

pub struct Prop {
	activate: fn(f32) -> f32,
}

impl Prop {
	fn network_init(net: &mut network::Network, input: &Signal) {
		net.layers[0].a = input;
	}

	fn network_update_layer(&self, net: &mut network::Network, ilayer: usize) {
		assert!(ilayer > 0);
		for i in 0..net.layer_len(ilayer) {
			let mut sum = 0.0f32;

			// W = Ax + b, W - vec. of weights, A - vec. of activation function values, b - vector of biases
			for i_prev in 0..layer_len(ilayer - 1) {
				let edge = net.edge(ilayer, i_prev, i)
				sum += edge.w * net.a(ilayer - 1, i_prev) + edge.b;
			}
		}
	}

	pub fn run(&self, net: &mut network::Network) {
		Prop::network_init(net, input);

		for ilayer in 1..net.n_layers() {
			self.network_update_layer(network, ilayer);
		}
	}
}

pub fn network_prop(net: &mut network::Network, input: &Signal) {
	for ilayer in 0..network.n_layers {
	}
}
