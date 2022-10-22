/// Neural network activation and training algorithms.
///

use crate::network;
use std::{assert, vec::Vec};
use rand::distributions::{Distribution, Uniform};

pub type Signal = std::vec::Vec<f32>;

/// Randomly initializes weights and biases of a network.
pub fn network_init_random(net: &mut network::Network) {
	let mut rng = rand::thread_rng();
	let gen = Uniform::from(0.0f32..1.0f32);

	for ilayer in 0..net.n_layers() {
		for edge in net.edges_iter_mut(ilayer) {
			edge.w = gen.sample(&mut rng);
			edge.b = gen.sample(&mut rng);
		}
	}
}

#[cfg(test)]
mod tests_module_functions {
	use crate::network::Network;
	use super::network_init_random;

	#[test]
	fn test_network_random_initialization() {
		let geometry = vec![128, 16, 32, 4];
		let mut network = Network::from_geometry(&geometry);
		network_init_random(&mut network);
	}
}

pub struct ForwardPropagation {
	activate: fn(f32) -> f32,
}


impl ForwardPropagation {
	/// Forward propagation between adjacent layers
	fn network_update_layer_propagate(&self, net: &mut network::Network, ilayer: usize) {
		assert!(ilayer > 0);

		for ito in 0..net.layer_len(ilayer) {
			let mut sum = 0.0f32;

			for ifrom in 0..net.layer_len(ilayer - 1) {
				let edge = net.edge(ilayer, ifrom, ito);
				sum += net.a(ilayer - 1, ifrom) * edge.w + edge.b;
			}

			net.set_z(ilayer, ito, sum);
		}
	}

	/// Activation of "sum" nodes
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
mod test_forward_propagation {
	use super::{ForwardPropagation, network_init_random, Signal};
	use crate::network;

	#[inline]
	fn step_function(x: f32) -> f32 {
		if x < 0.0f32 {
			0.0f32
		} else {
			x
		}
	}

	#[test]
	fn test_forward_propagation() {
		let geometry = vec![128, 16, 32, 4];
		let mut network = network::Network::from_geometry(&geometry);
		let mut signal = Signal::new();
		network_init_random(&mut network);
		let forward_propagation = ForwardPropagation {
			activate: step_function,
		};
		forward_propagation.run(&mut network, &signal);
	}
}

/// Cost function derivative for the output layer
/// arg. 1: desired output layer value
/// arg. 2: factual output layer value
pub type Dcdz = fn(f32, f32) -> f32;
/// Derivative of activation function by the weighed sum
/// arg. 1: weighed sum value
pub type Dadz = fn(f32) -> f32;

struct BackPropagation {
	dcdz_output: Dcdz,
	dadz: Dadz,
	net_cache: network::Network,
	epsilon: f32,
}

impl BackPropagation {
	// TODO dcdz
	// TODO dcda
	// TODO dcdw
	// TODO dcdb
	// TODO dzdw
	// TODO dzdb
	// TODO dzda
	// TODO dadz

	pub fn from_network(net: &network::Network, dcdz_output: Dcdz, dadz: Dadz, epsilon: f32) -> BackPropagation {
		let geometry: Vec<usize> = (0..net.n_layers()).map(|i| net.layer_len(i)).collect();
		BackPropagation {
			dcdz_output,
			dadz,
			net_cache: network::Network::from_geometry(&geometry),
			epsilon
		}
	}

	pub fn run(net: &mut network::Network, signal: &Signal) {
		// TODO
	}
}

#[cfg(test)]
mod test_back_propagation {
	use super::BackPropagation;
	use crate::network::Network;

	fn dcdz_output_stub(reference: f32, actual: f32) -> f32 {
		f32::NAN
	}

	fn dadz_stub(dz: f32) -> f32 {
		f32::NAN
	}

	#[test]
	fn test_back_propagation_construction() {
		let geometry = vec![128, 16, 32, 4];
		let network = Network::from_geometry(&geometry);
		let _back_propagation = BackPropagation::from_network(&network, dcdz_output_stub, dadz_stub);
	}
}
