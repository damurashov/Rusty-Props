/// Neural network activation and training algorithms.
///

use crate::network;
use std::{assert, vec::Vec};
use rand::distributions::{Distribution, Uniform};
use network::Network;

pub type Signal = std::vec::Vec<f32>;

/// Randomly initializes weights and biases of a network.
pub fn network_init_random(net: &mut network::Network) {
	let mut rng = rand::thread_rng();
	let gen = Uniform::from(0.0f32..1.0f32);

	for ilayer in 1..net.n_layers() {
		for (ifrom, ito) in net.edge_index_iter(ilayer) {
			net.set_w(ilayer, ifrom, ito, gen.sample(&mut rng));
			net.set_b(ilayer, ifrom, ito, gen.sample(&mut rng));
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
				let (w, b) = net.edge_coef_wb(ilayer, ifrom, ito);
				sum += net.a(ilayer - 1, ifrom) * w + b;
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
	/// Learning rate
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

	fn dcda(&mut self, ilayer: usize, inode: usize, net: &network::Network, reference: &Signal) -> f32 {
		0.0f32
	}

	fn dzdw(&mut self, ilayer: usize, inode: usize, net: &Network, reference: &Signal) -> f32 {
		0.0f32
	}

	fn dcdz(&mut self, ilayer: usize, inode: usize, net: &Network, reference: &Signal) -> f32 {
		let mut ret = self.net_cache.z(ilayer, inode);

		if ret.is_nan() {
			let z = net.z(ilayer, inode);

			ret = if ilayer == net.n_layers() - 1 {
				let ref_z = reference[inode];

				(self.dcdz_output)(ref_z, z)
			} else {
				let dcda = self.dcda(ilayer, inode, net, reference);
				let dadz = (self.dadz)(z);

				dcda * dadz
			}
		}

		self.net_cache.set_z(ilayer, inode, ret);

		ret
	}

	/// Calculates a partial derivative C by w
	fn dcdw(&mut self, ilayer: usize, ifrom: usize, ito: usize, net: &network::Network, reference: &Signal) -> f32 {
		let mut ret = self.net_cache.w(ilayer, ifrom, ito);

		// There is nothing in the cache, calculate
		if ret.is_nan() {
			// The output layer dc/dz is calculated w/ the use of the user-provided cost function derivative.
			let dcdz = self.dcdz(ilayer, ito, net, reference);
			let dzdw = self.dzdw(ilayer, ito, net, reference);
			ret = dcdz * dzdw;
			self.net_cache.set_w(ilayer, ifrom, ito, ret);
		}

		ret
	}

	fn dcdb(&mut self, ilayer: usize, ifrom: usize, ito: usize, net: &Network, reference: &Signal) -> f32 {
		0.0
	}

	/// Train the net using reference output
	/// `network`: the ANN instance
	/// `reference` the reference (desired) output
	///
	/// Pre: the network must be pre-activated, i.e. the output layer must be
	/// initialized by forward propagation
	pub fn run(&mut self, net: &mut network::Network, reference: &Signal) {
		for ilayer in (1..net.n_layers()).rev() {
			for (ifrom, ito) in net.edge_index_iter(ilayer) {
				let w = net.w(ilayer, ifrom, ito) - self.dcdw(ilayer, ifrom, ito, net, reference);
				net.set_w(ilayer, ifrom, ito, w);
				let b = net.b(ilayer, ifrom, ito) - self.dcdb(ilayer, ifrom, ito, net, reference);
				net.set_b(ilayer, ifrom, ito, b);
			}
		}
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
		let _back_propagation = BackPropagation::from_network(&network, dcdz_output_stub, dadz_stub, 0.01);
	}
}
