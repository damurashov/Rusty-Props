/// Basic representations pertaining to a neural network.
///

pub use std;
use std::{assert, iter::Iterator};
use crate::algorithm::Signal;

#[derive(Clone)]
pub struct Edge {
	/// Weight
	pub w: f32,
	/// Bias
	pub b: f32,
}

pub struct Layer {
	edges: std::vec::Vec<Edge>,
	/// Weighed sum from the previous layer
	z: std::vec::Vec<f32>,
	/// Activation function of the weighed sum
	a: std::vec::Vec<f32>,
}

/// Stores network weights and the results of intermediate calculations such as
/// partial derivatives.
///
/// Layers are counted from left to right (from input to output), starting from
/// 0. Edges have the same level as their destination nodes.
///
pub struct Network {
	layers: std::vec::Vec<Layer>
}

impl Network {
	/// Number of layers in the network
	#[inline]
	pub fn n_layers(&self) -> usize {
		self.layers.len()
	}

	/// Length of a layer
	#[inline]
	pub fn layer_len(&self, ilayer: usize) -> usize {
		self.layers[ilayer].a.len()
	}

	#[inline]
	pub fn layer_mut(&mut self, ilayer: usize) -> &mut Layer {
		assert!(ilayer < self.n_layers());

		&mut self.layers[ilayer]
	}

	/// Allocates a chunk in memory for a network with a specified geometry,
	/// i.e. number of nodes on each layer.
	///
	/// `geometry` specifies how many nodes reside on a layer. Indices of
	/// `geometry` items are layer indices in the network.
	pub fn from_geometry(geometry: &std::vec::Vec<usize>) -> Network {
		let mut network = Network{
			layers: std::vec::Vec::new(),
		};
		network.layers.reserve_exact(geometry.len());
		let mut size_prev = 0;

		for nnodes in geometry {
			let mut layer = Layer{
				edges: std::vec::Vec::new(),
				a: std::vec::Vec::new(),
				z: std::vec::Vec::new(),
			};
			layer.edges.reserve_exact(size_prev * nnodes);
			layer.edges.resize(size_prev * nnodes, Edge{
				w: 0.0f32,
				b: 0.0f32,
			});
			layer.a.reserve_exact(*nnodes);
			layer.a.resize(*nnodes, 0.0f32);

			if size_prev != 0 {
				layer.z.reserve_exact(*nnodes);
				layer.z.resize(*nnodes, 0.0f32);
			}

			network.layers.push(layer);
			size_prev = *nnodes;
		}

		return network;
	}

	#[inline]
	pub fn init_input_layer(&mut self, signal: &Signal) {
		assert!(self.layers.len() > 0);
		self.layers[0].a = signal.to_vec();
	}

	/// Returns ref. to an edge
	///
	/// `ifrom` - id of the edge's origin node
	/// `ito` - id of the edge's destination node
	///
	/// Expects layers of level 1 and higher (counting from 0)
	///
	#[inline]
	pub fn edge(&self, ilayer: usize, ifrom: usize, ito: usize) -> &Edge {
		assert!(ilayer > 0);

		&self.layers[ilayer].edges[self.layer_len(ilayer - 1) * ito + ifrom]
	}

	#[inline]
	pub fn edge_mut(&mut self, ilayer: usize, ifrom: usize, ito: usize) -> &mut Edge {
		assert!(ilayer > 0);

		let iedge = self.layer_len(ilayer - 1) * ifrom + ito;
		&mut self.layers[ilayer].edges[iedge]
	}

	/// Access activation value on a specified node and layer
	#[inline]
	pub fn a(&self, ilayer: usize, inode: usize) -> f32 {
		self.layers[ilayer].a[inode]
	}

	/// Access the weighted sum value on a specified node and layer
	#[inline]
	pub fn z(&self, ilayer: usize, inode: usize) -> f32 {
		self.layers[ilayer].z[inode]
	}

	#[inline]
	pub fn set_a(&mut self, ilayer: usize, inode: usize, val: f32) {
		self.layers[ilayer].a[inode] = val;
	}

	#[inline]
	pub fn set_z(&mut self, ilayer: usize, inode: usize, val: f32) {
		self.layers[ilayer].z[inode] = val;
	}

	#[inline]
	pub fn edges_iter_mut(&mut self, ilayer: usize) -> impl Iterator<Item = &'_ mut Edge> {
		assert!(ilayer < self.n_layers());
		self.layers[ilayer].edges.iter_mut()
	}
}

#[cfg(test)]
mod test {
	use super::Network;

	#[test]
	fn test_network_construction() {
		let geometry = vec![128, 16, 32, 4];
		let network = Network::from_geometry(&geometry);

		for i in 0..geometry.len() {
			assert_eq!(network.layer_len(i), geometry[i]);
		}
	}
}
