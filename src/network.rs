/// Basic representations pertaining to a neural network.
///

pub use std;
use std::assert;
use crate::algorithm::Signal;

#[derive(Clone)]
pub struct Edge {
	/// Weight
	pub w: f32,
	/// Bias
	pub b: f32,
}

pub struct Layer {
	pub edges: std::vec::Vec<Edge>,
	/// Weighed sum from the previous layer
	pub z: std::vec::Vec<f32>,
	/// Activation function of the weighed sum
	pub a: std::vec::Vec<f32>,
}

impl Layer {
	pub fn edge(&self, inode_from: usize, inode_to: usize) -> &Edge {
		let n_nodes_prev: usize = self.edges.len() / self.z.len();
		// TODO? Check geometry

		&self.edges[n_nodes_prev * inode_from + inode_to]
	}

	pub fn len(&self) -> usize {
		self.a.len()
	}
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
	pub fn n_layers(&self) -> usize {
		self.layers.len()
	}

	pub fn layer_len(&self, ilayer: usize) -> usize {
		self.layers[ilayer].len()
	}

	pub fn layer_mut(&mut self, ilayer: usize) -> &mut Layer {
		assert!(ilayer < self.n_layers());

		&mut self.layers[ilayer]
	}

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
	pub fn edge(&self, ilayer: usize, ifrom: usize, ito: usize) -> &Edge {
		assert!(ilayer > 0);

		&self.layers[ilayer].edges[self.layer_len(ilayer - 1) * ifrom + ito]
	}

	pub fn edge_mut(&mut self, ilayer: usize, ifrom: usize, ito: usize) -> &mut Edge {
		assert!(ilayer > 0);

		let iedge = self.layer_len(ilayer - 1) * ifrom + ito;
		&mut self.layers[ilayer].edges[iedge]
	}

	pub fn a(&self, ilayer: usize, inode: usize) -> f32 {
		self.layers[ilayer].a[inode]
	}

	pub fn set_z(&mut self, ilayer: usize, inode: usize, val: f32) {
		self.layers[ilayer].z[inode] = val;
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
