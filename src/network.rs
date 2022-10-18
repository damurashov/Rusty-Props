/// Basic representations pertaining to a neural network.
///

pub use std;
use std::assert;

struct Edge {
	/// Weight
	w: f32,
	/// Bias
	b: f32,
}

pub struct Layer {
	edges: std::vec::Vec<Edge>,
	/// Weighed sum from the previous layer
	z: std::vec::Vec<f32>,
	/// Activation function of the weighed sum
	a: std::vec::Vec<f32>,
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

	pub fn a(&self, ilayer: usize, inode: usize) -> f32 {
		self.layers[ilayer].a[inode]
	}

	pub fn set_z(&mut self, ilayer: usize, inode: usize, val: f32) {
		self.layers[ilayer].z[inode] = val;
	}
}
