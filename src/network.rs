
///

pub use std;
use std::{assert, iter::Iterator, vec::Vec, iter::Flatten, iter::FlatMap, iter::Map};
use crate::ut::data::Signal;
use core::cmp;
use crate::ut;

pub type Edge = Vec<Vec<f32>>;
pub type Coeff = Vec<f32>;
pub type LayerTuple<'a> = (&'a Coeff, &'a Coeff, &'a Edge, &'a Edge);
pub type OwnedLayerTuple = (Coeff, Coeff, Edge, Edge);

pub struct Layer {
    /// Weighed sum from the previous layer
    z: Coeff,
    /// Activation function of the weighed sum
    a: Coeff,
    /// Weigts
    w: Edge,
    /// Biases
    b: Edge,
}

impl Layer {
    /// Provides native representation, e.g. for serialization libraries
    pub fn as_tuple(&self) -> LayerTuple {
        (&self.z, &self.a, &self.w, &self.b)
    }

    pub fn from_layer_tuple(layer_tuple: LayerTuple) -> Layer {
        Layer {
            z: layer_tuple.0.clone(),
            a: layer_tuple.1.clone(),
            w: layer_tuple.2.clone(),
            b: layer_tuple.3.clone(),
        }
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
    pub fn geometry(&self) -> Vec<usize> {
        (0..self.layers.len()).map(|layer_id| self.layers[layer_id].a.len()).collect()
    }

    /// Provides std-native representation
    pub fn as_layer_tuple_vec(&self) -> Vec<LayerTuple> {
        let mut ret = Vec::new();
        ret.reserve_exact(self.n_layers());

        for layer in &self.layers {
            ret.push(layer.as_tuple());
        }

        ret
    }

    /// Constructs a network from a set of weights. A part of deserialization
    /// process.
    pub fn from_layer_tuple_vec(layer_tuple_vec: &Vec<OwnedLayerTuple>) -> Network {
        let geometry = layer_tuple_vec.iter()
            .map(|layer_tuple| layer_tuple.0.len())
            .collect::<Vec<usize>>();
        let layers = layer_tuple_vec
            .iter()
            .map(|layer_tuple| Layer::from_layer_tuple(
                (&layer_tuple.0, &layer_tuple.1, &layer_tuple.2, &layer_tuple.3)
            ))
            .collect::<Vec<Layer>>();
        let network = Network{layers};

        network
    }

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

    /// Allocates a chunk in memory for a network with a specified geometry,
    /// i.e. number of nodes on each layer.
    ///
    /// `geometry` specifies how many nodes reside on a layer. Indices of
    /// `geometry` items are layer indices in the network.
    ///
    /// Post: the network will be initialized w/ NAN values
    pub fn from_geometry(geometry: &std::vec::Vec<usize>) -> Network {
        let mut network = Network{
            layers: std::vec::Vec::new(),
        };
        network.layers.reserve_exact(geometry.len());
        let mut size_prev = 0;

        for nnodes in geometry {
            let mut layer = Layer{
                a: Vec::new(),
                z: Vec::new(),
                w: Vec::new(),
                b: Vec::new(),
            };
            // TODO: optimize input and output layers. Note the necessity to ensure size consistency when performing (de)serialization
            layer.a.reserve_exact(*nnodes);
            layer.a.resize(*nnodes, f32::NAN);
            layer.w.reserve_exact(size_prev);
            layer.w.resize(size_prev, Vec::new());
            layer.b.reserve_exact(size_prev);
            layer.b.resize(size_prev, Vec::new());

            if size_prev != 0 {
                layer.z.reserve_exact(*nnodes);
                layer.z.resize(*nnodes, f32::NAN);

                for i in 0..size_prev {
                    layer.w[i].reserve_exact(*nnodes);
                    layer.w[i].resize(*nnodes, f32::NAN);
                    layer.b[i].reserve_exact(*nnodes);
                    layer.b[i].resize(*nnodes, f32::NAN);
                }
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

    #[inline]
    pub fn output_layer(&self) -> &Signal {
        &self.layers[self.n_layers() - 1].z
    }

    #[inline]
    /// Returns (WEIGHT, BIAS) pair
    pub fn edge_coef_wb(&self, ilayer: usize, ifrom: usize, ito: usize) -> (f32, f32) {
        (self.w(ilayer, ifrom, ito), self.b(ilayer, ifrom, ito))
    }

    #[inline]
    pub fn w(&self, ilayer: usize, ifrom: usize, ito: usize) -> f32 {
        self.layers[ilayer].w[ifrom][ito]
    }

    #[inline]
    pub fn b(&self, ilayer: usize, ifrom: usize, ito: usize) -> f32 {
        self.layers[ilayer].b[ifrom][ito]
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
    pub fn set_w(&mut self, ilayer: usize, ifrom: usize, ito: usize, val: f32) {
        self.layers[ilayer].w[ifrom][ito] = val
    }

    #[inline]
    pub fn set_b(&mut self, ilayer: usize, ifrom: usize, ito: usize, val: f32) {
        self.layers[ilayer].b[ifrom][ito] = val
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
    pub fn edge_index_iter(&self, ilayer: usize) -> impl Iterator<Item=(usize, usize)> {
        assert!(ilayer > 0 && ilayer < self.n_layers());
        let len_from = self.layer_len(ilayer - 1);
        let len_to = self.layer_len(ilayer);

        (0..len_from)
            .flat_map(move |ifrom| {
                (0..len_to).map(move |ito| (ifrom, ito))
            })
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.a.resize(layer.a.len(), f32::NAN);
            layer.z.resize(layer.z.len(), f32::NAN);

            for w in layer.w.iter_mut().flat_map(|edge| edge.iter_mut()) {
                *w = f32::NAN;
            }

            for b in layer.b.iter_mut().flat_map(|edge| edge.iter_mut()) {
                *b = f32::NAN;
            }
        }
    }
}

#[cfg(test)]
mod test_network {
    use super::Network;

    #[test]
    fn construction() {
        let geometry = vec![128, 16, 32, 4];
        let network = Network::from_geometry(&geometry);

        for i in 0..geometry.len() {
            assert_eq!(network.layer_len(i), geometry[i]);
        }
    }
}

impl cmp::PartialEq for Network {
    fn eq(&self, other: &Self) -> bool {
        let geometry = self.geometry();

        if geometry == other.geometry() {
            let mut res = true;

            for i in 0..geometry.len() {
                res = res && ut::vecf32_float_safe_is_eq(&self.layers[i].a, &other.layers[i].a);
                res = res && ut::vecf32_float_safe_is_eq(&self.layers[i].z, &other.layers[i].z);
                res = res && self.layers[i].b.iter().zip(other.layers[i].b.iter())
                    .fold(true, |acc, item| acc && ut::vecf32_float_safe_is_eq(&item.0, &item.1));
                res = res && self.layers[i].w.iter().zip(other.layers[i].w.iter())
                    .fold(true, |acc, item| acc && ut::vecf32_float_safe_is_eq(&item.0, &item.1));

                if !res {
                    break
                }
            }

            res
        } else {
            false
        }
    }
}
