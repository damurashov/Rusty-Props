/// Neural network activation and training algorithms.

pub mod func;

use crate::{network, ut};
use std::{assert, vec::Vec};
use rand::distributions::{Distribution, Uniform};
use network::Network;
pub use crate::ut::data::Signal;

/// Randomly initializes weights and biases of a network.
pub fn network_init_random(net: &mut network::Network) {
    let mut rng = rand::thread_rng();
    let gen = Uniform::from(0.0f32..1.0f32);
    network_init_with_generator(net, &mut || gen.sample(&mut rng));
}

/// Initializes weights and biases of a network with a constant value
pub fn network_init_with_value(net: &mut network::Network, val: f32) {
    network_init_with_generator(net, &mut || val);
}

/// Initializes the network with a provided generator.
fn network_init_with_generator(net: &mut network::Network, generator: &mut dyn FnMut() -> f32) {
    for ilayer in 1..net.n_layers() {
        for (ifrom, ito) in net.edge_index_iter(ilayer) {
            net.set_w(ilayer, ifrom, ito, generator());
            net.set_b(ilayer, ifrom, ito, generator());
        }
    }
}

#[cfg(test)]
mod test_module_functions {
    use crate::network::Network;
    use super::network_init_random;

    #[test]
    fn network_random_initialization() {
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

        for i in 0..net.layer_len(ilayer) {
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
    use super::{
        ForwardPropagation,
        network_init_random,
        Signal,
        func::activation_step,
        Uniform, Distribution
    };
    use crate::{network, ut};

    #[test]
    fn run() {
        let geometry = vec![2, 4, 2, 4];
        let mut network = network::Network::from_geometry(&geometry);
        // Initialize random input
        let mut signal = ut::signal_stub_from_network_input(&network);
        ut::vec_init_random(&mut signal, 0.0f32, 1.0f32);

        network_init_random(&mut network);
        let forward_propagation = ForwardPropagation {
            activate: activation_step,
        };
        forward_propagation.run(&mut network, &signal);
        let ilayer = network.n_layers() - 1;
        let layer_len = network.layer_len(ilayer);

        for inode in 0..layer_len {
            assert!(!network.a(ilayer, inode).is_nan());
        }
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
    /// Const function
    dcdz_output: Dcdz,
    /// Activation function
    dadz: Dadz,
    /// Intermediate results storage
    net_cache: network::Network,
    /// Learning rate
    epsilon: f32,
}

impl BackPropagation {
    // TODO dcda
    // TODO dzda

    pub fn from_network(net: &network::Network,
            dcdz_output: Dcdz,
            dadz: Dadz, epsilon: f32
    ) -> BackPropagation {
        let geometry: Vec<usize> = (0..net.n_layers())
            .map(|i| net.layer_len(i)).collect();
        BackPropagation {
            dcdz_output,
            dadz,
            net_cache: network::Network::from_geometry(&geometry),
            epsilon
        }
    }

    fn dzda(
        &mut self,
        ialayer: usize,
        ia: usize,
        iz: usize,
        net: &Network,
        reference: &Signal
    ) -> f32 {
        net.w(ialayer + 1, ia, iz)
    }

    /// Returns partial derivative C by a
    ///
    /// `ialayer` - index of the layer on which `a` resides
    /// `ia` - index of `a`
    fn dcda(
        &mut self,
        ialayer: usize,
        ia: usize,
        net: &network::Network,
        reference: &Signal
    ) -> f32 {
        let mut ret = self.net_cache.a(ialayer, ia);

        if ret.is_nan() {
            ret = 0.0;

            for iz in 0..net.layer_len(ialayer + 1) {
                let dcdz = self.dcdz(ialayer + 1, iz, net, reference);
                let dzda = self.dzda(ialayer, ia, iz, net, reference);
                ret += dcdz * dzda;
            }
        }

        self.net_cache.set_a(ialayer, ia, ret);

        ret
    }

    /// Returns partial derivative z by w
    ///
    /// `ilayer` - layer of w
    /// `ifrom` - id of the edge's originating node
    fn dzdw(
        &self,
        iwlayer: usize,
        ifrom: usize,
        ito: usize,
        net: &Network,
        reference: &Signal
    ) -> f32 {
        net.a(iwlayer - 1, ifrom)
    }

    #[inline]
    fn dzdb(
        &self,
        ilayer: usize,
        ifrom: usize,
        ito: usize,
        net: &Network,
        reference: &Signal
    ) -> f32 {
        1.0
    }

    /// Calculates partial derivative C by z
    fn dcdz(
        &mut self,
        izlayer: usize,
        iz: usize,
        net: &Network,
        reference: &Signal
    ) -> f32 {
        let mut ret = self.net_cache.z(izlayer, iz);

        if ret.is_nan() {
            let z = net.z(izlayer, iz);

            ret = if izlayer == net.n_layers() - 1 {
                let ref_z = reference[iz];

                (self.dcdz_output)(ref_z, z)
            } else {
                let dcda = self.dcda(izlayer, iz, net, reference);
                let dadz = (self.dadz)(z);

                dcda * dadz
            }
        }

        self.net_cache.set_z(izlayer, iz, ret);

        ret
    }

    /// Calculates a partial derivative C by w
    fn dcdw(
        &mut self,
        ilayer: usize,
        ifrom: usize,
        ito: usize,
        net: &network::Network,
        reference: &Signal
    ) -> f32 {
        let mut ret = self.net_cache.w(ilayer, ifrom, ito);

        // There is nothing in the cache, calculate
        if ret.is_nan() {
            // The output layer dc/dz is calculated w/ the use of the user-provided cost function derivative.
            let dcdz = self.dcdz(ilayer, ito, net, reference);
            let dzdw = self.dzdw(ilayer, ifrom, ito, net, reference);
            ret = dcdz * dzdw;
            self.net_cache.set_w(ilayer, ifrom, ito, ret);
        }

        ret
    }

    /// Calculates a partial derivative C by b
    fn dcdb(
        &mut self,
        ilayer: usize,
        ifrom: usize,
        ito: usize,
        net: &Network,
        reference: &Signal
    ) -> f32 {
        let mut ret = self.net_cache.b(ilayer, ifrom, ito);

        if ret.is_nan() {
            let dcdz = self.dcdz(ilayer, ito, net, reference);
            let dzdb = self.dzdb(ilayer, ifrom, ito, net, reference);
            ret = dcdz * dzdb;
            self.net_cache.set_b(ilayer, ifrom, ito, ret);
        }

        ret
    }

    /// Train the net using reference output
    /// `network`: the ANN instance
    /// `reference` the reference (desired) output
    ///
    /// Pre: the network must be pre-activated, i.e. the output layer must be
    /// initialized by forward propagation
    pub fn run(&mut self, net: &mut network::Network, reference: &Signal) {
        self.net_cache.reset();

        for ilayer in (1..net.n_layers()).rev() {
            for (ifrom, ito) in net.edge_index_iter(ilayer) {
                let w = net.w(ilayer, ifrom, ito)
                    - self.dcdw(ilayer, ifrom, ito, net, reference) * self.epsilon;
                net.set_w(ilayer, ifrom, ito, w);
                let b = net.b(ilayer, ifrom, ito)
                    - self.dcdb(ilayer, ifrom, ito, net, reference) * self.epsilon;
                net.set_b(ilayer, ifrom, ito, b);
            }
        }
    }
}

#[cfg(test)]
mod test_back_propagation {
    use super::{BackPropagation, network_init_random, Signal, func, ForwardPropagation};
    use rand::distributions::{Uniform, Distribution};
    use crate::network::Network;
    use crate::ut;

    fn dcdz_output_stub(reference: f32, actual: f32) -> f32 {
        f32::NAN
    }

    fn dadz_stub(dz: f32) -> f32 {
        f32::NAN
    }

    #[test]
    fn construction() {
        let geometry = vec![128, 16, 32, 4];
        let network = Network::from_geometry(&geometry);
        let _back_propagation = BackPropagation::from_network(&network,
            dcdz_output_stub, dadz_stub, 0.01);
    }

    /// Runs full circle, viz. forward and back propagation, to make sure it
    /// won't fail
    #[test]
    fn run() {
        // Initialize network, its input, and output (reference) signals

        let geometry = vec![2, 2, 2];
        let mut network = Network::from_geometry(&geometry);
        network_init_random(&mut network);
        let mut signal_input = ut::signal_stub_from_network_input(&network);
        let mut signal_output = ut::signal_stub_from_network_output(&network);
        let epsilon = 0.01f32;
        ut::vec_init_random(&mut signal_input, 0.0f32, 1.0f32);
        ut::vec_init_random(&mut signal_output, 0.0f32, 1.0f32);

        // Initialize forward and back propagation algorithms w/ cost and activation functions
        let mut back_propagation = BackPropagation::from_network(&network,
            func::cost_mse_d,
            func::activation_step_d, epsilon);
        let forward_propagation = ForwardPropagation{activate: func::activation_step};

        /// Run fwd. and back propagation algorithms
        forward_propagation.run(&mut network, &signal_input);
        back_propagation.run(&mut network, &signal_output);

        for ilayer in 1..back_propagation.net_cache.n_layers() {
            for inode in 0..back_propagation.net_cache.layer_len(ilayer) {
                assert!(ilayer == back_propagation.net_cache.n_layers() - 1
                    || !back_propagation.net_cache.a(ilayer, inode).is_nan());  // NAN -> LAST LAYER, or !LAST_LAYER -> !NAN
                assert!(!back_propagation.net_cache.z(ilayer, inode).is_nan());

                for ifrom in 0..back_propagation.net_cache.layer_len(ilayer - 1) {
                    assert!(!back_propagation.net_cache.w(ilayer, ifrom, inode).is_nan());
                    assert!(!back_propagation.net_cache.b(ilayer, ifrom, inode).is_nan());
                }
            }
        }
    }
}

pub enum ActivationFunctionFamily {
    StepFunction = 0,
}

/// Forward / derivative activation function pairs
const ACTIVATION_FUNCTION_FAMILY_MAPPING: [(fn(f32) -> f32, fn(f32) -> f32); 1] = [
    (func::activation_step, func::activation_step_d),
];

/// Encapsulates training / recognition profile
struct ActivationProfile {
    activation_function: fn(f32) -> f32,
    activation_function_derivative: fn(f32) -> f32,
}

impl ActivationProfile {
    fn new(activation_function_family: ActivationFunctionFamily) -> ActivationProfile {
        let id = activation_function_family as usize;
        ActivationProfile {
            activation_function: ACTIVATION_FUNCTION_FAMILY_MAPPING[id].0,
            activation_function_derivative:
                ACTIVATION_FUNCTION_FAMILY_MAPPING[id].1,
        }
    }
}

// pub fn train_network_forward_propagation(net: &mut Network,
//         activation: ActivationFunctionFamily,
//         cost_function: fn(f32, f32) -> f32, training_rate: f32,
//         dataset: ut::data::Dataset) {
//     let ActivationProfile{activation_function, activation_function_derivative}
//         = ActivationProfile::new(activation);
//     let mut forward_propagation = ForwardPropagation{activate: activation_function};
//     let mut back_propagation = BackPropagation::from_network(net,
//         cost_function, activation_function_derivative, training_rate);
// }
