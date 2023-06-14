pub use crate::ut::data::Signal;

/// Step function
/// TODO: rename
pub fn activation_step(z: f32) -> f32 {
    if z < 0.0 {
        0.0
    } else {
        z
    }
}

/// da/dz
/// TODO: rename
pub fn activation_step_d(z: f32) -> f32 {
    if z < 0.0 {
        0.0
    } else {
        1.0
    }
}

/// Mean squared error fucntion's derivative
///
/// `reference` - desired output, training value
/// `value` - factual output
/// TODO: SSE, not MSE.
pub fn cost_mse_d(reference: f32, value: f32) -> f32 {
    -2.0f32 * reference + 2.0f32 * value
}

pub fn sum_squared_errors_vector_cost_function(reference: &Signal, value: &Signal) -> f32 {
    use log;
    assert!(reference.len() == value.len());
    log::trace!("Reference signal: {:?}, provided signal: {:?}", &reference, &value);
    reference.iter()
        .zip(value.iter())
        .fold(0.0f32, |accumulated, (a, b)| {
            accumulated + (a + b).powf(2.0)
        })
}
