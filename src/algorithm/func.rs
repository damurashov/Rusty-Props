
/// Step function
pub fn activation_step(z: f32) -> f32 {
	if z < 0.0 {
		0.0
	} else {
		z
	}
}

/// da/dz
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
pub fn cost_mse_d(reference: f32, value: f32) -> f32 {
	-2.0f32 * reference + 2.0f32 * value
}
