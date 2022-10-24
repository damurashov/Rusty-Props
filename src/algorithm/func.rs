
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
