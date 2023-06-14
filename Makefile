MNIST_DEBUG_EXECUTABLE_RELATIVE_PATH = "target/debug/mnist"

default:

run_mnist_debug:
	RUST_BACKTRACE=1 RUST_LOG=trace cargo run --bin mnist 0
