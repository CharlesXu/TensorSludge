Goals:
* Simple! Don't overextend yourself. 
    * Barriers after every step; this is a line, not a graph
* Don't be afraid to make _this_ version totally inefficient just to get it to work
* Train and evaluate on the MNIST dataset

Implementation:
* Per compute shader:
    * One descriptor set layout 
    * One pipeline layout
    * One pipeline 
* Compute shaders should be builtin ofc
* Pass creation:
    * Create DescriptorSets
    * Allocate and write command buffers
* Matrices:
    * Allow read/write, are created uninitialized..?

Interfacing:
```rust
let mut ts = TensorSludge::new();

const IMAGE_WIDTH: usize = 28;
const IMAGE_SIZE: usize = IMAGE_WIDTH * IMAGE_WIDTH;
const HIDDEN_SIZE: usize = 64;
const N_CLASSES: usize = 10;

let l0_weights = ts.matrix(HIDDEN_SIZE, IMAGE_SIZE);
let l1_weights = ts.matrix(HIDDEN_SIZE, HIDDEN_SIZE);
let l2_weights = ts.matrix(N_CLASSES, HIDDEN_SIZE);

let input_mat = ts.matrix(IMAGE_SIZE, 1); // Also l0_activations
let l1_activations = ts.matrix(HIDDEN_SIZE, 1);
let l2_activations = ts.matrix(HIDDEN_SIZE, 1);
let output_mat = ts.matrix(N_CLASSES, 1); // Also l3_activations

let forward_pass = [
    Op::MatrixMultiply {
        left: l0_weights,
        right: input_mat,
        dst: l1_activations,
    },
    Op::Sigmoid(l1_activations),
    Op::MatrixMultiply {
        left: l1_weights,
        right: l1_activations,
        dst: l2_activations
    },
    Op::Sigmoid(l2_activations),
    Op::MatrixMultiply {
        left: l2_weights,
        right: l2_activations,
        dst: output_mat,
    },
];

let forward_pass = ts.make_pass(&forward_pass);

let mut rng = rand::thread_rng();
let unif = Uniform::new(-1., 1.);
ts.write(l0_weights, &unif.sample_iter(&mut rng).take(l0_weights.size()).collect());
ts.write(l1_weights, &unif.sample_iter(&mut rng).take(l1_weights.size()).collect());
ts.write(l2_weights, &unif.sample_iter(&mut rng).take(l2_weights.size()).collect());

for (x, y) in mnist.train {
    ts.write(input_mat, x);
    ts.flow(forward_pass); // Does a vkQueueWaitIdle()
    let output = ts.read(output_mat);
    softmax(output)

}

```
