# TensorSludge
A small, GPU-accelerated neural net library I've built from scratch in order to practice GPU programming and machine learning. Written in Rust, using Vulkan to orchestrate the execution of compute shaders written in GLSL.

## Features
- [X] Matrix read/write from CPU
- [X] Sigmoid function, and derivative
- [X] Matrix multiply with optionally transposed inputs
- [X] Elementwise addition, multiplication, subtraction
- [X] Scalar multiplication
- [X] Automatic synchronization between operations
- [ ] GPU-only matricies
- [ ] Batches of size >1
- [ ] Softmax, and derivative
- [ ] Models, saving and loading
- [ ] Crossentropy, MSE, etc.
- [ ] Better pipelining (Run forward pass of next batch while backpropagating, etc.)
- [ ] Convolutional layers

## Examples
* Basic: Multiply a vector by a matrix and print the output
* MNIST: Train and infer on the MNIST dataset

## Goals
* Ease of modification (project should remain small and easy to follow)
* Relative ease of use
* Good documentation
* Well tested, should provide example code for most functions

## Non-goals
* Excellent performance
* Production-readiness
* Having _every feature_

## Ideas for the future
* Python interface, much like PyTorch or TensorFlow
* Real time training experiments
* 3D visualizations of internals working in real time (The memory is already on-GPU, why not display it?)
