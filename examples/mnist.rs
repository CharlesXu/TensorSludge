use anyhow::Result;
use mnist::MnistBuilder;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use tensorsludge::*;

// Output size is # of rows
fn random_weights(
    mat: Matrix,
    size: usize,
    ts: &mut TensorSludge,
    rng: &mut impl Rng,
) -> Result<()> {
    let unif = Uniform::new(-1., 1.);
    let buf = unif.sample_iter(rng).take(size).collect::<Vec<f32>>();
    ts.write(mat, &buf)
}

fn softmax(data: &mut [f32]) {
    let sum = data.iter().map(|v| v.exp()).sum::<f32>();
    data.iter_mut().for_each(|v| *v /= sum);
}

fn mse(input: &[f32]) -> f32 {
    input.iter().map(|&v| v * v).sum::<f32>() / input.len() as f32
}

/* Wishlist:
 * Sigmoid deriv
 * Elementwise mul
 * Elementwise add
 */

fn main() -> Result<()> {
    let mut ts = TensorSludge::new()?;
    let mnist = MnistBuilder::new().download_and_extract().finalize();

    // Size constants
    const IMG_WIDTH: usize = 28;
    const IMG_SIZE: usize = IMG_WIDTH * IMG_WIDTH;
    const HIDDEN_L1: usize = 128;
    const HIDDEN_L2: usize = 64;
    const OUTPUT_SIZE: usize = 10;

    // Build weight and activation buffers
    let input_layer = ts.matrix(IMG_SIZE, 1)?;

    let weights_l0 = ts.matrix(HIDDEN_L1, IMG_SIZE)?;
    let activations_l0 = ts.matrix(HIDDEN_L1, 1)?;
    let grad_l0 = ts.matrix(HIDDEN_L1, IMG_SIZE)?;
    let error_l0 = ts.matrix(HIDDEN_L1, 1)?;

    let weights_l1 = ts.matrix(HIDDEN_L2, HIDDEN_L1)?;
    let activations_l1 = ts.matrix(HIDDEN_L2, 1)?;
    let grad_l1 = ts.matrix(HIDDEN_L2, HIDDEN_L1)?;
    let error_l1 = ts.matrix(HIDDEN_L2, 1)?;

    let weights_l2 = ts.matrix(OUTPUT_SIZE, HIDDEN_L2)?;
    let output_layer = ts.matrix(OUTPUT_SIZE, 1)?;
    let grad_l2 = ts.matrix(OUTPUT_SIZE, HIDDEN_L2)?;
    let output_error_layer = ts.matrix(OUTPUT_SIZE, 1)?;

    // Weight initialization
    let mut rng = rand::thread_rng();
    random_weights(weights_l0, HIDDEN_L1 * IMG_SIZE, &mut ts, &mut rng)?;
    random_weights(weights_l1, HIDDEN_L2 * HIDDEN_L1, &mut ts, &mut rng)?;
    random_weights(weights_l2, OUTPUT_SIZE * HIDDEN_L2, &mut ts, &mut rng)?;

    let forward_pass = vec![
        // The boof
        Operation::MatrixMultiply {
            left: weights_l0,
            right: input_layer,
            dst: activations_l0,
            left_transpose: false,
            right_transpose: false,
        },
        Operation::Sigmoid(activations_l0),
        Operation::MatrixMultiply {
            left: weights_l1,
            right: activations_l0,
            dst: activations_l1,
            left_transpose: false,
            right_transpose: false,
        },
        Operation::Sigmoid(activations_l1),
        Operation::MatrixMultiply {
            left: weights_l2,
            right: activations_l1,
            dst: output_layer,
            left_transpose: false,
            right_transpose: false,
        },
    ];
    let forward_pass = ts.create_pass(&forward_pass)?;

    let learning_rate = 0.05;

    let backward_pass = vec![
        // The reverse boof
        Operation::MatrixMultiply {
            // Compute gradient for weights in layer 2 (outer product)
            left: output_error_layer, //
            right: activations_l1,
            dst: grad_l2,
            left_transpose: false,
            right_transpose: true,
        },
        Operation::MatrixMultiply {
            // Compute errors for next layer
            left: weights_l2,
            right: output_error_layer,
            dst: error_l1,
            left_transpose: true,
            right_transpose: false,
        },
        Operation::ScalarMultiply(grad_l2, learning_rate), // Gradient update for layer 2
        Operation::InplaceSub(weights_l2, grad_l2),
        Operation::SigmoidDerivative(activations_l1), // More error propagation
        Operation::InplaceMultiply(error_l1, activations_l1),
        //
        Operation::MatrixMultiply {
            // Compute gradient for weights in layer 1 (outer product)
            left: error_l1, //
            right: activations_l0,
            dst: grad_l1,
            left_transpose: false,
            right_transpose: true,
        },
        Operation::MatrixMultiply {
            // Compute errors for next layer
            left: weights_l1,
            right: error_l1,
            dst: error_l0,
            left_transpose: true,
            right_transpose: false,
        },
        Operation::ScalarMultiply(grad_l1, learning_rate), // Gradient update for layer 2
        Operation::InplaceSub(weights_l1, grad_l1),
        Operation::SigmoidDerivative(activations_l0), // More error propagation
        Operation::InplaceMultiply(error_l0, activations_l0),
        //
        Operation::MatrixMultiply {
            // Compute gradient for weights in layer 1 (outer product)
            left: error_l0, //
            right: input_layer,
            dst: grad_l0,
            left_transpose: false,
            right_transpose: true,
        },
        Operation::ScalarMultiply(grad_l0, learning_rate), // Gradient update for layer 2
        Operation::InplaceSub(weights_l0, grad_l0),
    ];
    let backward_pass = ts.create_pass(&backward_pass)?;

    // Intermediate, re-used buffers
    let mut input_buf = vec![0.; IMG_SIZE];
    let mut output_buf = vec![0.; OUTPUT_SIZE];

    // Training loop
    let mut num_correct = 0;
    let mut num_total = 0;
    for (idx, (label, img)) in mnist
        .trn_lbl
        .iter()
        .zip(mnist.trn_img.chunks_exact(IMG_SIZE))
        .enumerate()
    {
        // Feed forward
        image_norm(img, &mut input_buf);
        ts.write(input_layer, &input_buf)?;
        ts.flow(forward_pass)?;
        ts.read(output_layer, &mut output_buf)?;

        if argmax(&output_buf) == *label as usize {
            num_correct += 1;
        }
        num_total += 1;

        if idx % 100 == 0 {
            println!("Accuracy: {}", num_correct as f32 / num_total as f32);
            num_correct = 0;
            num_total = 0;
        }

        // Difference with train val
        output_buf[*label as usize] -= 1.;

        // Softmax before backprop step
        softmax(&mut output_buf);

        // Write to output error for backprop
        ts.write(output_error_layer, &output_buf)?;
        ts.flow(backward_pass)?;
    }

    println!("Computing accuracy...");
    let mut num_correct = 0;
    let mut num_total = 0;
    for (label, img) in mnist
        .tst_lbl
        .iter()
        .zip(mnist.tst_img.chunks_exact(IMG_SIZE))
    {
        image_norm(img, &mut input_buf);
        ts.write(input_layer, &input_buf)?;
        ts.flow(forward_pass)?;
        ts.read(output_layer, &mut output_buf)?;

        if argmax(&output_buf) == *label as usize {
            num_correct += 1;
        }
        num_total += 1;
    }

    println!("Accuracy: {}", num_correct as f32 / num_total as f32);

    Ok(())
}

fn image_norm(image: &[u8], out: &mut [f32]) {
    out.iter_mut()
        .zip(image.iter().map(|&v| v as f32 / 255.))
        .for_each(|(o, i)| *o = i);
}

fn argmax(output: &[f32]) -> usize {
    let mut max = 0.;
    let mut max_idx = 0;
    for (idx, &entry) in output.iter().enumerate() {
        if entry > max {
            max_idx = idx;
            max = entry;
        }
    }
    max_idx
}
