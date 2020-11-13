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
    let tmp = ts.matrix(size, 1, 1, true, "rand_tmp")?;
    let unif = Uniform::new(-1., 1.);
    let buf = unif.sample_iter(rng).take(size).collect::<Vec<f32>>();
    ts.write(tmp, &buf)?;
    ts.transfer(tmp, mat)?;
    ts.remove_matrix(tmp)?;
    Ok(())
}

fn softmax(data: &mut [f32]) {
    let sum = data.iter().map(|v| v.exp()).sum::<f32>();
    data.iter_mut().for_each(|v| *v /= sum);
}

/*
fn mse(input: &[f32]) -> f32 {
    input.iter().map(|&v| v * v).sum::<f32>() / input.len() as f32
}
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
    const BATCH_SIZE: usize = 10;
    const LEARNING_RATE: f32 = 0.01;
    const EPOCHS: usize = 2;

    // Build weight and activation buffers
    let input_layer = ts.matrix(IMG_SIZE, 1, BATCH_SIZE, true, "input_layer")?;

    let weights_l0 = ts.matrix(HIDDEN_L1, IMG_SIZE, 1, false, "weights_l0")?;
    let activations_l0 = ts.matrix(HIDDEN_L1, 1, BATCH_SIZE, false, "activations_l0")?;
    let grad_l0 = ts.matrix(HIDDEN_L1, IMG_SIZE, BATCH_SIZE, false, "grad_l0")?;
    let error_l0 = ts.matrix(HIDDEN_L1, 1, BATCH_SIZE, false, "error_l0")?;

    let weights_l1 = ts.matrix(HIDDEN_L2, HIDDEN_L1, 1, false, "weights_l1")?;
    let activations_l1 = ts.matrix(HIDDEN_L2, 1, BATCH_SIZE, false, "activations_l1")?;
    let grad_l1 = ts.matrix(HIDDEN_L2, HIDDEN_L1, BATCH_SIZE, false, "grad_l1")?;
    let error_l1 = ts.matrix(HIDDEN_L2, 1, BATCH_SIZE, false, "error_l1")?;

    let weights_l2 = ts.matrix(OUTPUT_SIZE, HIDDEN_L2, 1, false, "weights_l2")?;
    let grad_l2 = ts.matrix(OUTPUT_SIZE, HIDDEN_L2, BATCH_SIZE, false, "grad_l2")?;
    let output_error_layer = ts.matrix(OUTPUT_SIZE, 1, BATCH_SIZE, true, "output_error_layer")?;
    let output_layer = ts.matrix(OUTPUT_SIZE, 1, BATCH_SIZE, true, "output_layer")?;

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
        Operation::ScalarMultiply(grad_l2, LEARNING_RATE), // Gradient update for layer 2
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
        Operation::ScalarMultiply(grad_l1, LEARNING_RATE), // Gradient update for layer 2
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
        Operation::ScalarMultiply(grad_l0, LEARNING_RATE), // Gradient update for layer 2
        Operation::InplaceSub(weights_l0, grad_l0),
    ];
    let backward_pass = ts.create_pass(&backward_pass)?;

    // Intermediate, re-used buffers
    let mut input_buf = vec![0.; IMG_SIZE * BATCH_SIZE];
    let mut output_buf = vec![0.; OUTPUT_SIZE * BATCH_SIZE];

    // Training loop
    for _ in 0..EPOCHS {
        let mut num_correct = 0;
        let mut num_total = 0;
        for (idx, (labels, img)) in mnist
            .trn_lbl
                .chunks_exact(BATCH_SIZE)
                .zip(mnist.trn_img.chunks_exact(IMG_SIZE * BATCH_SIZE))
                .enumerate()
                {
                    // Feed forward
                    image_norm(img, &mut input_buf);
                    ts.write(input_layer, &input_buf)?;
                    ts.flow(forward_pass)?;
                    ts.read(output_layer, &mut output_buf)?;

                    for (outputs, label) in output_buf.chunks_exact_mut(OUTPUT_SIZE).zip(labels) {
                        if argmax(&outputs) == *label as usize {
                            num_correct += 1;
                        }
                        num_total += 1;

                        // Difference with train val
                        outputs[*label as usize] -= 1.;

                        // Softmax before backprop step
                        softmax(outputs);
                    }

                    if idx % 100 == 0 {
                        println!("Accuracy: {}", num_correct as f32 / num_total as f32);
                        num_correct = 0;
                        num_total = 0;
                    }

                    // Write to output error for backprop
                    ts.write(output_error_layer, &output_buf)?;
                    ts.flow(backward_pass)?;
                }
    }

    println!("Computing accuracy...");
    let mut num_correct = 0;
    let mut num_total = 0;
    for (labels, img) in mnist
        .tst_lbl
        .chunks_exact(BATCH_SIZE)
        .zip(mnist.tst_img.chunks_exact(IMG_SIZE * BATCH_SIZE))
    {
        image_norm(img, &mut input_buf);
        ts.write(input_layer, &input_buf)?;
        ts.flow(forward_pass)?;
        ts.read(output_layer, &mut output_buf)?;

        for (outputs, label) in output_buf.chunks_exact_mut(OUTPUT_SIZE).zip(labels) {
            if argmax(&outputs) == *label as usize {
                num_correct += 1;
            }
            num_total += 1;
        }
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
