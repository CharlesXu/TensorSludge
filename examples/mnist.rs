use anyhow::Result;
use mnist::MnistBuilder;
use tensorsludge::*;
use rand::Rng;
use rand::distributions::{Distribution, Uniform};

// Output size is # of rows
fn random_weights(mat: Matrix, size: usize, ts: &mut TensorSludge, rng: &mut impl Rng) -> Result<()> {
    let unif = Uniform::new(-1., 1.);
    let buf = unif.sample_iter(rng).take(size).collect::<Vec<f32>>();
    ts.write(mat, &buf)
}

/* Wishlist:
 * Sigmoid deriv
 * Elementwise mul
 */

fn main() -> Result<()> {
    let mut ts = TensorSludge::new()?;
    let mnist = MnistBuilder::new().download_and_extract().finalize();

    const IMG_WIDTH: usize = 28;
    const IMG_SIZE: usize = IMG_WIDTH * IMG_WIDTH;
    const HIDDEN_L1: usize = 128;
    //const HIDDEN_L2: usize = 64;
    //const OUTPUT_SIZE: usize = 10;

    let input_layer = ts.matrix(IMG_SIZE, 1)?;
    let weights_l0 = ts.matrix(HIDDEN_L1, IMG_SIZE)?;
    let activations_l0 = ts.matrix(HIDDEN_L1, 1)?;

    let mut rng = rand::thread_rng();
    random_weights(weights_l0, HIDDEN_L1 * IMG_SIZE, &mut ts, &mut rng)?;

    let forward_pass = vec![
        Operation::MatrixMultiply {
            left: weights_l0,
            right: input_layer,
            dst: activations_l0,
            left_transpose: false,
            right_transpose: false,
        },
        Operation::Sigmoid(activations_l0),
    ];
    let forward_pass = ts.create_pass(&forward_pass)?;

    for (label, img) in mnist
        .trn_lbl
        .iter()
        .zip(mnist.trn_img.chunks_exact(IMG_SIZE))
    {
        let mut output = vec![0.; HIDDEN_L1];

        // Feed forward
        let img: Vec<f32> = img.iter().map(|&v| v as f32 / 255.).collect();
        ts.write(input_layer, &img)?;
        ts.flow(forward_pass)?;

        ts.read(activations_l0, &mut output)?;

        dbg!(&output);
    }

    Ok(())
}
