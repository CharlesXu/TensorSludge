use anyhow::Result;
use tensorsludge::*;

fn main() -> Result<()> {
    let mut ts = TensorSludge::new()?;

    let a = ts.matrix(3, 3, 1, "A")?;
    let b = ts.matrix(3, 1, 1, "B")?;
    let dst = ts.matrix(3, 1, 1, "Output")?;

    let pass = ts.create_pass(&[
        Operation::MatrixMultiply {
            left: a,
            right: b,
            left_transpose: true,
            right_transpose: false,
            dst,
        },
        //Operation::Sigmoid(dst),
    ])?;

    ts.write(
        a,
        &[
            1., 2., 3., //
            4., 5., 6., //
            7., 8., 9., //
        ],
    )?;

    ts.write(
        b,
        &[
            0.1, //
            0.2, //
            0.3, //
        ],
    )?;

    ts.flow(pass)?;

    let mut output = [0.; 3];
    ts.read(dst, &mut output)?;
    dbg!(output);

    Ok(())
}
