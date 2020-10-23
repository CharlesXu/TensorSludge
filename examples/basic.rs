use anyhow::Result;
use tensorsludge::*;

fn main() -> Result<()> {
    let mut ts = TensorSludge::new()?;

    let a = ts.matrix(3, 3)?;
    let b = ts.matrix(3, 1)?;
    let dst = ts.matrix(3, 1)?;

    let pass = ts.create_pass(&[
        Operation::MatrixMultiply {
            left: a,
            right: b,
            left_transpose: false,
            right_transpose: true,
            dst,
        },
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
            10., //
            11., //
            12., //
        ],
    )?;

    ts.flow(pass)?;

    let mut output = [0.; 3];
    ts.read(dst, &mut output)?;
    dbg!(output);


    Ok(())
}
