use crate::*;

#[cfg(test)]
#[test]
fn sigmoid() -> Result<()> {
    let mut ts = TensorSludge::new()?;
    const ROWS: usize = 300;
    const COLS: usize = 300;

    let matrix = ts.matrix(ROWS, COLS)?;

    let pass = ts.create_pass(&[Operation::Sigmoid(matrix)])?;

    let data = (1..=ROWS * COLS).map(|v| v as f32).into_iter().collect::<Vec<_>>();

    ts.write(matrix, &data)?;

    ts.flow(pass)?;

    let mut output = [0.; ROWS * COLS];
    ts.read(matrix, &mut output)?;

    assert!(data
        .iter()
        .map(|v| 1. / (1. + (-v).exp()))
        .zip(output.iter())
        .all(|(a, &b)| (a - b).abs() < std::f32::EPSILON));

    Ok(())
}


#[cfg(test)]
#[test]
fn matrix_multiply() -> Result<()> {
    let mut ts = TensorSludge::new()?;
    const ROWS: usize = 300;
    const INNER: usize = 300;
    const COLS: usize = 200;

    let a = ts.matrix(ROWS, INNER)?;
    let b = ts.matrix(INNER, COLS)?;
    let output = ts.matrix(ROWS, COLS)?;

    let pass = ts.create_pass(&[Operation::MatrixMultiply {
        left: a,
        right: b,
        dst: output,
        left_transpose: false,
        right_transpose: false,
    }])?;

    // Identity matrix
    let mut data = vec![0.; ROWS * INNER];
    for row in 0..INNER {
        data[row * INNER + row] = 1.;
    }
    ts.write(a, &data)?;

    let data = (1..=ROWS * COLS).map(|v| v as f32).into_iter().collect::<Vec<_>>();
    ts.write(a, &data)?;

    ts.flow(pass)?;

    let mut out_vec = vec![0.; ROWS * COLS];
    ts.read(output, &mut out_vec)?;

    assert!(data
        .iter()
        .zip(out_vec.iter())
        .all(|(a, &b)| (a - b).abs() < std::f32::EPSILON));

    Ok(())
}
