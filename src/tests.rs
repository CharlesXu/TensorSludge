use crate::*;
use anyhow::Result;
use std::f32::EPSILON;

#[test]
fn sigmoid() -> Result<()> {
    let mut ts = TensorSludge::new()?;
    const ROWS: usize = 300;
    const COLS: usize = 300;

    let matrix = ts.matrix(ROWS, COLS, 1, "0")?;

    let pass = ts.create_pass(&[Operation::Sigmoid(matrix)])?;

    let data = (1..=ROWS * COLS)
        .map(|v| v as f32)
        .into_iter()
        .collect::<Vec<_>>();

    ts.write(matrix, &data)?;

    ts.flow(pass)?;

    let mut output = [0.; ROWS * COLS];
    ts.read(matrix, &mut output)?;

    assert!(data
        .iter()
        .map(|v| 1. / (1. + (-v).exp()))
        .zip(output.iter())
        .all(|(a, &b)| (a - b).abs() < EPSILON));

    Ok(())
}

#[test]
fn sigmoid_deriv() -> Result<()> {
    let mut ts = TensorSludge::new()?;
    const ROWS: usize = 300;
    const COLS: usize = 300;

    let matrix = ts.matrix(ROWS, COLS, 1, "0")?;

    let pass = ts.create_pass(&[Operation::SigmoidDerivative(matrix)])?;

    let data = (1..=ROWS * COLS)
        .map(|v| v as f32)
        .into_iter()
        .collect::<Vec<_>>();

    ts.write(matrix, &data)?;

    ts.flow(pass)?;

    let mut output = [0.; ROWS * COLS];
    ts.read(matrix, &mut output)?;

    assert!(data
        .iter()
        .map(|v| v * (1. - v))
        .zip(output.iter())
        .all(|(a, &b)| (a - b).abs() < EPSILON));

    Ok(())
}

#[test]
fn matrix_multiply() -> Result<()> {
    let mut ts = TensorSludge::new()?;
    const IDENT_SIZE: usize = 3;
    const COLS: usize = 2;

    let identity = ts.matrix(IDENT_SIZE, IDENT_SIZE, 1, "Identity")?;
    let b = ts.matrix(IDENT_SIZE, COLS, 1, "B")?;
    let output = ts.matrix(IDENT_SIZE, COLS, 1, "Output")?;

    let pass = ts.create_pass(&[Operation::MatrixMultiply {
        left: identity,
        right: b,
        dst: output,
        left_transpose: false,
        right_transpose: false,
    }])?;

    // Identity matrix
    let mut data = vec![0.; IDENT_SIZE * IDENT_SIZE];
    for row in 0..IDENT_SIZE {
        data[row * IDENT_SIZE + row] = 1.;
    }
    ts.write(identity, &data)?;

    let data = (1..=IDENT_SIZE * COLS)
        .map(|v| v as f32)
        .into_iter()
        .collect::<Vec<_>>();
    ts.write(b, &data)?;

    ts.flow(pass)?;

    let mut out_vec = vec![0.; IDENT_SIZE * COLS];
    ts.read(output, &mut out_vec)?;

    assert!(data
        .iter()
        .zip(out_vec.iter())
        .all(|(a, &b)| (a - b).abs() < EPSILON));

    Ok(())
}

#[test]
fn matrix_multiply_transposes() -> Result<()> {
    let mut ts = TensorSludge::new()?;

    let a = ts.matrix(3, 3, 1, "A")?;
    let b = ts.matrix(3, 3, 1, "B")?;
    let output = ts.matrix(3, 3, 1, "Output")?;

    let none = ts.create_pass(&[Operation::MatrixMultiply {
        left: a,
        right: b,
        dst: output,
        left_transpose: false,
        right_transpose: false,
    }])?;

    let a_t = ts.create_pass(&[Operation::MatrixMultiply {
        left: a,
        right: b,
        dst: output,
        left_transpose: true,
        right_transpose: false,
    }])?;

    let b_t = ts.create_pass(&[Operation::MatrixMultiply {
        left: a,
        right: b,
        dst: output,
        left_transpose: false,
        right_transpose: true,
    }])?;

    let a_t_b_t = ts.create_pass(&[Operation::MatrixMultiply {
        left: a,
        right: b,
        dst: output,
        left_transpose: true,
        right_transpose: true,
    }])?;

    let reset_mats = |ts: &mut TensorSludge| -> Result<()> {
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
                10., 11., 12., //
                13., 14., 15., //
                16., 17., 18., //
            ],
        )
    };

    // Identity matrix
    let mut output_data = [0.; 3 * 3];

    // Normal
    let expected = [84., 90., 96., 201., 216., 231., 318., 342., 366.];
    reset_mats(&mut ts)?;
    ts.flow(none)?;
    ts.read(output, &mut output_data)?;
    assert_eq!(output_data, expected);

    // A transpose
    let expected = [174., 186., 198., 213., 228., 243., 252., 270., 288.];
    reset_mats(&mut ts)?;
    ts.flow(a_t)?;
    ts.read(output, &mut output_data)?;
    assert_eq!(output_data, expected);

    // B transpose
    let expected = [68., 86., 104., 167., 212., 257., 266., 338., 410.];
    reset_mats(&mut ts)?;
    ts.flow(b_t)?;
    ts.read(output, &mut output_data)?;
    assert_eq!(output_data, expected);

    // Both transpose
    let expected = [138., 174., 210., 171., 216., 261., 204., 258., 312.];
    reset_mats(&mut ts)?;
    ts.flow(a_t_b_t)?;
    ts.read(output, &mut output_data)?;
    assert_eq!(output_data, expected);

    Ok(())
}

#[test]
fn elementwise_ops() -> Result<()> {
    let mut ts = TensorSludge::new()?;
    const ROWS: usize = 300;
    const COLS: usize = 100;

    let a = ts.matrix(ROWS, COLS, 1, "A")?;
    let b = ts.matrix(ROWS, COLS, 1, "B")?;

    let pass = ts.create_pass(&[
        Operation::InplaceAdd(a, b),
        Operation::InplaceMultiply(a, b),
        Operation::InplaceSub(a, b),
    ])?;

    let a_data = (1..=ROWS * COLS)
        .map(|v| v as f32)
        .into_iter()
        .collect::<Vec<_>>();
    ts.write(a, &a_data)?;

    let b_data = (1..=ROWS * COLS)
        .rev()
        .map(|v| (v * 3) as f32)
        .into_iter()
        .collect::<Vec<_>>();
    ts.write(b, &b_data)?;

    ts.flow(pass)?;

    let mut output = [0.; ROWS * COLS];
    ts.read(a, &mut output)?;

    assert!(a_data
        .iter()
        .zip(b_data.iter())
        .map(|(a, b)| (a + b) * b - b)
        .zip(output.iter())
        .all(|(a, &b)| (a - b).abs() < EPSILON));

    Ok(())
}

#[test]
fn scalar_mul() -> Result<()> {
    let mut ts = TensorSludge::new()?;
    const ROWS: usize = 300;
    const COLS: usize = 300;
    const SCALAR: f32 = 1.5324;

    let matrix = ts.matrix(ROWS, COLS, 1, "0")?;

    let pass = ts.create_pass(&[Operation::ScalarMultiply(matrix, SCALAR)])?;

    let data = (1..=ROWS * COLS)
        .map(|v| v as f32)
        .into_iter()
        .collect::<Vec<_>>();

    ts.write(matrix, &data)?;

    ts.flow(pass)?;

    let mut output = [0.; ROWS * COLS];
    ts.read(matrix, &mut output)?;

    assert!(data
        .iter()
        .map(|v| v * SCALAR)
        .zip(output.iter())
        .all(|(a, &b)| (a - b).abs() < EPSILON));

    Ok(())
}

#[test]
fn matrix_layers() -> Result<()> {
    let mut ts = TensorSludge::new()?;

    let a = ts.matrix(3, 3, 2, "A")?;
    let b = ts.matrix(3, 3, 2, "B")?;
    let output = ts.matrix(3, 3, 2, "Output")?;

    ts.write(
        a,
        &[
        1., 2., 3., //
        4., 5., 6., //
        7., 8., 9., //
        //
        19., 20., 21., //
        22., 23., 24., //
        25., 26., 27., //
        ],
    )?;

    ts.write(
        b,
        &[
        10., 11., 12., //
        13., 14., 15., //
        16., 17., 18., //
        //
        28., 29., 30., //
        31., 32., 33., //
        34., 35., 36., //
        ],
    )?;

    let pass = ts.create_pass(&[Operation::MatrixMultiply {
        left: a,
        right: b,
        dst: output,
        left_transpose: false,
        right_transpose: false,
    }])?;

    ts.flow(pass)?;
    let mut output_data = [0.; 3 * 3 * 2];
    ts.read(output, &mut output_data)?;

    let expected = [
        84., 90., 96., //
        201., 216., 231., //
        318., 342., 366., //
        //
        1866., 1926., 1986., //
        2145., 2214., 2283., //
        2424., 2502., 2580., //
    ];
    assert_eq!(output_data, expected);

    Ok(())
}
