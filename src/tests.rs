use crate::*;
use anyhow::Result;
use std::f32::EPSILON;

#[test]
fn sigmoid() -> Result<()> {
    let mut ts = TensorSludge::new()?;
    const ROWS: usize = 300;
    const COLS: usize = 300;
    const LAYERS: usize = 2;

    let matrix = ts.matrix(ROWS, COLS, LAYERS, true, "0")?;

    let pass = ts.create_pass(&[Operation::Sigmoid(matrix)])?;

    let data = (1..=ROWS * COLS * LAYERS)
        .map(|v| v as f32)
        .into_iter()
        .collect::<Vec<_>>();

    ts.write(matrix, &data)?;

    ts.flow(pass)?;

    let mut output = [0.; ROWS * COLS * LAYERS];
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
    const ROWS: usize = 30;
    const COLS: usize = 30;
    const LAYERS: usize = 2;

    let matrix = ts.matrix(ROWS, COLS, LAYERS, true, "0")?;

    let pass = ts.create_pass(&[Operation::SigmoidDerivative(matrix)])?;

    let data = (1..=ROWS * COLS * LAYERS)
        .map(|v| v as f32)
        .into_iter()
        .collect::<Vec<_>>();

    ts.write(matrix, &data)?;

    ts.flow(pass)?;

    let mut output = [0.; ROWS * COLS * LAYERS];
    ts.read(matrix, &mut output)?;

    assert!(data
        .iter()
        .map(|v| v * (1. - v))
        .zip(output.iter())
        .all(|(a, &b)| dbg!(dbg!(a) - dbg!(b)).abs() < EPSILON));

    Ok(())
}

#[test]
fn matrix_multiply() -> Result<()> {
    let mut ts = TensorSludge::new()?;
    const IDENT_SIZE: usize = 3;
    const COLS: usize = 2;

    let identity = ts.matrix(IDENT_SIZE, IDENT_SIZE, 1, true, "Identity")?;
    let b = ts.matrix(IDENT_SIZE, COLS, 1, true, "B")?;
    let output = ts.matrix(IDENT_SIZE, COLS, 1, true, "Output")?;

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

    let a = ts.matrix(3, 3, 1, true, "A")?;
    let b = ts.matrix(3, 3, 1, true, "B")?;
    let output = ts.matrix(3, 3, 1, true, "Output")?;

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

    let a = ts.matrix(ROWS, COLS, 1, true, "A")?;
    let b = ts.matrix(ROWS, COLS, 1, true, "B")?;

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
fn elementwise_layers() -> Result<()> {
    let mut ts = TensorSludge::new()?;
    let a = ts.matrix(3, 3, 1, true, "A")?;
    let b = ts.matrix(3, 3, 2, true, "B")?;

    ts.write(a, &[ 8.,  1.,  3.,  1.,  9.,  7.,  1.,  3.,  0.])?;
    ts.write(b, &[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 
        10., 11., 12., 13., 14., 15., 16., 17., 18.])?;

    // One layer with two layers added to it
    let add = ts.create_pass(&[Operation::InplaceAdd(a, b)])?;

    ts.flow(add)?;

    let mut output = [0.; 9];
    ts.read(a, &mut output)?;

    assert_eq!(output, [19.0, 14.0, 18.0, 18.0, 28.0, 28.0, 24.0, 28.0, 27.0]);

    // Two layers with one layer added across both
    ts.write(a, &[ 8.,  1.,  3.,  1.,  9.,  7.,  1.,  3.,  0.])?;
    ts.write(b, &[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 
        10., 11., 12., 13., 14., 15., 16., 17., 18.])?;
    let add = ts.create_pass(&[Operation::InplaceAdd(b, a)])?;
    let mut output = [0.; 18];
    ts.flow(add)?;
    ts.read(b, &mut output)?;
    assert_eq!(output, [9.0, 3.0, 6.0, 5.0, 14.0, 13.0, 8.0, 11.0, 9.0, 18.0, 12.0, 15.0, 14.0, 23.0, 22.0, 17.0, 20.0, 18.0]);

    Ok(())
}

#[test]
fn scalar_mul() -> Result<()> {
    let mut ts = TensorSludge::new()?;
    const ROWS: usize = 300;
    const COLS: usize = 300;
    const LAYERS: usize = 3;
    const SCALAR: f32 = 1.5324;

    let matrix = ts.matrix(ROWS, COLS, LAYERS, true, "0")?;

    let pass = ts.create_pass(&[Operation::ScalarMultiply(matrix, SCALAR)])?;

    let data = (1..=ROWS * COLS * LAYERS)
        .map(|v| v as f32)
        .into_iter()
        .collect::<Vec<_>>();

    ts.write(matrix, &data)?;

    ts.flow(pass)?;

    let mut output = vec![0.; ROWS * COLS * LAYERS];
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

    let a = ts.matrix(3, 3, 2, true, "A")?;
    let b = ts.matrix(3, 3, 2, true, "B")?;
    let c = ts.matrix(3, 3, 1, true, "C")?;
    let output = ts.matrix(3, 3, 2, true, "Output")?;
    let output_single = ts.matrix(3, 3, 1, true, "Single output")?;

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

    ts.write(
        c,
        &[
        8., 7., 6., //
        5., 4., 3., //
        2., 1., 0., //
        ],
    )?;

    // Double layer test
    let pass_double = ts.create_pass(&[Operation::MatrixMultiply {
        left: a,
        right: b,
        dst: output,
        left_transpose: false,
        right_transpose: false,
    }])?;

    ts.flow(pass_double)?;
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

    // Single output layer test
    let pass_single = ts.create_pass(&[Operation::MatrixMultiply {
        left: a,
        right: b,
        dst: output_single,
        left_transpose: false,
        right_transpose: false,
    }])?;

    ts.flow(pass_single)?;
    let mut output_data = [0.; 3 * 3 * 1];
    ts.read(output_single, &mut output_data)?;

    let expected = [
        84., 90., 96., //
        201., 216., 231., //
        318., 342., 366., //
    ];
    assert_eq!(output_data, expected);

    // Double A, single B, double output
    let pass_hybrid = ts.create_pass(&[Operation::MatrixMultiply {
        left: a,
        right: c,
        dst: output,
        left_transpose: false,
        right_transpose: false,
    }])?;

    ts.flow(pass_hybrid)?;
    let mut output_data = [0.; 3 * 3 * 2];
    ts.read(output, &mut output_data)?;

    let expected = [
        24., 18., 12., //
        69., 54., 39., //
        114., 90., 66., //
        //
        294., 234., 174., //
        339., 270., 201., //
        384., 306., 228., //
    ];
    assert_eq!(output_data, expected);

    Ok(())
}

#[test]
fn transfer() -> Result<()> {
    let mut ts = TensorSludge::new()?;
    let a = ts.matrix(3, 3, 1, true, "A")?;
    let b = ts.matrix(3, 3, 1, true, "B")?;

    let a_content = &[
        1., 2., 3., //
        4., 5., 6., //
        7., 8., 9., //
        ];
    ts.write(
        a,
        a_content,
    )?;

    ts.write(
        b,
        &[
        10., 11., 12., //
        13., 14., 15., //
        16., 17., 18., //
        ],
    )?;

    ts.transfer(a, b)?;
    let mut buf = vec![0.; 3 * 3];
    ts.read(b, &mut buf)?;
    assert_eq!(&buf, a_content);

    Ok(())
}
