use crate::*;

#[test]
fn sigmoid() -> Result<()> {
    let mut ts = TensorSludge::new()?;
    let matrix = ts.matrix(3, 3)?;

    let pass = ts.create_pass(&[Operation::Sigmoid(matrix)])?;

    let data = (1..=9).map(|v| v as f32).into_iter().collect::<Vec<_>>();

    ts.write(matrix, &data)?;

    ts.flow(pass)?;

    let mut output = [0.; 3 * 3];
    ts.read(matrix, &mut output)?;

    assert!(data
        .iter()
        .map(|v| 1. / (1. + (-v).exp()))
        .zip(output.iter())
        .all(|(a, &b)| (a - b).abs() < std::f32::EPSILON));

    Ok(())
}
