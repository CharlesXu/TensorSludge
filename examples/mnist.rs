use mnist::MnistBuilder;
use tensorsludge::*;

fn main() {
    let mnist = MnistBuilder::new().download_and_extract().finalize();
}
