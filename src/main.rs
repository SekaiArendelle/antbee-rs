use std::path::PathBuf;
mod antbee;

fn main() {
    let dataset_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("dataset");

    // train dataset
    let train = antbee::Dataset::from_dataset_path(&dataset_dir.join("train"));
    // test dataset
    let val = antbee::Dataset::from_dataset_path(&dataset_dir.join("val"));

    let mut model = antbee::Model::new(4, 3);
    model.train(&train);
}
