use std::path::PathBuf;
mod antbee;
use antbee::Dataset;
use antbee::Model;

fn train_model(model: &mut Model, dataset: &Dataset) {
    const EPOCHS: usize = 150;
    let n = dataset.len() as f32;

    for epoch in 0..EPOCHS {
        let mut total_loss = 0.0;

        for data in dataset.get_values() {
            total_loss += model.train_step(data);
        }

        if epoch % 10 == 0 {
            let avg_loss = total_loss / n;
            let accuracy = model.evaluate(dataset);
            println!(
                "Epoch {:3}: loss={:.4}, acc={:.2}%",
                epoch,
                avg_loss,
                accuracy * 100.0
            );
        }
    }
}

fn test_model(model: &Model, dataset: &Dataset) {
    let accuracy = model.evaluate(dataset);
    println!("Test Accuracy: {:.2}%", accuracy * 100.0);
}

fn main() {
    let dataset_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("dataset");

    println!("loading train dataset");
    let train_dataset = antbee::Dataset::from_dataset_path(&dataset_dir.join("train"));

    println!("starting training");
    let mut model = antbee::Model::new();
    train_model(&mut model, &train_dataset);

    println!("loading test dataset");
    let test_dataset = antbee::Dataset::from_dataset_path(&dataset_dir.join("val"));

    println!("starting testing");
    test_model(&model, &test_dataset);
}
