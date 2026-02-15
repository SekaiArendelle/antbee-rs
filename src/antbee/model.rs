use super::dataset::Data;
use super::dataset::Dataset;
use super::kind::Kind;
use ndarray::Array1;
use rand::random;

/// A binary classification model using logistic regression with sigmoid activation.
///
/// This model performs binary classification (Ant vs Bee) on 28x28 RGB images
/// (3 channels * 28 * 28 = 2352 input features) using a single-layer neural network
/// with sigmoid activation and cross-entropy loss.
pub struct Model {
    /// Weight vector of shape (INPUT_DIM,).
    /// Stores the learned parameters for each input feature.
    w: Array1<f32>,
    /// Bias term (intercept).
    /// Allows the decision boundary to shift from the origin.
    b: f32,
}

impl Model {
    /// Learning rate for gradient descent optimization.
    /// Controls the step size during weight updates.
    const LEARNING_RATE: f32 = 0.001;

    /// Input dimensionality.
    /// 3 channels (RGB) * 28 pixels * 28 pixels = 2352 features.
    const INPUT_DIM: usize = 2352;

    /// Creates a new `Model` with Xavier/He-inspired weight initialization.
    ///
    /// Weights are initialized uniformly in the range [-scale, scale] where
    /// scale = sqrt(2.0 / INPUT_DIM). This helps prevent vanishing/exploding
    /// gradients in early training stages.
    ///
    /// # Returns
    /// A new `Model` instance with initialized weights and zero bias.
    pub fn new() -> Self {
        let scale = (2.0 / Self::INPUT_DIM as f32).sqrt();
        return Self {
            w: Array1::from_shape_fn(Self::INPUT_DIM, |_| (random::<f32>() - 0.5) * 2.0 * scale),
            b: 0.0,
        };
    }

    /// Sigmoid activation function.
    ///
    /// Maps any real-valued number to the range (0, 1), which can be
    /// interpreted as a probability.
    ///
    /// # Arguments
    /// * `z` - The input value (logit).
    ///
    /// # Returns
    /// The sigmoid of `z`: 1 / (1 + exp(-z))
    fn sigmoid(z: f32) -> f32 {
        return 1.0 / (1.0 + (-z).exp());
    }

    /// Computes the probability that the input belongs to class `Bee`.
    ///
    /// Performs forward propagation: z = w·x + b, then applies sigmoid.
    ///
    /// # Arguments
    /// * `x` - Input feature vector of shape (INPUT_DIM,).
    ///
    /// # Returns
    /// A value in (0, 1) representing P(class = Bee | x).
    fn predict_prob(&self, x: &Array1<f32>) -> f32 {
        let z = self.w.dot(x) + self.b;
        return Self::sigmoid(z);
    }

    /// Predicts the class label for the given input.
    ///
    /// Uses a threshold of 0.5 on the predicted probability.
    ///
    /// # Arguments
    /// * `x` - Input feature vector.
    ///
    /// # Returns
    /// * `Kind::Bee` if P(Bee) > 0.5
    /// * `Kind::Ant` otherwise
    fn predict(&self, x: &Array1<f32>) -> Kind {
        if self.predict_prob(x) > 0.5 {
            return Kind::Bee;
        } else {
            return Kind::Ant;
        }
    }

    /// Computes the binary cross-entropy loss.
    ///
    /// For numerical stability, predictions are clamped to [EPS, 1-EPS]
    /// to avoid log(0) which would result in infinity.
    ///
    /// # Arguments
    /// * `y_pred` - Predicted probability (output of sigmoid).
    /// * `y_true` - Ground truth label.
    ///
    /// # Returns
    /// The cross-entropy loss value.
    fn cross_entropy_loss(y_pred: f32, y_true: Kind) -> f32 {
        const EPS: f32 = 1e-7;
        return match y_true {
            Kind::Ant => -(1.0 - y_pred.clamp(EPS, 1.0 - EPS)).ln(),
            Kind::Bee => -y_pred.clamp(EPS, 1.0 - EPS).ln(),
        };
    }

    /// Performs backward propagation and updates model parameters.
    ///
    /// Computes gradients of the loss with respect to weights and bias,
    /// then performs gradient descent update.
    ///
    /// # Mathematical Derivations
    /// - dL/dz = prob - y (where y is 0 for Ant, 1 for Bee)
    /// - dL/dw = x * dL/dz (chain rule)
    /// - dL/db = dL/dz
    ///
    /// # Arguments
    /// * `prob` - Predicted probability from forward pass.
    /// * `data` - Training data containing input features and label.
    fn backward(&mut self, prob: f32, data: &Data) {
        // Compute gradient of loss w.r.t. z (pre-activation)
        let dz = match data.get_kind() {
            Kind::Ant => prob,       // y = 0, so dz = prob - 0 = prob
            Kind::Bee => prob - 1.0, // y = 1, so dz = prob - 1
        };

        // Compute gradients w.r.t. parameters
        let dw = data.get_data() * dz; // dL/dw = x * dz
        let db = dz; // dL/db = dz

        // Gradient descent parameter update
        // w = w - learning_rate * dw
        // b = b - learning_rate * db
        self.w.scaled_add(-Self::LEARNING_RATE, &dw);
        self.b -= Self::LEARNING_RATE * db;
    }

    /// Performs one training step on a single data point.
    ///
    /// Executes forward propagation, computes loss, and performs
    /// backward propagation with parameter update.
    ///
    /// # Arguments
    /// * `data` - A single training example.
    ///
    /// # Returns
    /// The computed loss value for this training step.
    pub fn train_step(&mut self, data: &Data) -> f32 {
        let prob = self.predict_prob(data.get_data()); // Forward pass
        let loss = Self::cross_entropy_loss(prob, data.get_kind());
        self.backward(prob, data); // Backward pass and update

        return loss;
    }

    /// Evaluates the model accuracy on a given dataset.
    ///
    /// Compares predicted labels against ground truth labels.
    ///
    /// # Arguments
    /// * `dataset` - The dataset to evaluate on.
    ///
    /// # Returns
    /// Accuracy as a float in range [0.0, 1.0].
    pub fn evaluate(&self, dataset: &Dataset) -> f32 {
        let mut correct = 0;
        for data in dataset.get_values() {
            let pred = self.predict(data.get_data());
            // Compare discriminant to check if variants match
            if pred == data.get_kind() {
                correct += 1;
            }
        }
        return correct as f32 / dataset.len() as f32;
    }
}
