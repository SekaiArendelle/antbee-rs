use super::dataset;
use super::kind;
use rand::random;
use ndarray::Array;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;

pub struct Model {
    w: Array2<f32>,
    b: Array1<f32>,
}

impl Model {
    const LEARNING_RATE: f32 = 0.5;

    pub fn new(n_features: usize, n_classes: usize) -> Self {
        return Self {
            w: Array::from_shape_fn((n_features, n_classes), |_| random()),
            b: Array1::zeros(n_classes),
        };
    }

    fn max_axis(input: &Array2<f32>, axis: Axis) -> Array1<f32> {
        let (m, n) = (input.nrows(), input.ncols());

        match axis {
            Axis(0) => {
                // column
                let mut result = Array1::zeros(n);
                for col in 0..n {
                    let mut max_val = f32::NEG_INFINITY;
                    for row in 0..m {
                        let val = input[[row, col]];
                        if val > max_val {
                            max_val = val;
                        }
                    }
                    result[col] = max_val;
                }
                return result;
            }
            Axis(1) => {
                // row
                let mut result = Array1::zeros(m);
                for row in 0..m {
                    let mut max_val = f32::NEG_INFINITY;
                    for col in 0..n {
                        let val = input[[row, col]];
                        if val > max_val {
                            max_val = val;
                        }
                    }
                    result[row] = max_val;
                }
                return result;
            }
            #[cfg(debug_assertions)]
            _ => unreachable!(),
            #[cfg(not(debug_assertions))]
            _ => unsafe {
                std::hint::unreachable_unchecked();
            },
        }
    }

    fn softmax(input: &Array2<f32>, axis: Axis) -> Array2<f32> {
        let max_vals = Self::max_axis(input, axis);
        let shifted = input - &max_vals.insert_axis(axis);

        let exp_vals = shifted.mapv(|x| x.exp());

        let sum_exp = exp_vals.sum_axis(axis).insert_axis(axis);
        return exp_vals / &sum_exp;
    }

    fn cross_entropy_loss(real: kind::Kind, probability_of_bee: f32) -> f32 {
        match real {
            kind::Kind::Ant => {
                return -((1.0 - probability_of_bee).ln());
            }
            kind::Kind::Bee => {
                return -(probability_of_bee.ln());
            }
        }
    }

    pub fn forward(&self, x: Array2<f32>) {
        let logits = self.w.dot(&x) + &self.b;
        let probs = Self::softmax(&logits, Axis(1));
    }

    pub fn train(&mut self, dataset: &dataset::Dataset) {
        for value in dataset.get_values() {
            let kind = value.get_kind();
            let data = value.get_data();
        }
    }
}
