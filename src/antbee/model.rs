use super::dataset::Data;
use super::dataset::Dataset;
use super::kind::Kind;
use ndarray::Array1;
use rand::random;

pub struct Model {
    // z = w * x + b
    w: Array1<f32>, // (2352,)
    b: f32,
}

impl Model {
    const LEARNING_RATE: f32 = 0.001; // 图像数据用更小学习率
    const INPUT_DIM: usize = 2352; // 3 * 28 * 28

    pub fn new() -> Self {
        // Xavier 初始化
        let scale = (2.0 / Self::INPUT_DIM as f32).sqrt();
        return Self {
            w: Array1::from_shape_fn(Self::INPUT_DIM, |_| (random::<f32>() - 0.5) * 2.0 * scale),
            b: 0.0,
        };
    }

    fn sigmoid(z: f32) -> f32 {
        return 1.0 / (1.0 + (-z).exp());
    }

    /// 预测为 bee 的概率
    fn predict_prob(&self, x: &Array1<f32>) -> f32 {
        let z = self.w.dot(x) + self.b;
        return Self::sigmoid(z);
    }

    /// 预测类别
    fn predict(&self, x: &Array1<f32>) -> Kind {
        if self.predict_prob(x) > 0.5 {
            return Kind::Bee;
        } else {
            return Kind::Ant;
        }
    }

    /// 单步训练，返回 loss
    pub fn train_step(&mut self, data: &Data) -> f32 {
        let x = data.get_data();
        let y = match data.get_kind() {
            Kind::Ant => 0.0,
            Kind::Bee => 1.0,
        };

        // 前向
        let prob = self.predict_prob(&x);

        // 计算 loss (二元交叉熵)
        const EPS: f32 = 1e-7;
        let loss = -(y * (prob + EPS).ln() + (1.0 - y) * (1.0 - prob + EPS).ln());

        // 反向传播
        let dz = prob - y; // dL/dz
        let dw = x * dz; // dL/dw = x * dz
        let db = dz; // dL/db = dz

        // 更新
        self.w.scaled_add(-Self::LEARNING_RATE, &dw);
        self.b -= Self::LEARNING_RATE * db;

        return loss;
    }

    /// 计算准确率
    pub fn evaluate(&self, dataset: &Dataset) -> f32 {
        let mut correct = 0;
        for data in dataset.get_values() {
            let pred = self.predict(data.get_data());
            if std::mem::discriminant(&pred) == std::mem::discriminant(data.get_kind()) {
                correct += 1;
            }
        }
        return correct as f32 / dataset.len() as f32;
    }
}
