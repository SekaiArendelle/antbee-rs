use super::kind;
use image::ImageReader;
use image::imageops::FilterType;
use image::imageops::resize;
use ndarray::Array1;
use rand::prelude::SliceRandom;
use rand::rng;
use std::fs::read_dir;
use std::path::Path;

pub struct Data {
    kind: kind::Kind,
    data: Array1<f32>, // CHW flattened: 3*28*28 = 2352
}

impl Data {
    pub fn get_kind(&self) -> kind::Kind {
        return self.kind;
    }

    pub fn get_data(&self) -> &Array1<f32> {
        // data is already flattened
        return &self.data;
    }
}

pub struct Dataset {
    values: Vec<Data>,
}

impl Dataset {
    fn jpg_to_chw(path: &Path) -> Array1<f32> {
        let rgb = ImageReader::open(path).unwrap().decode().unwrap().to_rgb8();
        let resized = resize(&rgb, 28, 28, FilterType::Lanczos3);

        let mut data = Vec::<f32>::with_capacity(28 * 28 * 3);

        for pixel in resized.pixels() {
            data.push(pixel[0] as f32 / 255.0);
        }
        for pixel in resized.pixels() {
            data.push(pixel[1] as f32 / 255.0);
        }
        for pixel in resized.pixels() {
            data.push(pixel[2] as f32 / 255.0);
        }

        return Array1::from_vec(data);
    }

    #[cfg(debug_assertions)]
    fn assert_is_valid_dir(path: &Path) {
        debug_assert!(path.exists(), "Dataset path does not exist");
        debug_assert!(path.is_dir(), "Dataset path is not a directory");
    }

    pub fn from_dataset_path(paths: &Path) -> Self {
        #[cfg(debug_assertions)]
        Self::assert_is_valid_dir(paths);

        let ants_dir = paths.join("ants");
        let bees_dir = paths.join("bees");
        #[cfg(debug_assertions)]
        {
            Self::assert_is_valid_dir(&ants_dir);
            Self::assert_is_valid_dir(&bees_dir);
        }

        let mut values = Vec::<Data>::new();

        for ant_img_path in read_dir(ants_dir).unwrap() {
            let origin_img = Self::jpg_to_chw(&ant_img_path.unwrap().path());
            values.push(Data {
                kind: kind::Kind::Ant,
                data: origin_img,
            });
        }
        for bee_img_path in read_dir(bees_dir).unwrap() {
            let origin_img = Self::jpg_to_chw(&bee_img_path.unwrap().path());
            values.push(Data {
                kind: kind::Kind::Bee,
                data: origin_img,
            });
        }

        values.shuffle(&mut rng());
        return Self { values };
    }

    pub fn get_values(&self) -> &Vec<Data> {
        return &self.values;
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }
}
