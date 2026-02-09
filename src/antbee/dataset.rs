use image::ImageReader;
use image::imageops::FilterType;
use image::imageops::resize;
use ndarray::Array3;
use std::fs::read_dir;
use std::path::Path;

pub struct Dataset {
    ants: Vec<Array3<f32>>,
    bees: Vec<Array3<f32>>,
}

impl Dataset {
    fn jpg_to_chw(path: &Path) -> Array3<f32> {
        let rgb = ImageReader::open(path).unwrap().decode().unwrap().to_rgb8();
        let resized = resize(&rgb, 28, 28, FilterType::Lanczos3);

        let mut data = Vec::<f32>::with_capacity(28 * 28 * 3);

        for (index, pixel) in resized.pixels().enumerate() {
            data[index] = pixel[0] as f32 / 255.0;
            data[index + 784] = pixel[1] as f32 / 255.0;
            data[index + 2 * 784] = pixel[2] as f32 / 255.0;
        }

        return Array3::from_shape_vec((3, 28, 28), data).unwrap();
    }

    fn check_is_valid_dir(path: &Path) {
        if !path.exists() {
            panic!("Dataset path does not exist");
        }
        if !path.is_dir() {
            panic!("Dataset path is not a directory");
        }
    }

    pub fn from_dataset_path(paths: &Path) -> Self {
        #[cfg(debug_assertions)]
        Self::check_is_valid_dir(paths);
        let ants_dir = paths.join("ants");
        #[cfg(debug_assertions)]
        Self::check_is_valid_dir(&ants_dir);
        let bees_dir = paths.join("bees");
        #[cfg(debug_assertions)]
        Self::check_is_valid_dir(&bees_dir);

        let mut ants: Vec<Array3<f32>> = Vec::new();
        let mut bees: Vec<Array3<f32>> = Vec::new();

        for ant_img_path in read_dir(ants_dir).unwrap() {
            let origin_img = Self::jpg_to_chw(ant_img_path.unwrap().path().as_path());
            ants.push(origin_img);
        }
        for bee_img_path in read_dir(bees_dir).unwrap() {
            let origin_img = Self::jpg_to_chw(bee_img_path.unwrap().path().as_path());
            bees.push(origin_img);
        }
        return Self { ants, bees };
    }

    pub fn get_ants(&self) -> &Vec<Array3<f32>> {
        return &self.ants;
    }

    pub fn get_bees(&self) -> &Vec<Array3<f32>> {
        return &self.bees;
    }
}
