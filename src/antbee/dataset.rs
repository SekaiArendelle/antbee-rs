use image::ImageReader;
use image::RgbImage;
use std::fs::read_dir;
use std::path::Path;

pub enum Kind {
    Ant = 0,
    Bee,
}

pub struct Dataset {
    ants: Vec<RgbImage>,
    bees: Vec<RgbImage>,
}

impl Dataset {
    fn check_is_valid_dir(path: &Path) {
        if !path.exists() {
            panic!("Dataset path does not exist");
        }
        if !path.is_dir() {
            panic!("Dataset path is not a directory");
        }
    }

    pub fn from_dataset_path(paths: &Path) -> Dataset {
        #[cfg(debug_assertions)]
        Dataset::check_is_valid_dir(paths);
        let ants_dir = paths.join("ants");
        #[cfg(debug_assertions)]
        Dataset::check_is_valid_dir(&ants_dir);
        let bees_dir = paths.join("bees");
        #[cfg(debug_assertions)]
        Dataset::check_is_valid_dir(&bees_dir);

        let mut ants: Vec<RgbImage> = Vec::new();
        let mut bees: Vec<RgbImage> = Vec::new();

        for ant_img_path in read_dir(ants_dir).unwrap() {
            let img = ImageReader::open(ant_img_path.unwrap().path())
                .unwrap()
                .decode()
                .unwrap()
                .to_rgb8();
            ants.push(img);
        }
        for bee_img_path in read_dir(bees_dir).unwrap() {
            let img = ImageReader::open(bee_img_path.unwrap().path())
                .unwrap()
                .decode()
                .unwrap()
                .to_rgb8();
            bees.push(img);
        }
        return Dataset { ants, bees };
    }
}
