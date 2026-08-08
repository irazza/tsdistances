use std::fmt::write;

use csv::ReaderBuilder;
use tsdistances_gpu::{
    cpu::{erp, lcss, dtw, wdtw, adtw, msm, twe},
    utils::get_device,
};

fn read_txt<T>(file_path: &str) -> Result<Vec<Vec<T>>, Box<dyn std::error::Error>>
where
    T: std::str::FromStr,
    T::Err: 'static + std::error::Error,
{
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(if file_path.ends_with(".tsv") { b'\t' } else { b',' })
        .from_path(file_path)?;

    let mut records = Vec::new();
    for result in reader.records() {
        let record = result?;
        let row: Vec<T> = record
            .iter()
            .skip(1) // Skip the label column
            .map(|s| s.parse::<T>())
            .collect::<Result<Vec<_>, _>>()?;
        records.push(row);
    }
    Ok(records)
}

pub fn write_csv<T>(file_path: &str, data: &[Vec<T>]) -> Result<(), Box<dyn std::error::Error>>
where
    T: std::fmt::Display,
{
    let mut wtr = csv::Writer::from_path(file_path)?;
    for row in data {
        wtr.write_record(row.iter().map(|item| item.to_string()))?;
    }
    wtr.flush()?;
    Ok(())
}

const WEIGHT_MAX: f32 = 1.0;
fn dtw_weights(len: usize, g: f32) -> Vec<f32> {
    let mut weights = vec![0.0; len];
    let half_len = len as f32 / 2.0;
    let e = std::f64::consts::E as f32;
    for i in 0..len {
        weights[i] = WEIGHT_MAX / (1.0 + e.powf(-g * (i as f32 - half_len)));
    }
    weights
}

#[test]
fn test_erp_distance() {
    let train_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TRAIN.csv").unwrap();
    let test_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TEST.csv").unwrap();

    let start_time = std::time::Instant::now();
    let (device, queue, sba, sda, ma) = get_device();

    let result = erp(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &train_data,
        &test_data,
        0.0,
    );
    let elapsed_time = start_time.elapsed();
    println!("ERP elapsed time: {:?}", elapsed_time);
    // write_csv("erp_result.csv", &result).unwrap();

}

#[test]
fn test_lcss_distance() {
    let train_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TRAIN.csv").unwrap();
    let test_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TEST.csv").unwrap();
    let epsilon = 1.0;

    let start = std::time::Instant::now();
    let (device, queue, sba, sda, ma) = get_device();
    let result = lcss(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &train_data,
        &test_data,
        epsilon,
    );
    let elapsed = start.elapsed();
    println!("LCSS elapsed time: {:?}", elapsed);
    // write_csv("lcss_result.csv", &result).unwrap();
}

#[test]
fn test_dtw_distance() {
    let train_data: Vec<Vec<f32>> = read_txt("../../DATA/ucr/Wafer/Wafer_TRAIN.tsv").unwrap();
    let test_data: Vec<Vec<f32>> = read_txt("../../DATA/ucr/Wafer/Wafer_TEST.tsv").unwrap();

    let start_time = std::time::Instant::now();
    let (device, queue, sba, sda, ma) = get_device();

    let result = dtw(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &train_data,
        &test_data,
    );
    let elapsed_time = start_time.elapsed();
    println!("DTW elapsed time: {:?}", elapsed_time);
    // write_csv("dtw_result.csv", &result).unwrap();
}

#[test]
fn test_wdtw_distance() {
    let train_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TRAIN.csv").unwrap();
    let test_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TEST.csv").unwrap();

    let g = 0.05;
    let weights = dtw_weights(train_data[0].len(), g);

    let start = std::time::Instant::now();
    let (device, queue, sba, sda, ma) = get_device();

    let result = wdtw(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &train_data,
        &test_data,
        &weights,
    );
    let elapsed_time = start.elapsed();
    println!("WDTW elapsed time: {:?}", elapsed_time);
    // write_csv("wdtw_result.csv", &result).unwrap();
}

#[test]
fn test_adtw_distance() {
    let train_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TRAIN.csv").unwrap();
    let test_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TEST.csv").unwrap();
    
    let w = 0.1;

    let start_time = std::time::Instant::now();
    let (device, queue, sba, sda, ma) = get_device();

    let result = adtw(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &train_data,
        &test_data,
        w,
    );
    let elapsed_time = start_time.elapsed();
    println!("ADTW elapsed time: {:?}", elapsed_time);
    // write_csv("adtw_result.csv", &result).unwrap();
}

#[test]
fn test_msm_distance() {
    let train_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TRAIN.csv").unwrap();
    let test_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TEST.csv").unwrap();

    let start_time = std::time::Instant::now();
    let (device, queue, sba, sda, ma) = get_device();

    let result = msm(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &train_data,
        &test_data,
    );
    let elapsed_time = start_time.elapsed();
    println!("MSM elapsed time: {:?}", elapsed_time);
    // write_csv("msm_result.csv", &result).unwrap();
}

#[test]
fn test_twe_distance() {
    let train_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TRAIN.csv").unwrap();
    let test_data: Vec<Vec<f32>> = read_txt("tests/ACSF1/ACSF1_TEST.csv").unwrap();

    let stiffness = 0.001;
    let penalty = 1.0;

    let start_time = std::time::Instant::now();
    let (device, queue, sba, sda, ma) = get_device();

    let result = twe(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &train_data,
        &test_data,
        stiffness,
        penalty,
    );
    let elapsed_time = start_time.elapsed();
    println!("TWE elapsed time: {:?}", elapsed_time);
    // write_csv("twe_result.csv", &result).unwrap();
}
