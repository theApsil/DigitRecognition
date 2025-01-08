use onnxruntime::session::Session;
use onnxruntime::{environment::Environment, tensor::OrtOwnedTensor, GraphOptimizationLevel};
use std::error::Error;

lazy_static::lazy_static! {
    static ref ENV: Environment = Environment::builder()
        .with_name("onnxruntime_env")
        .with_log_level(onnxruntime::LoggingLevel::Warning)
        .build()
        .unwrap();
}

pub fn load_model() -> Session<'static> {
    ENV.new_session_builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::All)
        .unwrap()
        .with_model_from_file("models/model.onnx")
        .unwrap()
}

pub fn predict(body: &[u8], session: &Session) -> Result<i64, Box<dyn Error>> {
    let input_data = preprocess_input(body)?;

    let input_tensor = vec![input_data];

    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(vec![input_tensor])?;
    let prediction = outputs[0].to_vec();

    // Определяем индекс максимального значения (т.е. предсказанную цифру)
    let predicted_digit = prediction
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .ok_or("Failed to process prediction")?;

    Ok(predicted_digit as i64)
}

fn preprocess_input(body: &[u8]) -> Result<Vec<f32>, Box<dyn Error>> {
    let input_data: Vec<f32> = body
        .iter()
        .map(|&byte| byte as f32 / 255.0) // Нормализация значений
        .collect();

    if input_data.len() != 28 * 28 {
        return Err("Invalid input size, expected 28x28 grayscale image".into());
    }

    Ok(input_data)
}
