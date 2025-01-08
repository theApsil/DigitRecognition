use onnxruntime::{environment::Environment, ndarray::Array, session::Session};
use std::error::Error;
use image::io::Reader as ImageReader;
use image::DynamicImage;

pub fn load_model() -> Session {
    let environment = Environment::builder()
        .with_name("handwritten_digit_recognition")
        .build()
        .unwrap();

    environment
        .new_session_builder()
        .unwrap()
        .with_model_from_file("models/model.onnx")
        .unwrap()
}

pub async fn predict_digit(
    body: &[u8],
    model: std::sync::Arc<tokio::sync::Mutex<Session>>,
) -> Result<i64, Box<dyn Error>> {
    // Конвертируем байты в изображение
    let img = ImageReader::new(std::io::Cursor::new(body))
        .with_guessed_format()?
        .decode()?;

    let processed = preprocess_image(&img)?;

    // Запускаем модель
    let session = model.lock().await;
    let input_tensor = Array::from_shape_vec((1, 1, 28, 28), processed)?.into_dyn();
    let outputs = session.run(vec![input_tensor.into()])?;
    let result = outputs[0].as_array().unwrap();

    Ok(result.argmax().unwrap() as i64)
}

fn preprocess_image(img: &DynamicImage) -> Result<Vec<f32>, Box<dyn Error>> {
    let gray_image = img.to_luma8();
    let resized = image::imageops::resize(&gray_image, 28, 28, image::imageops::Nearest);
    let normalized = resized
        .pixels()
        .map(|p| p[0] as f32 / 255.0)
        .collect::<Vec<f32>>();
    Ok(normalized)
}
