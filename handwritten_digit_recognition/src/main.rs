use warp::Filter;
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::recognize::{load_model, predict_digit};

mod recognize;

#[tokio::main]
async fn main() {
    // Загружаем модель и оборачиваем её в Arc<Mutex<>>
    let model = Arc::new(Mutex::new(load_model()));

    // Роут для предсказания
    let recognize_route = warp::path("recognize")
        .and(warp::post())
        .and(warp::body::bytes())
        .and(with_model(model.clone()))
        .and_then(handle_recognition);

    // Роут для статических файлов
    let static_files = warp::fs::dir("./static");

    // Объединяем маршруты
    let routes = recognize_route.or(static_files);

    // Запуск сервера
    warp::serve(routes).run(([127, 0, 0, 1], 3030)).await;
}

// Обработчик для распознавания
async fn handle_recognition(
    body: warp::hyper::body::Bytes,
    model: Arc<Mutex<onnxruntime::Session>>,
) -> Result<impl warp::Reply, warp::Rejection> {
    // Конвертируем изображение в формат для модели
    match predict_digit(&body, model).await {
        Ok(prediction) => Ok(warp::reply::json(&serde_json::json!({ "digit": prediction }))),
        Err(e) => {
            eprintln!("Ошибка распознавания: {}", e);
            Ok(warp::reply::with_status(
                warp::reply::json(&serde_json::json!({ "error": "Failed to process image" })),
                warp::http::StatusCode::INTERNAL_SERVER_ERROR,
            ))
        }
    }
}

// Передача модели в обработчики
fn with_model(
    model: Arc<Mutex<onnxruntime::Session>>,
) -> impl Filter<Extract = (Arc<Mutex<onnxruntime::Session>>,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || model.clone())
}
