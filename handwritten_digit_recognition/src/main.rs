use std::sync::Arc;
use tokio::sync::Mutex;
use warp::Filter;
use serde_json::json;
use onnxruntime::session::Session;

mod recognize;

#[tokio::main]
async fn main() {
    let model = Arc::new(Mutex::new(recognize::load_model()));

    let routes = warp::path!("predict")
        .and(warp::post())
        .and(warp::body::bytes())
        .and(warp::any().map(move || Arc::clone(&model)))
        .and_then(handle_prediction);

    println!("Server running at http://127.0.0.1:3030");
    warp::serve(routes).run(([127, 0, 0, 1], 3030)).await;
}

async fn handle_prediction(
    body: bytes::Bytes,
    model: Arc<Mutex<Session<'static>>>,
) -> Result<impl warp::Reply, warp::Rejection> {
    let model = model.lock().await;

    match recognize::predict(&body, &model) {
        Ok(prediction) => Ok(warp::reply::json(&json!({
            "success": true,
            "prediction": prediction
        }))),
        Err(err) => Ok(warp::reply::with_status(
            warp::reply::json(&json!({
                "success": false,
                "error": err.to_string()
            })),
            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
        )),
    }
}
