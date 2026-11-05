use infinity_provider_openrouter::{Backend, BoxError};
#[tokio::main]
async fn main() -> Result<(), BoxError> {
    infinity_provider_openrouter::run(Backend::OpenRouter).await
}
