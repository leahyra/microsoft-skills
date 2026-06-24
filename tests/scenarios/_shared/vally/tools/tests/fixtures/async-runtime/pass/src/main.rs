#[tokio::main]
async fn main() {
    let _ = do_work().await;
}

async fn do_work() -> usize {
    1
}
