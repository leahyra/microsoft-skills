fn main() {
    let _ = futures::executor::block_on(do_work());
}

async fn do_work() -> usize {
    1
}
