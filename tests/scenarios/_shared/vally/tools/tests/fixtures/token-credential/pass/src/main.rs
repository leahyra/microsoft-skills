use azure_identity::AzureCliCredential;

fn main() {
    let _credential = AzureCliCredential::new(None).expect("credential should be constructible");
}
