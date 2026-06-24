# Azure Storage Queue SDK for Rust Acceptance Criteria

**Crate**: `azure_storage_queue`
**Repository**: <https://github.com/Azure/azure-sdk-for-rust/tree/main/sdk/storage/azure_storage_queue>
**Purpose**: Skill testing acceptance criteria for validating generated Rust code correctness

---

## 0. Dependency Management Gate (Required)

### 0.1 ✅ CORRECT: Use cargo commands for dependency changes

```sh
cargo add azure_storage_queue azure_identity tokio
cargo add azure_core
cargo remove azure_core
```

### 0.2 ✅ CORRECT: Add `azure_core` only for direct `azure_core` imports

```rust
use azure_core::http::Url;
use azure_storage_queue::QueueServiceClient;
// Direct azure_core import is used, so `azure_core` should be a direct dependency.
```

### 0.3 ❌ INCORRECT: Manual Cargo.toml dependency edits in generated guidance

```toml
# WRONG in generated guidance - use `cargo add` / `cargo remove` commands instead
[dependencies]
azure_core = "*"
```

### 0.4 ❌ INCORRECT: Requiring `azure_core` when no direct `azure_core` imports exist

```rust
use azure_storage_queue::QueueServiceClient;
// No direct azure_core import here, so forcing direct azure_core dependency is unnecessary.
```

---

## 1. Correct Import Patterns

### 1.1 ✅ CORRECT: Client and model imports

```rust
use azure_storage_queue::QueueServiceClient;
use azure_storage_queue::QueueClient;
use azure_storage_queue::models::QueueMessage;
use azure_identity::DeveloperToolsCredential;
use azure_core::http::Url;
```

---

## 2. Client Creation

### 2.1 ✅ CORRECT: QueueServiceClient with Entra ID

```rust
let credential = DeveloperToolsCredential::new(None)?;
let service_url = Url::parse("https://<account>.queue.core.windows.net/")?;
let service_client = QueueServiceClient::new(service_url, Some(credential), None)?;
let queue_client = service_client.queue_client("<queue_name>")?;
```

### 2.2 ❌ INCORRECT: Hardcoded account key in code

```rust
// WRONG - use Entra ID auth via DeveloperToolsCredential/ManagedIdentityCredential
let account_key = "actual-key-here";
```

---

## 3. Queue Operations

### 3.1 ✅ CORRECT: Send message

```rust
let message = QueueMessage {
    message_text: Some("hello world".to_string()),
};
queue_client.send_message(message.try_into()?, None).await?;
```

### 3.2 ✅ CORRECT: Receive messages

```rust
let response = queue_client.receive_messages(None).await?;
let messages = response.into_model()?;
for msg in messages.items.unwrap_or_default() {
    println!("{}", msg.message_text.as_deref().unwrap_or("<empty>"));
}
```

### 3.3 ✅ CORRECT: Delete received messages with message_id + pop_receipt

```rust
let response = queue_client.receive_messages(None).await?;
let messages = response.into_model()?;
for msg in messages.items.unwrap_or_default() {
    if let (Some(id), Some(pop_receipt)) = (&msg.message_id, &msg.pop_receipt) {
        queue_client.delete_message(id, pop_receipt, None).await?;
    }
}
```

### 3.4 ✅ CORRECT: Peek messages

```rust
let response = queue_client.peek_messages(None).await?;
let messages = response.into_model()?;
for msg in messages.items.unwrap_or_default() {
    println!("Peeked: {}", msg.message_text.as_deref().unwrap_or("<empty>"));
}
```

### 3.5 ❌ INCORRECT: Deleting without pop receipt

```rust
// WRONG - delete_message needs both message_id and pop_receipt
queue_client.delete_message("id-only", "", None).await?;
```
