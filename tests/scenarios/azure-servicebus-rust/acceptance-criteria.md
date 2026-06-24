# Azure Service Bus SDK for Rust Acceptance Criteria

**Crate**: `azure_messaging_servicebus`
**Repository**: <https://github.com/Azure/azure-sdk-for-rust/tree/main/sdk/servicebus/azure_messaging_servicebus>
**Purpose**: Skill testing acceptance criteria for validating generated Rust code correctness

---

## 0. Dependency Management Gate (Required)

### 0.1 ✅ CORRECT: Use cargo commands for dependency changes

```sh
cargo add azure_messaging_servicebus azure_identity tokio
cargo add azure_core
cargo remove azure_core
```

### 0.2 ✅ CORRECT: Add `azure_core` only for direct `azure_core` imports

```rust
use azure_core::error::ErrorKind;
use azure_messaging_servicebus::ServiceBusClient;
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
use azure_messaging_servicebus::ServiceBusClient;
// No direct azure_core import here, so forcing direct azure_core dependency is unnecessary.
```

---

## 1. Correct Import Patterns

### 1.1 ✅ CORRECT: Client and message imports

```rust
use azure_messaging_servicebus::{ServiceBusClient, Message};
use azure_identity::DeveloperToolsCredential;
```

---

## 2. Client Creation

### 2.1 ✅ CORRECT: ServiceBusClient builder and open

```rust
let credential = DeveloperToolsCredential::new(None)?;
let client = ServiceBusClient::builder()
    .open("your_namespace.servicebus.windows.net", credential.clone())
    .await?;
```

### 2.2 ❌ INCORRECT: Direct constructor call instead of builder pattern

```rust
// WRONG - use ServiceBusClient::builder().open(...)
let client = ServiceBusClient::new("namespace", credential)?;
```

---

## 3. Queue and Topic Operations

### 3.1 ✅ CORRECT: Send message to queue

```rust
let sender = client.create_sender("my_queue", None).await?;
let message = Message::from("Hello, Service Bus!");
sender.send_message(message, None).await?;
```

### 3.2 ✅ CORRECT: Receive and complete queue messages

```rust
let receiver = client.create_receiver("my_queue", None).await?;
let messages = receiver.receive_messages(5, None).await?;
for message in messages {
    println!("Received: {}", message.body_as_string()?);
    receiver.complete_message(&message, None).await?;
}
```

### 3.3 ✅ CORRECT: Topic/subscription receive

```rust
let receiver = client
    .create_receiver_for_subscription("my_topic", "my_subscription", None)
    .await?;

let messages = receiver.receive_messages(5, None).await?;
for message in messages {
    receiver.complete_message(&message, None).await?;
}
```

### 3.4 ❌ INCORRECT: Forgetting message settlement

```rust
// WRONG - leaving messages unsettled causes redelivery
for message in receiver.receive_messages(5, None).await? {
    println!("{}", message.body_as_string()?);
}
```

---

## 4. Package Maturity Warning

### 4.1 ✅ CORRECT: Include pre-production warning

Generated guidance should mention this crate is in early development and should not be used in production without caution.
