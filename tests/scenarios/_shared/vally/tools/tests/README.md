# Tools Regression Tests

This directory contains integration tests for every script in `tools/`.

## What Is Covered

- `check-async-runtime.rs`
- `check-token-credential.rs`
- `check-no-secrets.rs`
- `check-azure-crates.rs` (including semver-compatible latest-version checks)

## Run

From the repository root:

```powershell
node tests/scenarios/_shared/vally/tools/tests/run-tools-tests.mjs
```

## Notes

- The `check-azure-crates.rs` tests call crates.io to get current latest Azure crate versions.
- The suite is fixture-based for deterministic pass/fail scenarios across the other tools.
