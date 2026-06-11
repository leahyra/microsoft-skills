#!/usr/bin/env -S cargo +nightly -Zscript
---
[package]
edition = "2021"

[dependencies]
toml = "0.8"
---

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Validates that a Rust project uses TokenCredential-based authentication
//! from the Azure Identity SDK. Checks Cargo.toml for `azure_identity` and
//! scans .rs source files for valid credential types, rejecting banned
//! patterns like DefaultAzureCredential, connection strings, and SAS tokens.
//! Outputs a GraderResult JSON with scale_1_10 scoring.
//! - 10/10 (score 1.0): All checks pass
//! -  7/10 (score 0.67): Uses azure_identity but no valid credential type in source
//! -  5/10 (score 0.44): Missing azure_identity dependency
//! -  1/10 (score 0.0): Uses banned authentication patterns

use std::{env, fs, path::PathBuf, process};
use toml::Value;

/// Valid TokenCredential types in the Rust Azure SDK.
const VALID_CREDENTIALS: &[&str] = &[
    "DeveloperToolsCredential",
    "AzureCliCredential",
    "EnvironmentCredential",
    "ManagedIdentityCredential",
    "ClientSecretCredential",
    "WorkloadIdentityCredential",
];

/// Banned authentication patterns in source code (regex-free substring checks).
const BANNED_PATTERNS: &[(&str, &str)] = &[
    (
        "DefaultAzureCredential",
        "Uses DefaultAzureCredential which does not exist in the Rust SDK",
    ),
    (
        "connection_string",
        "Uses connection string authentication (banned)",
    ),
    (
        "from_connection_string",
        "Uses connection string authentication (banned)",
    ),
    (
        "ConnectionString",
        "Uses connection string authentication (banned)",
    ),
    (
        "SharedAccessSignature",
        "Uses shared access signatures (banned)",
    ),
    (
        "shared_access_signature",
        "Uses shared access signatures (banned)",
    ),
];

/// Normalize a raw score on the 1-10 scale to [0, 1].
fn normalize_1_10(raw: u8) -> f64 {
    (raw as f64 - 1.0) / 9.0
}

fn main() {
    let search_root = env::var("EVALUATE_WORKSPACE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    // Find the project root containing Cargo.toml
    let Some(cargo_path) = find_cargo_toml(&search_root) else {
        emit_result(
            false,
            1,
            "No Cargo.toml found",
            "incorrect",
            vec![format!(
                "No Cargo.toml found under {}",
                search_root.display()
            )],
            format!(
                "\"search_root\":\"{}\",\"error_kind\":\"missing_cargo_toml\"",
                json_escape(&search_root.display().to_string())
            ),
        );
        process::exit(0);
    };
    let project_root = cargo_path.parent().unwrap().to_path_buf();

    // Read and parse Cargo.toml
    let cargo_content = match fs::read_to_string(&cargo_path) {
        Ok(content) => content,
        Err(e) => {
            emit_result(
                false,
                1,
                &format!("Cannot read Cargo.toml: {e}"),
                "incorrect",
                vec![format!("Cannot read {}: {e}", cargo_path.display())],
                format!(
                    "\"cargo_toml\":\"{}\",\"project_root\":\"{}\",\"error_kind\":\"read_error\"",
                    json_escape(&cargo_path.display().to_string()),
                    json_escape(&project_root.display().to_string())
                ),
            );
            process::exit(0);
        }
    };

    let manifest: Value = match cargo_content.parse() {
        Ok(m) => m,
        Err(e) => {
            emit_result(
                false,
                1,
                &format!("Failed to parse Cargo.toml: {e}"),
                "incorrect",
                vec![format!("Failed to parse {}: {e}", cargo_path.display())],
                format!(
                    "\"cargo_toml\":\"{}\",\"project_root\":\"{}\",\"error_kind\":\"parse_error\"",
                    json_escape(&cargo_path.display().to_string()),
                    json_escape(&project_root.display().to_string())
                ),
            );
            process::exit(0);
        }
    };

    // Collect all .rs source files recursively under project root
    let mut rs_source = String::new();
    collect_rs_files(&project_root, &mut rs_source);
    let rs_file_count = count_rs_files(&project_root);

    if rs_source.is_empty() {
        emit_result(
            false,
            1,
            "No Rust source files found",
            "incorrect",
            vec![format!("No .rs files found under {}", project_root.display())],
            format!(
                "\"cargo_toml\":\"{}\",\"project_root\":\"{}\",\"checked_rs_files\":0,\"error_kind\":\"missing_rs_files\"",
                json_escape(&cargo_path.display().to_string()),
                json_escape(&project_root.display().to_string())
            ),
        );
        process::exit(0);
    }

    // Build list of all manifests to check (root + workspace members)
    let mut all_manifests: Vec<Value> = vec![manifest.clone()];
    if let Some(workspace_table) = manifest.get("workspace").and_then(|v| v.as_table()) {
        if let Some(members) = workspace_table.get("members").and_then(|v| v.as_array()) {
            for member in members {
                if let Some(member_path) = member.as_str() {
                    if !member_path.contains('*') && !member_path.contains('?') {
                        let member_manifest_path = project_root.join(member_path).join("Cargo.toml");
                        if let Ok(content) = fs::read_to_string(&member_manifest_path) {
                            if let Ok(m) = content.parse::<Value>() {
                                all_manifests.push(m);
                            }
                        }
                    }
                }
            }
        }
    }

    let mut failures: Vec<String> = Vec::new();

    // Check 1: azure_identity must be in Cargo.toml dependencies (root or any workspace member)
    let has_azure_identity = all_manifests.iter().any(|m| has_dependency(m, "azure_identity"));
    if !has_azure_identity {
        failures.push("azure_identity not found in Cargo.toml dependencies".to_string());
    }

    // Check 2: At least one valid TokenCredential type must appear in source
    let has_valid_credential = VALID_CREDENTIALS
        .iter()
        .any(|cred| rs_source.contains(cred));
    if !has_valid_credential {
        failures.push(format!(
            "No valid TokenCredential type found (expected one of: {})",
            VALID_CREDENTIALS.join(", ")
        ));
    }

    // Check 3: No banned authentication patterns
    for (pattern, message) in BANNED_PATTERNS {
        if rs_source.contains(pattern) {
            failures.push(message.to_string());
        }
    }

    // Deduplicate messages (e.g., multiple connection string patterns)
    failures.dedup();

    // Determine raw score on scale_1_10
    let has_banned = failures.iter().any(|f| {
        f.contains("DefaultAzureCredential")
            || f.contains("connection string")
            || f.contains("shared access signatures")
    });

    let raw_score: u8 = if has_banned {
        1
    } else if !has_azure_identity {
        5
    } else if !has_valid_credential {
        7
    } else {
        10
    };

    let passed = raw_score >= 5;
    let evidence = if failures.is_empty() {
        "Project uses TokenCredential authentication correctly with azure_identity".to_string()
    } else {
        failures.join("; ")
    };

    let label = if raw_score >= 10 {
        "correct"
    } else if raw_score >= 5 {
        "partially-correct"
    } else {
        "incorrect"
    };

    emit_result(
        passed,
        raw_score,
        &evidence,
        label,
        failures,
        format!(
            "\"cargo_toml\":\"{}\",\"project_root\":\"{}\",\"checked_rs_files\":{},\"has_azure_identity\":{},\"has_valid_credential\":{},\"has_banned_pattern\":{},\"valid_credentials\":{},\"raw_score\":{}",
            json_escape(&cargo_path.display().to_string()),
            json_escape(&project_root.display().to_string()),
            rs_file_count,
            has_azure_identity,
            has_valid_credential,
            has_banned,
            json_str_slice_array(VALID_CREDENTIALS),
            raw_score
        ),
    );

    process::exit(0);
}

fn emit_result(
    passed: bool,
    raw_score: u8,
    evidence: &str,
    label: &str,
    failures: Vec<String>,
    metadata_fields: String,
) {
    let normalized_score = normalize_1_10(raw_score);
    let summary = if passed {
        "TokenCredential authentication checks passed"
    } else {
        "TokenCredential authentication checks failed"
    };
    let failures_json = json_string_array(&failures);
    let metadata_with_diagnostics = format!(
        "{metadata_fields},\"summary\":\"{}\",\"failures\":{}",
        json_escape(summary),
        failures_json
    );

    let json = format!(
        r#"{{"name":"token-credential-authentication","kind":"code","passed":{passed},"score":{score},"evidence":"{evidence}","label":"{label}","metadata":{{{metadata}}}}}"#,
        passed = passed,
        score = normalized_score,
        evidence = json_escape(evidence),
        label = label,
        metadata = metadata_with_diagnostics,
    );

    println!("{json}");
}

/// Recursively collect all .rs file contents under `dir`, skipping hidden
/// directories, `target/`, and `node_modules/`.
fn collect_rs_files(dir: &PathBuf, out: &mut String) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if name_str.starts_with('.') || name_str == "target" || name_str == "node_modules" {
            continue;
        }

        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if name_str.ends_with(".rs") {
            if let Ok(content) = fs::read_to_string(&path) {
                out.push_str(&content);
                out.push('\n');
            }
        }
    }
}

fn count_rs_files(dir: &PathBuf) -> usize {
    let mut count = 0usize;
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return count,
    };

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if name_str.starts_with('.') || name_str == "target" || name_str == "node_modules" {
            continue;
        }

        let path = entry.path();
        if path.is_dir() {
            count += count_rs_files(&path);
        } else if name_str.ends_with(".rs") {
            count += 1;
        }
    }

    count
}

/// Check if a crate name appears in any dependency table of the manifest.
fn has_dependency(manifest: &Value, crate_name: &str) -> bool {
    let dep_tables = ["dependencies", "dev-dependencies", "build-dependencies"];
    for table_name in dep_tables {
        if let Some(deps) = manifest.get(table_name).and_then(|v| v.as_table()) {
            if deps.contains_key(crate_name) {
                return true;
            }
        }
    }
    false
}

/// Recursively search for a Cargo.toml file under `dir`, skipping hidden
/// directories, `target/`, and `node_modules/`.
fn find_cargo_toml(dir: &PathBuf) -> Option<PathBuf> {
    let candidate = dir.join("Cargo.toml");
    if candidate.exists() {
        return Some(candidate);
    }
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with('.') || name_str == "target" || name_str == "node_modules" {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_cargo_toml(&path) {
                return Some(found);
            }
        }
    }
    None
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn json_string_array(items: &[String]) -> String {
    let mut out = String::from("[");
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&json_escape(item));
        out.push('"');
    }
    out.push(']');
    out
}

fn json_str_slice_array(items: &[&str]) -> String {
    let mut out = String::from("[");
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&json_escape(item));
        out.push('"');
    }
    out.push(']');
    out
}
