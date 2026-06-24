#!/usr/bin/env -S cargo +nightly -Zscript
---
[package]
edition = "2021"

[dependencies]
regex = "1"
---

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Validates that a Rust project does not contain hardcoded secrets such as
//! Azure Storage account keys, shared access signatures, connection strings,
//! or suspicious base64 string literals.
//! Outputs GraderResult JSON with diagnostics in `details` and `metadata`.

use regex::Regex;
use std::{env, fs, path::PathBuf, process};

fn normalize_1_10(raw: u8) -> f64 {
    (raw as f64 - 1.0) / 9.0
}

/// Secret detection patterns with their descriptions.
const PATTERNS: &[(&str, &str)] = &[
    (
        r"AccountKey\s*=\s*[A-Za-z0-9+/=]{20,}",
        "Contains what appears to be a hardcoded AccountKey",
    ),
    (
        r"SharedAccessSignature\s*=|[?&]sig=[A-Za-z0-9%+/=]{20,}",
        "Contains what appears to be a shared access signature",
    ),
    (
        r"DefaultEndpointsProtocol=.*AccountName=.*AccountKey=",
        "Contains a connection string",
    ),
    (
        r#"["'][A-Za-z0-9+/]{40,}={0,2}['"]"#,
        "Contains a suspicious base64 string literal (possible key)",
    ),
];

fn main() {
    let search_root = env::var("EVALUATE_WORKSPACE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    // Find the project root containing Cargo.toml
    let Some(cargo_toml) = find_cargo_toml(&search_root) else {
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
    let project_root = cargo_toml.parent().unwrap().to_path_buf();

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
                json_escape(&cargo_toml.display().to_string()),
                json_escape(&project_root.display().to_string())
            ),
        );
        process::exit(0);
    }

    let mut failures: Vec<String> = Vec::new();
    let mut matched_patterns: Vec<String> = Vec::new();

    for (pattern, message) in PATTERNS {
        let re = Regex::new(pattern).unwrap_or_else(|e| {
            emit_result(
                false,
                1,
                "Invalid grader regex",
                "incorrect",
                vec![format!("Invalid regex '{pattern}': {e}")],
                format!(
                    "\"cargo_toml\":\"{}\",\"project_root\":\"{}\",\"checked_rs_files\":{},\"error_kind\":\"invalid_regex\"",
                    json_escape(&cargo_toml.display().to_string()),
                    json_escape(&project_root.display().to_string()),
                    rs_file_count
                ),
            );
            process::exit(0);
        });
        if re.is_match(&rs_source) {
            failures.push(message.to_string());
            matched_patterns.push((*pattern).to_string());
        }
    }

    let passed = failures.is_empty();
    let raw_score = if passed { 10 } else { 1 };
    let evidence = if passed {
        "No hardcoded secret patterns found in Rust source files".to_string()
    } else {
        failures.join("; ")
    };
    let label = if passed { "correct" } else { "incorrect" };

    emit_result(
        passed,
        raw_score,
        &evidence,
        label,
        failures,
        format!(
            "\"cargo_toml\":\"{}\",\"project_root\":\"{}\",\"checked_rs_files\":{},\"matched_patterns\":{},\"pattern_count\":{}",
            json_escape(&cargo_toml.display().to_string()),
            json_escape(&project_root.display().to_string()),
            rs_file_count,
            json_string_array(&matched_patterns),
            PATTERNS.len()
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
        "No-hardcoded-secrets checks passed"
    } else {
        "No-hardcoded-secrets checks failed"
    };
    let failures_json = json_string_array(&failures);
    let metadata_with_diagnostics = format!(
        "{metadata_fields},\"summary\":\"{}\",\"failures\":{}",
        json_escape(summary),
        failures_json
    );

    let json = format!(
        r#"{{"name":"no-hardcoded-secrets","kind":"code","passed":{passed},"score":{score},"evidence":"{evidence}","label":"{label}","metadata":{{{metadata}}}}}"#,
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
