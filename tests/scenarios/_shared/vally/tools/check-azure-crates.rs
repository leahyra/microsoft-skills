#!/usr/bin/env -S cargo +nightly -Zscript
---
[package]
edition = "2021"

[dependencies]
toml = "0.8"
serde_json = "1"
ureq = { version = "2", default-features = true }
semver = "1"
---

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Validates that a Cargo.toml uses official Azure SDK crates, rejects
//! banned/obsolete dependencies, and checks dependency version hygiene.
//! Outputs a GraderResult JSON with scale_1_10 scoring.
//! - 10/10 (score 1.0): All checks pass
//! -  7/10 (score 0.67): Uses azure_core directly (prefer higher-level crates)
//! -  6/10 (score 0.56): Azure crates are not on the latest crates.io version
//! -  5/10 (score 0.44): Missing required deps or wildcard versions
//! -  1/10 (score 0.0): Uses banned/obsolete legacy crates

use semver::{Version, VersionReq};
use serde_json::Value as JsonValue;
use std::{collections::HashMap, env, fs, path::PathBuf, process};
use toml::Value;

/// Crate names that must never appear in dependencies.
const BANNED_CRATES: &[&str] = &[
    "azure_storage_blobs", // unofficial plural form
    "azure_storage",       // legacy standalone crate
];

/// Version that indicates an obsolete azure_* crate.
const OBSOLETE_VERSION: &str = "0.21.0";

/// Normalize a raw score on the 1-10 scale to [0, 1].
fn normalize_1_10(raw: u8) -> f64 {
    (raw as f64 - 1.0) / 9.0
}

fn main() {
    let arg = env::args().nth(1);

    // Resolve the path to Cargo.toml:
    // - If an explicit argument is given, resolve relative paths against EVALUATE_WORKSPACE
    // - If no argument is given, search recursively under EVALUATE_WORKSPACE (or cwd)
    let path = match arg {
        Some(a) => {
            if PathBuf::from(&a).is_relative() {
                if let Ok(workspace) = env::var("EVALUATE_WORKSPACE") {
                    PathBuf::from(workspace).join(&a)
                } else {
                    PathBuf::from(&a)
                }
            } else {
                PathBuf::from(&a)
            }
        }
        None => {
            let search_root = env::var("EVALUATE_WORKSPACE")
                .map(PathBuf::from)
                .unwrap_or_else(|_| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
            let found = find_cargo_toml(&search_root);
            match found {
                Some(p) => p,
                None => {
                    eprintln!("ERROR: No Cargo.toml found under {}", search_root.display());
                    process::exit(2);
                }
            }
        }
    };

    let content = fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("ERROR: Cannot read {}: {e}", path.display());
        process::exit(2);
    });

    let manifest: Value = content.parse().unwrap_or_else(|e| {
        eprintln!("ERROR: Failed to parse {}: {e}", path.display());
        process::exit(2);
    });

    let mut manifests_to_check: Vec<(PathBuf, Value)> = vec![(path.clone(), manifest)];
    let mut workspace_member_manifests: Vec<PathBuf> = Vec::new();
    let mut workspace_member_warnings: Vec<String> = Vec::new();

    if let Some((member_paths, member_warnings)) =
        resolve_workspace_member_manifests(&path, &manifests_to_check[0].1)
    {
        workspace_member_warnings.extend(member_warnings);
        for member_manifest_path in member_paths {
            let member_content = match fs::read_to_string(&member_manifest_path) {
                Ok(c) => c,
                Err(e) => {
                    workspace_member_warnings.push(format!(
                        "Failed to read workspace member manifest {}: {e}",
                        member_manifest_path.display()
                    ));
                    continue;
                }
            };

            let member_manifest: Value = match member_content.parse() {
                Ok(v) => v,
                Err(e) => {
                    workspace_member_warnings.push(format!(
                        "Failed to parse workspace member manifest {}: {e}",
                        member_manifest_path.display()
                    ));
                    continue;
                }
            };

            workspace_member_manifests.push(member_manifest_path.clone());
            manifests_to_check.push((member_manifest_path, member_manifest));
        }
    }

    let mut has_banned_crate = false;
    let mut has_obsolete_version = false;
    let mut has_azure_core_dep = false;
    let mut has_azure_identity_dep = false;
    let mut has_official_azure_crate = false;
    let mut has_wildcard_version = false;
    let mut has_outdated_azure_crate = false;
    let mut has_tokio = false;
    let mut has_futures = false;
    let mut messages: Vec<String> = Vec::new();
    let mut observed_azure_requirements: HashMap<String, String> = HashMap::new();
    let mut outdated_azure_crates: Vec<String> = Vec::new();

    let dep_tables = ["dependencies", "dev-dependencies", "build-dependencies"];

    for (manifest_path, manifest_value) in &manifests_to_check {
        for table_name in dep_tables {
            let Some(deps) = manifest_value.get(table_name).and_then(|v| v.as_table()) else {
                continue;
            };

            scan_dependency_table(
                deps,
                &format!("{} [{table_name}]", manifest_path.display()),
                &mut has_banned_crate,
                &mut has_obsolete_version,
                &mut has_azure_core_dep,
                &mut has_azure_identity_dep,
                &mut has_official_azure_crate,
                &mut has_wildcard_version,
                &mut has_tokio,
                &mut has_futures,
                &mut messages,
                &mut observed_azure_requirements,
            );
        }

        if let Some(workspace_deps) = manifest_value
            .get("workspace")
            .and_then(|v| v.as_table())
            .and_then(|t| t.get("dependencies"))
            .and_then(|v| v.as_table())
        {
            scan_dependency_table(
                workspace_deps,
                &format!("{} [workspace.dependencies]", manifest_path.display()),
                &mut has_banned_crate,
                &mut has_obsolete_version,
                &mut has_azure_core_dep,
                &mut has_azure_identity_dep,
                &mut has_official_azure_crate,
                &mut has_wildcard_version,
                &mut has_tokio,
                &mut has_futures,
                &mut messages,
                &mut observed_azure_requirements,
            );
        }
    }

    let mut version_check_warnings: Vec<String> = Vec::new();
    for (crate_name, declared_requirement) in &observed_azure_requirements {
        match fetch_latest_crate_version(crate_name) {
            Ok(latest_version) => {
                match requirement_satisfies_latest(declared_requirement, &latest_version) {
                    Ok(true) => {}
                    Ok(false) => {
                        has_outdated_azure_crate = true;
                        messages.push(format!(
                            "Outdated Azure crate '{crate_name}': requirement '{declared_requirement}' does not include latest crates.io version {latest_version}"
                        ));
                        outdated_azure_crates.push(format!(
                            "{crate_name}:{declared_requirement}->{latest_version}"
                        ));
                    }
                    Err(err) => {
                        version_check_warnings.push(format!(
                            "Unable to compare version requirement for '{crate_name}': {err}"
                        ));
                    }
                }
            }
            Err(err) => {
                version_check_warnings.push(format!(
                    "Unable to verify latest crates.io version for '{crate_name}': {err}"
                ));
            }
        }
    }

    messages.extend(workspace_member_warnings);
    messages.extend(version_check_warnings);

    if !has_official_azure_crate {
        messages.push("No official azure_* crates found in scanned manifests".to_string());
    }

    // Dependency version checks
    let mut missing_deps: Vec<&str> = Vec::new();
    if !has_azure_core_dep && !has_official_azure_crate {
        // azure_core is implicitly required if any azure crate is used
    }
    if has_official_azure_crate && !has_azure_identity_dep {
        missing_deps.push("azure_identity");
        messages.push("Missing required dependency 'azure_identity'".to_string());
    }
    if has_official_azure_crate && !has_tokio {
        missing_deps.push("tokio");
        messages.push("Missing async runtime dependency 'tokio'".to_string());
    }
    if has_official_azure_crate && !has_futures {
        missing_deps.push("futures/futures-util");
        messages
            .push("Missing 'futures' or 'futures-util' for async stream operations".to_string());
    }

    // Determine raw score on scale_1_10
    let raw_score: u8 = if has_banned_crate || has_obsolete_version || !has_official_azure_crate {
        1
    } else if has_wildcard_version {
        3
    } else if !missing_deps.is_empty() {
        5
    } else if has_outdated_azure_crate {
        6
    } else if has_azure_core_dep {
        7
    } else {
        10
    };

    let normalized_score = normalize_1_10(raw_score);
    // Treat missing required dependencies (raw_score 5) as failing.
    let passed = raw_score >= 7;
    let failures = messages.clone();
    let evidence = if messages.is_empty() {
        "Cargo.toml uses official Azure SDK crates with correct dependency versions".to_string()
    } else {
        messages.join("; ")
    };

    let label = if raw_score >= 10 {
        "correct"
    } else if raw_score >= 7 {
        "partially-correct"
    } else {
        "incorrect"
    };

    let details_summary = if passed {
        "Azure SDK dependency checks passed".to_string()
    } else {
        "Azure SDK dependency checks failed".to_string()
    };
    let details_failures = json_string_array(&failures);
    let metadata_missing_deps = json_str_slice_array(&missing_deps);
    let metadata_manifests_checked = json_path_array(
        &manifests_to_check
            .iter()
            .map(|(p, _)| p.clone())
            .collect::<Vec<PathBuf>>(),
    );
    let metadata_workspace_members = json_path_array(&workspace_member_manifests);
    let metadata_summary = json_escape(&details_summary);
    let metadata_outdated_crates = json_string_array(&outdated_azure_crates);

    // Output GraderResult JSON
    let json = format!(
        r#"{{"name":"official-azure-sdk-crate-selection","kind":"code","passed":{passed},"score":{normalized_score},"evidence":"{evidence}","label":"{label}","metadata":{{"cargo_toml":"{cargo_toml}","manifest_count":{manifest_count},"manifests_checked":{metadata_manifests_checked},"workspace_member_manifests":{metadata_workspace_members},"has_official_azure_crate":{has_official_azure_crate},"has_azure_core_dep":{has_azure_core_dep},"has_azure_identity_dep":{has_azure_identity_dep},"has_tokio":{has_tokio},"has_futures":{has_futures},"has_banned_crate":{has_banned_crate},"has_obsolete_version":{has_obsolete_version},"has_wildcard_version":{has_wildcard_version},"has_outdated_azure_crate":{has_outdated_azure_crate},"outdated_azure_crates":{metadata_outdated_crates},"missing_dependencies":{metadata_missing_deps},"raw_score":{raw_score},"summary":"{metadata_summary}","failures":{details_failures}}}}}"#,
        passed = passed,
        normalized_score = normalized_score,
        evidence = json_escape(&evidence),
        label = label,
        details_failures = details_failures,
        metadata_summary = metadata_summary,
        cargo_toml = json_escape(&path.display().to_string()),
        manifest_count = manifests_to_check.len(),
        metadata_manifests_checked = metadata_manifests_checked,
        metadata_workspace_members = metadata_workspace_members,
        has_official_azure_crate = has_official_azure_crate,
        has_azure_core_dep = has_azure_core_dep,
        has_azure_identity_dep = has_azure_identity_dep,
        has_tokio = has_tokio,
        has_futures = has_futures,
        has_banned_crate = has_banned_crate,
        has_obsolete_version = has_obsolete_version,
        has_wildcard_version = has_wildcard_version,
        has_outdated_azure_crate = has_outdated_azure_crate,
        metadata_outdated_crates = metadata_outdated_crates,
        metadata_missing_deps = metadata_missing_deps,
        raw_score = raw_score,
    );

    println!("{json}");
    process::exit(0);
}

/// Extract the version string from a dependency value.
/// Handles both `crate = "version"` and `crate = { version = "..." }` forms.
fn extract_version(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Table(t) => t.get("version").and_then(|v| v.as_str()).map(String::from),
        _ => None,
    }
}

/// Returns true if the version requirement includes the obsolete version.
fn is_obsolete_version(version_req: &str) -> bool {
    let trimmed = version_req.trim();
    let version_part = trimmed.trim_start_matches(|c: char| {
        c == '^' || c == '~' || c == '=' || c == '>' || c == '<' || c == ' '
    });
    version_part == OBSOLETE_VERSION
}

/// Returns true if the version requirement is a wildcard (e.g., "*").
fn is_wildcard_version(version_req: &str) -> bool {
    let trimmed = version_req.trim();
    trimmed == "*" || trimmed.ends_with(".*")
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

fn json_path_array(items: &[PathBuf]) -> String {
    let mut out = String::from("[");
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&json_escape(&item.display().to_string()));
        out.push('"');
    }
    out.push(']');
    out
}

fn scan_dependency_table(
    deps: &toml::map::Map<String, Value>,
    source_label: &str,
    has_banned_crate: &mut bool,
    has_obsolete_version: &mut bool,
    has_azure_core_dep: &mut bool,
    has_azure_identity_dep: &mut bool,
    has_official_azure_crate: &mut bool,
    has_wildcard_version: &mut bool,
    has_tokio: &mut bool,
    has_futures: &mut bool,
    messages: &mut Vec<String>,
    observed_azure_requirements: &mut HashMap<String, String>,
) {
    for (crate_name, dep_value) in deps {
        if BANNED_CRATES.contains(&crate_name.as_str()) {
            *has_banned_crate = true;
            messages.push(format!(
                "Banned crate '{crate_name}' found in {source_label}"
            ));
        }

        if crate_name == "tokio" {
            *has_tokio = true;
        }
        if crate_name == "futures" || crate_name == "futures-util" {
            *has_futures = true;
        }

        if crate_name.starts_with("azure_") {
            *has_official_azure_crate = true;

            if crate_name == "azure_core" {
                *has_azure_core_dep = true;
            }
            if crate_name == "azure_identity" {
                *has_azure_identity_dep = true;
            }

            if let Some(ver) = extract_version(dep_value) {
                observed_azure_requirements
                    .entry(crate_name.clone())
                    .or_insert(ver.clone());
                if is_obsolete_version(&ver) {
                    *has_obsolete_version = true;
                    messages.push(format!(
                        "Obsolete azure crate '{crate_name}' at version {ver} in {source_label} \
                             (azure_* crates at {OBSOLETE_VERSION} are obsolete)"
                    ));
                }
                if is_wildcard_version(&ver) {
                    *has_wildcard_version = true;
                    messages.push(format!(
                        "Wildcard version '{ver}' for '{crate_name}' in {source_label}; use a concrete version"
                    ));
                }
            }
        }
    }
}

fn requirement_satisfies_latest(requirement: &str, latest_version: &str) -> Result<bool, String> {
    let req = VersionReq::parse(requirement.trim())
        .map_err(|e| format!("invalid requirement '{requirement}': {e}"))?;
    let latest = Version::parse(latest_version.trim())
        .map_err(|e| format!("invalid latest version '{latest_version}': {e}"))?;
    Ok(req.matches(&latest))
}

fn fetch_latest_crate_version(crate_name: &str) -> Result<String, String> {
    let url = format!("https://crates.io/api/v1/crates/{crate_name}");
    let response = ureq::get(&url)
        .set("User-Agent", "azure-vally-evals/check-azure-crates")
        .call()
        .map_err(|e| format!("request failed: {e}"))?;

    let body = response
        .into_string()
        .map_err(|e| format!("unable to read response body: {e}"))?;
    let json: JsonValue =
        serde_json::from_str(&body).map_err(|e| format!("invalid JSON response: {e}"))?;

    let crate_obj = json
        .get("crate")
        .and_then(|v| v.as_object())
        .ok_or_else(|| "missing 'crate' object in response".to_string())?;

    for key in ["max_stable_version", "max_version", "newest_version"] {
        if let Some(ver) = crate_obj.get(key).and_then(|v| v.as_str()) {
            if !ver.trim().is_empty() {
                return Ok(ver.to_string());
            }
        }
    }

    Err("no version field found in crates.io response".to_string())
}

fn resolve_workspace_member_manifests(
    root_manifest_path: &PathBuf,
    root_manifest: &Value,
) -> Option<(Vec<PathBuf>, Vec<String>)> {
    let mut manifests = Vec::new();
    let mut warnings = Vec::new();

    let workspace = root_manifest.get("workspace")?.as_table()?;
    let members = workspace.get("members")?.as_array()?;
    let root_dir = root_manifest_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));

    for member in members {
        let Some(member_path) = member.as_str() else {
            warnings.push("Ignoring non-string workspace member entry".to_string());
            continue;
        };

        if member_path.contains('*') || member_path.contains('?') || member_path.contains('[') {
            warnings.push(format!(
                "Skipping glob workspace member pattern '{member_path}' (only explicit member paths are supported)"
            ));
            continue;
        }

        let member_manifest = root_dir.join(member_path).join("Cargo.toml");
        if member_manifest.exists() {
            manifests.push(member_manifest);
        } else {
            warnings.push(format!(
                "Workspace member manifest not found: {}",
                member_manifest.display()
            ));
        }
    }

    Some((manifests, warnings))
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
