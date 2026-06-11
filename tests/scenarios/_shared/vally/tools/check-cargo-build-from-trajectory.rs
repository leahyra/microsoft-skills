#!/usr/bin/env -S cargo +nightly -Zscript
---
[package]
edition = "2021"

[dependencies]
serde_json = "1"
---

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Validates trajectory events in results.jsonl and fails when a cargo build
//! tool call (bash/powershell) has a corresponding failed tool_result.

use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    env, fs,
    path::PathBuf,
    process,
};

fn normalize_1_10(raw: u8) -> f64 {
    (raw as f64 - 1.0) / 9.0
}

#[derive(Clone, Debug)]
struct BuildCall {
    line: usize,
    event_id: Option<String>,
    call_id: Option<String>,
    tool_name: String,
    command: String,
    failed: bool,
    failure_reason: Option<String>,
    result_excerpt: Option<String>,
}

fn main() {
    let search_root = env::var("EVALUATE_WORKSPACE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let candidate_arg = env::args().nth(1);
    let Some(results_path) = resolve_results_path(candidate_arg, &search_root) else {
        emit_result(
            false,
            1,
            "No results.jsonl found",
            "incorrect",
            vec![format!(
                "No results.jsonl found under {}",
                search_root.display()
            )],
            json!({
                "search_root": search_root.display().to_string(),
                "error_kind": "missing_results_jsonl",
            }),
        );
        process::exit(0);
    };

    let text = match fs::read_to_string(&results_path) {
        Ok(t) => t,
        Err(e) => {
            emit_result(
                false,
                1,
                "Unable to read results.jsonl",
                "incorrect",
                vec![format!("Failed to read {}: {e}", results_path.display())],
                json!({
                    "results_jsonl": results_path.display().to_string(),
                    "error_kind": "read_failed",
                }),
            );
            process::exit(0);
        }
    };

    let mut trajectory_nodes = 0usize;
    let mut parse_failures = 0usize;
    let mut build_calls: Vec<BuildCall> = Vec::new();

    for (line_index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let value: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => {
                parse_failures += 1;
                continue;
            }
        };

        let mut events: Vec<&Value> = Vec::new();
        collect_trajectory_events(&value, &mut events);
        if events.is_empty() {
            continue;
        }

        trajectory_nodes += 1;

        let mut pending_set: HashSet<usize> = HashSet::new();
        let mut pending_order: Vec<usize> = Vec::new();
        let mut pending_by_call_id: HashMap<String, Vec<usize>> = HashMap::new();
        let mut pending_by_event_id: HashMap<String, Vec<usize>> = HashMap::new();

        for event in events {
            match event_type(event) {
                Some("tool_call") => {
                    let tool = tool_name(event).unwrap_or_default().to_string();
                    let command = tool_command(event).unwrap_or_default();

                    if is_shell_tool(&tool) && command.to_ascii_lowercase().contains("cargo build")
                    {
                        let idx = build_calls.len();
                        let event_id = event_id(event);
                        let call_id = call_id(event);

                        if let Some(ref id) = call_id {
                            pending_by_call_id.entry(id.clone()).or_default().push(idx);
                        }
                        if let Some(ref id) = event_id {
                            pending_by_event_id.entry(id.clone()).or_default().push(idx);
                        }

                        pending_set.insert(idx);
                        pending_order.push(idx);

                        build_calls.push(BuildCall {
                            line: line_index + 1,
                            event_id,
                            call_id,
                            tool_name: tool,
                            command,
                            failed: false,
                            failure_reason: None,
                            result_excerpt: None,
                        });
                    }
                }
                Some("tool_result") => {
                    let matched_idx = match_tool_result(
                        event,
                        &pending_set,
                        &pending_order,
                        &mut pending_by_call_id,
                        &mut pending_by_event_id,
                    );

                    let Some(idx) = matched_idx else {
                        continue;
                    };

                    pending_set.remove(&idx);

                    if let Some((reason, excerpt)) = tool_result_failure(event) {
                        if let Some(build) = build_calls.get_mut(idx) {
                            build.failed = true;
                            build.failure_reason = Some(reason);
                            build.result_excerpt = excerpt;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    let total_builds = build_calls.len();
    let failed_builds: Vec<&BuildCall> = build_calls.iter().filter(|b| b.failed).collect();
    let failed_count = failed_builds.len();

    let (passed, raw_score, label, evidence) = if total_builds == 0 {
        (
            true,
            7,
            "partially-correct",
            "No cargo build tool_call found in trajectory events".to_string(),
        )
    } else if failed_count == 0 {
        (
            true,
            10,
            "correct",
            "No failed cargo build tool_result detected for trajectory tool calls".to_string(),
        )
    } else {
        (
            false,
            scaled_failed_score(failed_count, total_builds),
            "incorrect",
            format!(
                "Detected {failed_count} failed cargo build(s) out of {total_builds} cargo build call(s)"
            ),
        )
    };

    let mut failures: Vec<String> = Vec::new();
    for failed in &failed_builds {
        failures.push(format!(
            "line {}: {} (call_id={})",
            failed.line,
            failed
                .failure_reason
                .clone()
                .unwrap_or_else(|| "failed cargo build".to_string()),
            failed
                .call_id
                .clone()
                .unwrap_or_else(|| "<none>".to_string())
        ));
    }

    if trajectory_nodes == 0 {
        failures.push("No trajectory node found in results.jsonl".to_string());
    }
    if parse_failures > 0 {
        failures.push(format!(
            "Skipped {parse_failures} malformed JSONL lines during parsing"
        ));
    }

    let failed_build_objects: Vec<Value> = failed_builds
        .iter()
        .map(|b| {
            json!({
                "line": b.line,
                "event_id": b.event_id,
                "call_id": b.call_id,
                "tool_name": b.tool_name,
                "command": b.command,
                "reason": b.failure_reason,
                "result_excerpt": b.result_excerpt,
            })
        })
        .collect();

    let build_call_objects: Vec<Value> = build_calls
        .iter()
        .map(|b| {
            json!({
                "line": b.line,
                "event_id": b.event_id,
                "call_id": b.call_id,
                "tool_name": b.tool_name,
                "command": b.command,
                "failed": b.failed,
            })
        })
        .collect();

    emit_result(
        passed,
        raw_score,
        &evidence,
        label,
        failures,
        json!({
            "results_jsonl": results_path.display().to_string(),
            "trajectory_nodes": trajectory_nodes,
            "build_calls_found": total_builds,
            "failed_build_count": failed_count,
            "failed_build_ratio": if total_builds > 0 { failed_count as f64 / total_builds as f64 } else { 0.0 },
            "parse_failures": parse_failures,
            "build_calls": build_call_objects,
            "failed_builds": failed_build_objects,
        }),
    );

    process::exit(0);
}

fn match_tool_result(
    event: &Value,
    pending_set: &HashSet<usize>,
    pending_order: &[usize],
    pending_by_call_id: &mut HashMap<String, Vec<usize>>,
    pending_by_event_id: &mut HashMap<String, Vec<usize>>,
) -> Option<usize> {
    if let Some(result_call_id) = call_id(event) {
        if let Some(idx) = take_pending_for_key(pending_by_call_id, &result_call_id, pending_set) {
            return Some(idx);
        }
    }

    if let Some(for_event_id) = result_for_id(event) {
        if let Some(idx) = take_pending_for_key(pending_by_event_id, &for_event_id, pending_set) {
            return Some(idx);
        }
    }

    pending_order
        .iter()
        .copied()
        .find(|idx| pending_set.contains(idx))
}

fn take_pending_for_key(
    map: &mut HashMap<String, Vec<usize>>,
    key: &str,
    pending_set: &HashSet<usize>,
) -> Option<usize> {
    let found = if let Some(indices) = map.get_mut(key) {
        while let Some(idx) = indices.first().copied() {
            indices.remove(0);
            if pending_set.contains(&idx) {
                return Some(idx);
            }
        }
        None
    } else {
        None
    };

    if map.get(key).map(|v| v.is_empty()).unwrap_or(false) {
        map.remove(key);
    }

    found
}

fn scaled_failed_score(failed_count: usize, total_builds: usize) -> u8 {
    if total_builds == 0 {
        return 7;
    }

    // 0 failed => 10. Higher failed ratio reduces score toward 1.
    let penalty = ((failed_count as f64 / total_builds as f64) * 9.0).ceil() as u8;
    let raw = 10u8.saturating_sub(penalty);
    raw.max(1)
}

fn emit_result(
    passed: bool,
    raw_score: u8,
    evidence: &str,
    label: &str,
    failures: Vec<String>,
    metadata: Value,
) {
    let summary = if passed {
        "Cargo build trajectory check passed"
    } else {
        "Cargo build trajectory check failed"
    };

    let mut metadata_obj = metadata;
    if let Some(map) = metadata_obj.as_object_mut() {
        map.insert("summary".to_string(), Value::String(summary.to_string()));
        map.insert(
            "failures".to_string(),
            Value::Array(failures.into_iter().map(Value::String).collect()),
        );
        map.insert("raw_score".to_string(), Value::from(raw_score));
    }

    let output = json!({
        "name": "cargo-build-trajectory-check",
        "kind": "code",
        "passed": passed,
        "score": normalize_1_10(raw_score),
        "evidence": evidence,
        "label": label,
        "metadata": metadata_obj,
    });

    println!("{}", output);
}

fn resolve_results_path(arg: Option<String>, search_root: &PathBuf) -> Option<PathBuf> {
    if let Some(path_arg) = arg {
        let candidate = PathBuf::from(&path_arg);
        if candidate.is_absolute() {
            return if candidate.exists() {
                Some(candidate)
            } else {
                None
            };
        }
        let joined = search_root.join(candidate);
        return if joined.exists() { Some(joined) } else { None };
    }

    find_results_jsonl(search_root)
}

fn find_results_jsonl(dir: &PathBuf) -> Option<PathBuf> {
    let candidate = dir.join("results.jsonl");
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
            if let Some(found) = find_results_jsonl(&path) {
                return Some(found);
            }
        }
    }

    None
}

fn is_trajectory_node(value: &Value) -> bool {
    if let Some(t) = value.get("type").and_then(|v| v.as_str()) {
        if t.eq_ignore_ascii_case("trajectory") {
            return true;
        }
    }
    if let Some(t) = value.get("node_type").and_then(|v| v.as_str()) {
        if t.eq_ignore_ascii_case("trajectory") {
            return true;
        }
    }

    value
        .get("node")
        .and_then(|n| n.get("type"))
        .and_then(|v| v.as_str())
        .map(|s| s.eq_ignore_ascii_case("trajectory"))
        .unwrap_or(false)
}

fn collect_trajectory_events<'a>(value: &'a Value, out: &mut Vec<&'a Value>) {
    if let Some(trajectory) = value.get("trajectory") {
        collect_event_objects(trajectory, out);
    }

    // Back-compat if the line itself is a trajectory node.
    if is_trajectory_node(value) {
        collect_event_objects(value, out);
    }
}

fn collect_event_objects<'a>(value: &'a Value, out: &mut Vec<&'a Value>) {
    match value {
        Value::Object(map) => {
            if let Some(events) = map.get("events").and_then(|v| v.as_array()) {
                for ev in events {
                    if ev.is_object() {
                        out.push(ev);
                    }
                }
            }

            for nested in map.values() {
                collect_event_objects(nested, out);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_event_objects(item, out);
            }
        }
        _ => {}
    }
}

fn event_type(event: &Value) -> Option<&str> {
    event
        .get("type")
        .and_then(|v| v.as_str())
        .or_else(|| event.get("event_type").and_then(|v| v.as_str()))
}

fn event_data(event: &Value) -> &Value {
    event.get("data").unwrap_or(event)
}

fn tool_name(event: &Value) -> Option<&str> {
    let data = event_data(event);
    data.get("toolName")
        .and_then(|v| v.as_str())
        .or_else(|| data.get("tool_name").and_then(|v| v.as_str()))
        .or_else(|| data.get("name").and_then(|v| v.as_str()))
        .or_else(|| event.get("name").and_then(|v| v.as_str()))
        .or_else(|| event.get("tool_name").and_then(|v| v.as_str()))
}

fn tool_command(event: &Value) -> Option<String> {
    let data = event_data(event);

    data.get("command")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            data.get("arguments")
                .and_then(command_from_value)
                .or_else(|| data.get("args").and_then(command_from_value))
        })
        .or_else(|| {
            event
                .get("command")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .or_else(|| event.get("arguments").and_then(command_from_value))
        .or_else(|| event.get("args").and_then(command_from_value))
}

fn command_from_value(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Object(map) => map
            .get("command")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                map.get("commandLine")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .or_else(|| {
                map.get("command_line")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            }),
        _ => None,
    }
}

fn is_shell_tool(name: &str) -> bool {
    name.eq_ignore_ascii_case("powershell") || name.eq_ignore_ascii_case("bash")
}

fn event_id(event: &Value) -> Option<String> {
    let data = event_data(event);
    data.get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            data.get("eventId")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .or_else(|| {
            data.get("event_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
}

fn call_id(event: &Value) -> Option<String> {
    let data = event_data(event);
    data.get("toolCallId")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            data.get("tool_call_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .or_else(|| {
            data.get("call_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
}

fn result_for_id(event: &Value) -> Option<String> {
    let data = event_data(event);
    data.get("forEventId")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            data.get("for_event_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .or_else(|| {
            data.get("result_for")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
}

fn tool_result_failure(event: &Value) -> Option<(String, Option<String>)> {
    let data = event_data(event);

    if let Some(success) = data
        .get("success")
        .and_then(|v| v.as_bool())
        .or_else(|| event.get("success").and_then(|v| v.as_bool()))
    {
        if !success {
            return Some((
                "tool_result.success=false".to_string(),
                tool_result_text(data).map(|s| truncate_text(&s, 280)),
            ));
        }
    }

    if let Some(exit_code) = data
        .get("exitCode")
        .and_then(|v| v.as_i64())
        .or_else(|| data.get("exit_code").and_then(|v| v.as_i64()))
        .or_else(|| event.get("exit_code").and_then(|v| v.as_i64()))
    {
        if exit_code != 0 {
            return Some((
                format!("non-zero exit code ({exit_code})"),
                tool_result_text(data).map(|s| truncate_text(&s, 280)),
            ));
        }
    }

    if let Some(status) = data
        .get("status")
        .and_then(|v| v.as_str())
        .or_else(|| event.get("status").and_then(|v| v.as_str()))
    {
        let s = status.to_ascii_lowercase();
        if s.contains("fail") || s.contains("error") || s == "nonzero_exit" {
            return Some((
                format!("tool_result.status={status}"),
                tool_result_text(data).map(|s| truncate_text(&s, 280)),
            ));
        }
    }

    if let Some(error_text) = data
        .get("error")
        .and_then(|v| v.as_str())
        .or_else(|| event.get("error").and_then(|v| v.as_str()))
    {
        if !error_text.is_empty() {
            return Some((
                "tool_result.error is present".to_string(),
                Some(truncate_text(error_text, 280)),
            ));
        }
    }

    if let Some(result_text) = tool_result_text(data) {
        let lower = result_text.to_ascii_lowercase();
        if lower.contains("didn't exit successfully")
            || lower.contains("did not exit successfully")
            || lower.contains("error: process didn't exit successfully")
            || has_nonzero_exit_marker(&lower)
        {
            return Some((
                "non-zero process exit marker found in tool result content".to_string(),
                Some(truncate_text(&result_text, 280)),
            ));
        }
    }

    None
}

fn tool_result_text(data: &Value) -> Option<String> {
    let result = data.get("result")?;
    result
        .get("detailedContent")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            result
                .get("content")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
}

fn has_nonzero_exit_marker(text: &str) -> bool {
    if text.contains("<exited with exit code") {
        return !text.contains("<exited with exit code 0>");
    }

    for code in 1..=9 {
        let marker_a = format!("exit code: {code}");
        let marker_b = format!("exit code {code}");
        if text.contains(&marker_a) || text.contains(&marker_b) {
            return true;
        }
    }

    false
}

fn truncate_text(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }

    let mut end = max_len;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &s[..end])
}
