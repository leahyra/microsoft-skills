// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * Vally grader plugin: rust-cargo-build-failure-check (Rust-specific)
 *
 * Inspects the trajectory for shell tool_call events containing "cargo build"
 * and their corresponding tool_result events. Detects failed Rust builds,
 * extracts Rust compiler errors (E0XXX codes), and scales the score by failure
 * ratio: more failed builds => lower score.
 *
 * Usage in eval.yaml:
 *   graders:
 *     - type: rust-cargo-build-failure-check
 *
 * Load via CLI:
 *   vally eval -e eval.yaml --grader-plugin ./path/to/rust-cargo-build-failure
 */

import type {
  Grader,
  GraderInput,
  GraderMetadata,
  GraderRegistry,
  GraderResult,
  TrajectoryEvent,
} from "@microsoft/vally";

// ── Types ──────────────────────────────────────────────────────────────

interface CompilerError {
  errorCode: string;
  message: string;
  lineInfo?: string;
}

interface ErrorException {
  error_code: string;
  message_pattern: string;
  action: "ignore_for_fail";
}

interface GraderConfig {
  collect_compiler_errors?: boolean;
  emit_in_metadata?: boolean;
  error_exceptions?: ErrorException[];
}

interface BuildCall {
  toolCallId: string;
  toolName: string;
  command: string;
  failed: boolean;
  failureReason?: string;
  resultExcerpt?: string;
  compilerErrors?: CompilerError[];
}

// ── Helpers ────────────────────────────────────────────────────────────

const SHELL_TOOLS = new Set(["powershell", "bash"]);

function isShellTool(name: string): boolean {
  return SHELL_TOOLS.has(name.toLowerCase());
}

/** Normalize a 1–10 raw score to 0–1 range. */
function normalize(raw: number): number {
  return (raw - 1) / 9;
}

/**
 * Compute a raw score (1–10) based on failure ratio.
 * 0 failed => 10. Higher failure ratio => lower score, minimum 1.
 */
function scaledScore(failedCount: number, totalBuilds: number): number {
  if (totalBuilds === 0) return 7;
  const penalty = Math.ceil((failedCount / totalBuilds) * 9);
  return Math.max(1, 10 - penalty);
}

/** Truncate text to maxLen chars. */
function truncate(s: string, maxLen = 280): string {
  if (s.length <= maxLen) return s;
  return s.slice(0, maxLen) + "...";
}

/**
 * Extract the command string from a tool_call event's arguments.
 */
function extractCommand(args: Record<string, unknown> | undefined): string {
  if (!args) return "";
  if (typeof args["command"] === "string") return args["command"];
  if (typeof args["commandLine"] === "string") return args["commandLine"];
  if (typeof args["command_line"] === "string") return args["command_line"];
  return "";
}

/**
 * Extract text content from a tool_result's result field.
 * The result field is typed as `unknown` — it may contain
 * { content: string } or { detailedContent: string }.
 */
function resultContent(result: unknown): string | undefined {
  if (result == null || typeof result !== "object") return undefined;
  const r = result as Record<string, unknown>;
  if (typeof r["detailedContent"] === "string") return r["detailedContent"];
  if (typeof r["content"] === "string") return r["content"];
  return undefined;
}

/** Check for non-zero exit code markers in output text. */
function hasNonzeroExitMarker(text: string): boolean {
  if (text.includes("<exited with exit code")) {
    return !text.includes("<exited with exit code 0>");
  }
  for (let code = 1; code <= 9; code++) {
    if (
      text.includes(`exit code: ${code}`) ||
      text.includes(`exit code ${code}`)
    ) {
      return true;
    }
  }
  return false;
}

/** Check if a tool_result event represents a failure.
 * Returns [reason, excerpt] if failed, or null if succeeded.
 */
function detectFailure(
  event: TrajectoryEvent,
): [reason: string, excerpt: string | undefined] | null {
  if (event.type !== "tool_result") return null;
  const data = event.data;

  // Explicit success=false
  if (data.success === false) {
    const content = resultContent(data.result);
    return ["tool_result.success=false", content ? content : undefined];
  }

  // Check result content for failure markers
  const content = resultContent(data.result);
  if (content) {
    const lower = content.toLowerCase();
    if (
      lower.includes("didn't exit successfully") ||
      lower.includes("did not exit successfully") ||
      lower.includes("error: process didn't exit successfully") ||
      hasNonzeroExitMarker(lower)
    ) {
      return [
        "non-zero process exit marker found in tool result content",
        truncate(content),
      ];
    }
  }

  return null;
}

/**
 * Extract Rust compiler errors from cargo output.
 * Looks for patterns like: error[E0277]: ...
 */
function extractCompilerErrors(text: string): CompilerError[] {
  const errors: CompilerError[] = [];
  // Match: error[EXXX]: message
  const errorPattern = /error\[([A-Z]\d+)\]:\s*(.+?)(?:\n|$)/g;
  let match;

  while ((match = errorPattern.exec(text)) !== null) {
    const errorCode = match[1];
    const message = match[2].trim();
    errors.push({
      errorCode,
      message,
    });
  }

  return errors;
}

/**
 * Check if a compiler error matches an exception rule and should be ignored for failure.
 */
function matchesException(
  error: CompilerError,
  exception: ErrorException,
): boolean {
  if (error.errorCode !== exception.error_code) return false;
  const pattern = exception.message_pattern;
  return error.message.includes(pattern);
}

/**
 * Check if any compiler errors should block grader failure (none matched an ignore_for_fail exception).
 * Returns true if there are unignored compiler errors, false otherwise.
 */
function hasUnignoredCompilerErrors(
  errors: CompilerError[],
  exceptions?: ErrorException[],
): boolean {
  if (errors.length === 0) return false;
  if (!exceptions || exceptions.length === 0) return true;

  for (const error of errors) {
    let ignored = false;
    for (const exception of exceptions) {
      if (
        exception.action === "ignore_for_fail" &&
        matchesException(error, exception)
      ) {
        ignored = true;
        break;
      }
    }
    if (!ignored) return true; // Found an unignored error
  }
  return false; // All errors were ignored
}

// ── Grader ─────────────────────────────────────────────────────────────

class CargoBuildTrajectoryGrader implements Grader {
  metadata: GraderMetadata = {
    name: "rust-cargo-build-failure-check",
    description:
      "Rust-specific grader: checks trajectory for failed cargo build tool calls, extracts Rust compiler errors (E0XXX), and scales score by failure ratio",
    behavior: { execution: "single", requiresWorkspace: false },
    determinism: "static",
    portability: "t3a-scenario",
    reference: "reference-free",
    temporalScope: "trajectory-level",
    costProfile: "free",
  };

  async grade(input: GraderInput): Promise<GraderResult> {
    const trajectory = input.trajectory;
    if (!trajectory) {
      return {
        name: this.metadata.name,
        kind: "code",
        passed: false,
        score: 0,
        evidence: "No trajectory provided",
        label: "incorrect",
        metadata: { error_kind: "missing_trajectory" },
      };
    }

    // Parse config
    const config: GraderConfig = (input.config || {}) as GraderConfig;
    const collectCompilerErrors = config.collect_compiler_errors !== false; // default: true
    const emitInMetadata = config.emit_in_metadata !== false; // default: true
    const errorExceptions = config.error_exceptions || [];

    const events = trajectory.events;

    // Track build calls and match results.
    const buildCalls: BuildCall[] = [];
    const pendingByCallId = new Map<string, number>();
    const pendingOrder: number[] = [];

    for (const event of events) {
      if (event.type === "tool_call") {
        const { toolName, toolCallId, arguments: args } = event.data;
        const command = extractCommand(args);

        if (
          isShellTool(toolName) &&
          command.toLowerCase().includes("cargo build")
        ) {
          const idx = buildCalls.length;

          buildCalls.push({
            toolCallId,
            toolName,
            command,
            failed: false,
          });

          if (toolCallId) {
            pendingByCallId.set(toolCallId, idx);
          }
          pendingOrder.push(idx);
        }
        continue;
      }

      if (event.type === "tool_result") {
        const { toolCallId } = event.data;

        // Match to a pending build call
        let matchedIdx: number | undefined;
        if (toolCallId && pendingByCallId.has(toolCallId)) {
          matchedIdx = pendingByCallId.get(toolCallId);
          pendingByCallId.delete(toolCallId);
        } else if (pendingOrder.length > 0) {
          // Fallback: match to oldest pending build
          matchedIdx = pendingOrder[0];
        } else {
          continue;
        }

        if (matchedIdx === undefined) continue;

        // Remove from pending
        const orderIdx = pendingOrder.indexOf(matchedIdx);
        if (orderIdx >= 0) pendingOrder.splice(orderIdx, 1);

        const build = buildCalls[matchedIdx];
        if (!build) continue;

        // Extract compiler errors if enabled
        const content = resultContent(event.data.result);
        let compilerErrors: CompilerError[] = [];
        if (collectCompilerErrors && content) {
          compilerErrors = extractCompilerErrors(content);
        }

        if (compilerErrors.length > 0) {
          build.compilerErrors = compilerErrors;
        }

        // Detect failure
        const failure = detectFailure(event);
        if (failure) {
          // Check if this failure should be ignored due to error exceptions
          const shouldIgnore =
            compilerErrors.length > 0 &&
            !hasUnignoredCompilerErrors(compilerErrors, errorExceptions);

          if (!shouldIgnore) {
            build.failed = true;
            build.failureReason = failure[0];
            build.resultExcerpt = failure[1];
          }
        }
      }
    }

    const totalBuilds = buildCalls.length;
    const failedBuilds = buildCalls.filter((b) => b.failed);
    const failedCount = failedBuilds.length;

    let passed: boolean;
    let rawScore: number;
    let label: string;
    let evidence: string;

    if (totalBuilds === 0) {
      passed = true;
      rawScore = 7;
      label = "partially-correct";
      evidence = "No cargo build tool_call found in trajectory events";
    } else if (failedCount === 0) {
      passed = true;
      rawScore = 10;
      label = "correct";
      evidence =
        "No failed cargo build tool_result detected for trajectory tool calls";
    } else {
      passed = false;
      rawScore = scaledScore(failedCount, totalBuilds);
      label = "incorrect";
      evidence = `Detected ${failedCount} failed cargo build(s) out of ${totalBuilds} cargo build call(s)`;
    }

    const failures = failedBuilds.map(
      (b) =>
        `${b.failureReason ?? "failed cargo build"} (call_id=${b.toolCallId || "<none>"})`,
    );

    const metadata: Record<string, unknown> = {
      build_calls_found: totalBuilds,
      failed_build_count: failedCount,
      failed_build_ratio: totalBuilds > 0 ? failedCount / totalBuilds : 0,
      raw_score: rawScore,
      summary: passed
        ? "Cargo build trajectory check passed"
        : "Cargo build trajectory check failed",
      failures,
      build_calls: buildCalls.map((b) => ({
        tool_call_id: b.toolCallId,
        tool_name: b.toolName,
        command: b.command,
        failed: b.failed,
      })),
      failed_builds: failedBuilds.map((b) => ({
        tool_call_id: b.toolCallId,
        tool_name: b.toolName,
        command: b.command,
        reason: b.failureReason,
        result_excerpt: b.resultExcerpt,
      })),
    };

    // Emit compiler errors in metadata if configured
    if (emitInMetadata) {
      const allCompilerErrors: CompilerError[] = [];
      const buildCallsWithErrors = buildCalls.filter(
        (b) => b.compilerErrors && b.compilerErrors.length > 0,
      );

      for (const build of buildCallsWithErrors) {
        if (build.compilerErrors) {
          allCompilerErrors.push(...build.compilerErrors);
        }
      }

      if (allCompilerErrors.length > 0) {
        metadata.compiler_errors_collected = true;
        metadata.compiler_errors = allCompilerErrors;
        metadata.compiler_error_count = allCompilerErrors.length;
      }
    }

    return {
      name: this.metadata.name,
      kind: "code",
      passed,
      score: normalize(rawScore),
      evidence,
      label,
      metadata,
    };
  }
}

// ── Plugin entry point ─────────────────────────────────────────────────

/**
 * Called by Vally's loadGraderPlugin().
 */
export function registerGraders(registry: GraderRegistry): void {
  registry.register(new CargoBuildTrajectoryGrader());
}
