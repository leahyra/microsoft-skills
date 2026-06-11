import { describe, expect, it } from "vitest";

import { registerGraders } from "../scenarios/_shared/vally/grader-plugins/rust-cargo-build-failure/index.js";

type TestGrader = {
  grade: (input: unknown) => Promise<{
    passed: boolean;
    score: number;
    metadata?: Record<string, unknown>;
  }>;
};

function getGrader(): TestGrader {
  const registered: TestGrader[] = [];
  const registry = {
    register(grader: TestGrader) {
      registered.push(grader);
    },
  };

  registerGraders(registry as never);

  const grader = registered[0];
  if (!grader) {
    throw new Error("cargo-build-trajectory grader was not registered");
  }

  return grader;
}

function makeInput(
  resultContent: string,
  config?: Record<string, unknown>,
): unknown {
  return {
    config,
    trajectory: {
      events: [
        {
          type: "tool_call",
          data: {
            toolName: "powershell",
            toolCallId: "call-1",
            arguments: { command: "cargo build" },
          },
        },
        {
          type: "tool_result",
          data: {
            toolCallId: "call-1",
            success: false,
            result: { content: resultContent },
          },
        },
      ],
    },
  };
}

describe("rust-cargo-build-failure grader", () => {
  it("emits compiler errors in metadata", async () => {
    const grader = getGrader();

    const result = await grader.grade(
      makeInput("error[E0425]: cannot find value `x` in this scope"),
    );

    expect(result.passed).toBe(false);
    expect(result.metadata?.compiler_errors_collected).toBe(true);
    expect(result.metadata?.compiler_error_count).toBe(1);

    const compilerErrors = result.metadata?.compiler_errors as Array<{
      errorCode: string;
      message: string;
    }>;

    expect(compilerErrors[0]?.errorCode).toBe("E0425");
    expect(compilerErrors[0]?.message).toContain("cannot find value");
  });

  it("ignores matching E0277 Option<std::string::String> error for failure", async () => {
    const grader = getGrader();

    const result = await grader.grade(
      makeInput(
        "error[E0277]: the trait bound `Option<std::string::String>: Something` is not satisfied",
        {
          collect_compiler_errors: true,
          emit_in_metadata: true,
          error_exceptions: [
            {
              error_code: "E0277",
              message_pattern: "Option<std::string::String>",
              action: "ignore_for_fail",
            },
          ],
        },
      ),
    );

    expect(result.passed).toBe(true);
    expect(result.score).toBe(1);
    expect(result.metadata?.failed_build_count).toBe(0);
    expect(result.metadata?.compiler_error_count).toBe(1);
  });

  it("does not ignore non-matching compiler errors", async () => {
    const grader = getGrader();

    const result = await grader.grade(
      makeInput(
        "error[E0277]: the trait bound `Vec<u8>: Display` is not satisfied",
        {
          collect_compiler_errors: true,
          emit_in_metadata: true,
          error_exceptions: [
            {
              error_code: "E0277",
              message_pattern: "Option<std::string::String>",
              action: "ignore_for_fail",
            },
          ],
        },
      ),
    );

    expect(result.passed).toBe(false);
    expect(result.metadata?.failed_build_count).toBe(1);
    expect(result.metadata?.compiler_error_count).toBe(1);
  });

  it("does not emit compiler errors when emit_in_metadata is false", async () => {
    const grader = getGrader();

    const result = await grader.grade(
      makeInput("error[E0425]: cannot find value `x` in this scope", {
        collect_compiler_errors: true,
        emit_in_metadata: false,
      }),
    );

    expect(result.passed).toBe(false);
    expect(result.metadata?.compiler_errors_collected).toBeUndefined();
    expect(result.metadata?.compiler_errors).toBeUndefined();
    expect(result.metadata?.compiler_error_count).toBeUndefined();
  });

  it("does not collect compiler errors when collect_compiler_errors is false", async () => {
    const grader = getGrader();

    const result = await grader.grade(
      makeInput("error[E0425]: cannot find value `x` in this scope", {
        collect_compiler_errors: false,
        emit_in_metadata: true,
      }),
    );

    expect(result.passed).toBe(false);
    expect(result.metadata?.compiler_errors_collected).toBeUndefined();
    expect(result.metadata?.compiler_errors).toBeUndefined();
    expect(result.metadata?.compiler_error_count).toBeUndefined();
  });
});
