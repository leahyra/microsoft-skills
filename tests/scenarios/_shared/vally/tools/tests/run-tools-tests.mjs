#!/usr/bin/env node

import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import https from "node:https";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const toolsDir = resolve(__dirname, "..");
const fixturesDir = join(__dirname, "fixtures");

function fail(message) {
  console.error(`FAIL: ${message}`);
  process.exitCode = 1;
}

function pass(message) {
  console.log(`PASS: ${message}`);
}

function assert(condition, message) {
  if (!condition) {
    fail(message);
    return false;
  }
  return true;
}

function runCommand(command, args, options = {}) {
  const result = spawnSync(command, args, {
    encoding: "utf8",
    ...options,
  });

  if (result.error) {
    throw result.error;
  }

  return result;
}

function parseJsonFromOutput(stdout) {
  const lines = (stdout || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const line = lines[i];
    if (!line.startsWith("{")) {
      continue;
    }
    try {
      return JSON.parse(line);
    } catch {
      // Continue scanning older lines.
    }
  }

  throw new Error(`No JSON result found in stdout:\n${stdout}`);
}

function runCargoScript(toolFileName, extraArgs = [], env = {}) {
  const toolPath = join(toolsDir, toolFileName);
  const result = runCommand(
    "cargo",
    ["+nightly", "-Zscript", toolPath, ...extraArgs],
    {
      cwd: toolsDir,
      env: { ...process.env, ...env },
    },
  );

  if (result.status !== 0) {
    throw new Error(
      `cargo script failed for ${toolFileName} (exit ${result.status})\nSTDOUT:\n${result.stdout}\nSTDERR:\n${result.stderr}`,
    );
  }

  return parseJsonFromOutput(result.stdout);
}

function fetchJson(url) {
  return new Promise((resolvePromise, rejectPromise) => {
    const request = https.get(
      url,
      {
        headers: {
          "User-Agent": "azure-vally-evals/tools-tests",
        },
      },
      (response) => {
        let body = "";
        response.on("data", (chunk) => {
          body += chunk;
        });
        response.on("end", () => {
          if (response.statusCode !== 200) {
            rejectPromise(new Error(`HTTP ${response.statusCode} from ${url}`));
            return;
          }

          try {
            resolvePromise(JSON.parse(body));
          } catch (error) {
            rejectPromise(error);
          }
        });
      },
    );

    request.on("error", rejectPromise);
  });
}

async function getLatestAzureVersions() {
  const crates = ["azure_core", "azure_identity", "azure_storage_blob"];
  const versions = {};

  for (const crateName of crates) {
    const payload = await fetchJson(`https://crates.io/api/v1/crates/${crateName}`);
    const crateInfo = payload?.crate || {};
    const latest =
      crateInfo.max_stable_version || crateInfo.max_version || crateInfo.newest_version;

    if (!latest) {
      throw new Error(`No latest version found for ${crateName}`);
    }

    versions[crateName] = latest;
  }

  return versions;
}

function writeWorkspaceProject(rootPath, cargoToml, mainRs) {
  mkdirSync(join(rootPath, "src"), { recursive: true });
  writeFileSync(join(rootPath, "Cargo.toml"), cargoToml, "utf8");
  writeFileSync(join(rootPath, "src", "main.rs"), mainRs, "utf8");
}

function testAsyncRuntimeTool() {
  const goodWorkspace = join(fixturesDir, "async-runtime", "pass");
  const badWorkspace = join(fixturesDir, "async-runtime", "fail-block-on");

  const good = runCargoScript("check-async-runtime.rs", [], {
    EVALUATE_WORKSPACE: goodWorkspace,
  });
  assert(good.passed === true, "check-async-runtime.rs should pass for async fixture");

  const bad = runCargoScript("check-async-runtime.rs", [], {
    EVALUATE_WORKSPACE: badWorkspace,
  });
  assert(
    bad.passed === false,
    "check-async-runtime.rs should fail for block_on fixture",
  );
  assert(
    String(bad.evidence).includes("block_on") ||
      String(bad.evidence).includes("Missing #[tokio::main]"),
    "check-async-runtime.rs failure should mention async rule violations",
  );

  pass("check-async-runtime.rs");
}

function testTokenCredentialTool() {
  const goodWorkspace = join(fixturesDir, "token-credential", "pass");
  const badWorkspace = join(fixturesDir, "token-credential", "fail-banned");

  const good = runCargoScript("check-token-credential.rs", [], {
    EVALUATE_WORKSPACE: goodWorkspace,
  });
  assert(
    good.passed === true,
    "check-token-credential.rs should pass for valid credential fixture",
  );

  const bad = runCargoScript("check-token-credential.rs", [], {
    EVALUATE_WORKSPACE: badWorkspace,
  });
  assert(
    bad.passed === false,
    "check-token-credential.rs should fail for banned credential fixture",
  );
  assert(
    String(bad.evidence).includes("DefaultAzureCredential"),
    "check-token-credential.rs failure should mention DefaultAzureCredential",
  );

  pass("check-token-credential.rs");
}

function testNoSecretsTool() {
  const goodWorkspace = join(fixturesDir, "no-secrets", "pass");
  const badWorkspace = join(fixturesDir, "no-secrets", "fail-key");

  const good = runCargoScript("check-no-secrets.rs", [], {
    EVALUATE_WORKSPACE: goodWorkspace,
  });
  assert(good.passed === true, "check-no-secrets.rs should pass for clean fixture");

  const bad = runCargoScript("check-no-secrets.rs", [], {
    EVALUATE_WORKSPACE: badWorkspace,
  });
  assert(bad.passed === false, "check-no-secrets.rs should fail for secret fixture");
  assert(
    String(bad.evidence).toLowerCase().includes("accountkey") ||
      String(bad.evidence).toLowerCase().includes("connection string"),
    "check-no-secrets.rs failure should mention a detected secret pattern",
  );

  pass("check-no-secrets.rs");
}

async function testAzureCratesTool() {
  const tempRoot = mkdtempSync(join(tmpdir(), "azure-crates-tool-test-"));

  try {
    const latest = await getLatestAzureVersions();

    const semverWorkspace = join(tempRoot, "semver-pass");
    const outdatedWorkspace = join(tempRoot, "outdated-fail");

    writeWorkspaceProject(
      semverWorkspace,
      `[package]\nname = "azure-crates-pass"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\nazure_core = "^${latest.azure_core}"\nazure_identity = "^${latest.azure_identity}"\nazure_storage_blob = "^${latest.azure_storage_blob}"\nfutures = "0.3"\ntokio = { version = "1", features = ["full"] }\n`,
      `#[tokio::main]\nasync fn main() {\n    let _ = futures::future::ready(()).await;\n}\n`,
    );

    writeWorkspaceProject(
      outdatedWorkspace,
      `[package]\nname = "azure-crates-fail"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\nazure_core = "0.34"\nazure_identity = "0.34"\nazure_storage_blob = "0.11"\nfutures = "0.3"\ntokio = { version = "1", features = ["full"] }\n`,
      `fn main() {}\n`,
    );

    const semverPass = runCargoScript("check-azure-crates.rs", [join(semverWorkspace, "Cargo.toml")]);
    assert(
      semverPass.passed === true,
      "check-azure-crates.rs should pass for semver-equivalent latest versions",
    );

    const outdatedFail = runCargoScript("check-azure-crates.rs", [join(outdatedWorkspace, "Cargo.toml")]);
    assert(
      outdatedFail.passed === false,
      "check-azure-crates.rs should fail for clearly outdated azure crate versions",
    );
    assert(
      outdatedFail.metadata?.has_outdated_azure_crate === true,
      "check-azure-crates.rs should set has_outdated_azure_crate for outdated versions",
    );

    pass("check-azure-crates.rs");
  } finally {
    rmSync(tempRoot, { recursive: true, force: true });
  }
}

async function main() {
  try {
    console.log("Running tools regression tests...");
    testAsyncRuntimeTool();
    testTokenCredentialTool();
    testNoSecretsTool();
    await testAzureCratesTool();

    if (process.exitCode && process.exitCode !== 0) {
      console.error("\nOne or more tests failed.");
      process.exit(process.exitCode);
    }

    console.log("\nAll tool tests passed.");
  } catch (error) {
    console.error(error?.stack || String(error));
    process.exit(1);
  }
}

main();
