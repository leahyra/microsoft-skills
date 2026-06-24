#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const DEFAULT_CORE_GRADERS = [
  "Official Azure SDK Crate Selection",
  "TokenCredential Authentication",
  "Async-First with Tokio Runtime",
  "No Hardcoded Secrets",
];

function parseArgs(argv) {
  let resultsDir = "";
  let coreGraders = [...DEFAULT_CORE_GRADERS];

  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--results-dir" && i + 1 < argv.length) {
      resultsDir = argv[++i];
    } else if (arg === "--core-graders" && i + 1 < argv.length) {
      coreGraders = argv[++i]
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
    } else if (arg === "-h" || arg === "--help") {
      printHelp();
      process.exit(0);
    }
  }

  if (!resultsDir) {
    console.error("ERROR: --results-dir is required");
    printHelp();
    process.exit(2);
  }

  return { resultsDir, coreGraders };
}

function printHelp() {
  console.log(`Usage:
  node tests/scenarios/_shared/vally/tools/assert-core-graders.mjs \\
    --results-dir <vally-results-or-experiment-dir> \\
    [--core-graders "Official Azure SDK Crate Selection,TokenCredential Authentication,Async-First with Tokio Runtime,No Hardcoded Secrets"]
`);
}

function findResultsJsonlFiles(rootDir) {
  const files = [];

  function walk(dir) {
    let entries;
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
      } else if (entry.isFile() && entry.name === "results.jsonl") {
        files.push(fullPath);
      }
    }
  }

  walk(rootDir);
  return files;
}

function parseJsonl(filePath) {
  const content = fs.readFileSync(filePath, "utf8");
  return content
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    })
    .filter(Boolean);
}

function main() {
  const { resultsDir, coreGraders } = parseArgs(process.argv);
  const absResultsDir = path.resolve(resultsDir);

  if (!fs.existsSync(absResultsDir)) {
    console.error(`ERROR: results directory does not exist: ${absResultsDir}`);
    process.exit(2);
  }

  const resultsFiles = findResultsJsonlFiles(absResultsDir);
  if (resultsFiles.length === 0) {
    console.error(`ERROR: no results.jsonl files found under ${absResultsDir}`);
    process.exit(2);
  }

  const failures = [];
  let trialCount = 0;

  for (const resultsFile of resultsFiles) {
    const records = parseJsonl(resultsFile);
    for (const record of records) {
      if (record.type !== "trial-result") {
        continue;
      }

      trialCount += 1;
      const details = Array.isArray(record?.gradeResult?.details)
        ? record.gradeResult.details
        : [];

      for (const coreName of coreGraders) {
        const coreDetail = details.find((d) => d?.name === coreName);
        if (!coreDetail) {
          failures.push({
            file: resultsFile,
            itemId: record.itemId || "<unknown-item>",
            grader: coreName,
            reason: "missing core grader result",
          });
          continue;
        }

        if (coreDetail.passed !== true) {
          failures.push({
            file: resultsFile,
            itemId: record.itemId || "<unknown-item>",
            grader: coreName,
            reason:
              typeof coreDetail.evidence === "string" && coreDetail.evidence.trim().length > 0
                ? coreDetail.evidence
                : `core grader failed (score=${coreDetail.score ?? "n/a"})`,
          });
        }
      }
    }
  }

  if (trialCount === 0) {
    console.error(`ERROR: no trial-result records found under ${absResultsDir}`);
    process.exit(2);
  }

  if (failures.length > 0) {
    console.error(`Core grader gate failed: ${failures.length} issue(s) across ${trialCount} trial(s).`);
    for (const f of failures) {
      console.error(
        `- ${f.grader} :: ${f.reason} :: item=${f.itemId} :: file=${f.file}`
      );
    }
    process.exit(1);
  }

  console.log(
    `Core grader gate passed: all ${coreGraders.length} core graders passed across ${trialCount} trial(s).`
  );
}

main();
