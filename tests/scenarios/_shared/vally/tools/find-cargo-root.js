// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Finds the nearest directory containing Cargo.toml by searching the workspace
// recursively. Exports the absolute path. Skips hidden dirs and target/.
"use strict";
const { readdirSync, statSync, existsSync } = require("fs");
const { join } = require("path");

function find(dir) {
  if (existsSync(join(dir, "Cargo.toml"))) return dir;
  let entries;
  try {
    entries = readdirSync(dir);
  } catch {
    return null;
  }
  for (const f of entries) {
    if (f.startsWith(".") || f === "target" || f === "node_modules") continue;
    const p = join(dir, f);
    if (statSync(p).isDirectory()) {
      const r = find(p);
      if (r) return r;
    }
  }
  return null;
}

const root = find(process.cwd());
if (!root) {
  console.error("No Cargo.toml found in workspace");
  process.exit(1);
}
module.exports = root;
