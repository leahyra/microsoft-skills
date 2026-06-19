import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    // Keep discovery scoped to this repo's tests so vendored dependency tests are not executed.
    include: ["harness/**/*.test.ts", "plugins/**/*.test.ts"],
    exclude: [
      "**/node_modules/**",
      "**/dist/**",
      "**/.venv/**",
      "**/__pycache__/**",
      "**/pnpm-store/**",
    ],
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "html"],
      include: ["harness/**/*.ts"],
      exclude: ["harness/**/*.test.ts", "**/*.d.ts"],
    },
    testTimeout: 30000,
    hookTimeout: 10000,
  },
});
