---
trigger: always_on
---

## "repomix-docker-linux" and "filesystem" paths

Use alwyas filesystem to edit and read files if possible.

When using the "repomix-docker-linux" or "filesystem" use only relative paths. Use also only linux style paths as all MCP are docker mounted.

vs code project root is accesible at {"path": "/workspace"}

any file is accessible at {"path": "/workspace/file-path"}

## Some important paths (relative)

/workspace/app/algorithms/algorithm_01_palette

/workspace/app/algorithms/algorithm_05_lab_transfer

/workspace/app/webview

/workspace/app/core
