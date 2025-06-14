---
trigger: always_on
---

## "repomix-docker-linux" and "filesystem" paths

When using the "repomix-docker-linux" or "filesystem" use only relative paths. Use also only linux style paths as all MCP are docker mounted.

vs code project root is accesible at {"path": "/workspace"}

any file is accessible at {"path": "/workspace/file-path"}
