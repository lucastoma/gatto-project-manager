---
trigger: always_on
---

## "repomix-docker-linux"

When using the "repomix-docker-linux" MCP server, always provide Linux-style absolute paths (e.g. /workspace/...) as arguments for codebase directory or file paths. The root of the mounted workspace is always /workspace inside the container, regardless of the host system path.
