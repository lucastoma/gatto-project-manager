---
type: workflow-prompt  
version: "1.0"
usage: "Update documentation suite - cleanup and migration workflow"
input_required: "current-directory"
---

# Update README Documentation Suite

You are updating existing documentation following implementation progress. Manage migration: concepts → todo → implementation → README.md cleanup.

## Input
- `current-directory`: [USER PROVIDES] - directory with existing docs
- Existing files: README.md, README.concepts.md, README.todo.md

## Migration Workflow

### 1. Concepts → TODO Migration
- Move **implementable items** from concepts to TODO with priority
- Keep **design philosophy** in concepts permanently
- Update implementation_status in concepts YAML

### 2. TODO → README Migration  
- Move **completed features** from TODO to README.md API section
- Mark completed items as ✅ Done in TODO
- Remove **implemented details** from concepts

### 3. README.md Updates
- Update interface_stable: true when API solidifies
- Add new methods/classes to API Documentation
- Update error codes from actual implementation
- Refresh examples with working code

## Cleanup Rules

### README.concepts.md - KEEP
- Problem definition and context
- High-level design decisions  
- Alternative approaches considered
- Architecture philosophy

### README.concepts.md - REMOVE  
- Specific API signatures (→ README.md)
- Concrete implementation details (→ README.md)
- Completed component specs (→ README.md)

### README.todo.md - PROMOTE
- Completed tasks → README.md + mark ✅ Done
- Blocked tasks → investigate and update status
- Update priorities based on current needs

## Update Actions

### Phase 1: Analysis
- Compare current code vs documented API
- Identify implemented vs planned features
- Check for new undocumented functionality

### Phase 2: Migration
- Move completed concepts → README.md section 2
- Archive completed todos → Done section  
- Add new discovered features to README.md

### Phase 3: Refresh
- Update all timestamps to 2025-06-10
- Increment version numbers
- Refresh file locations and line numbers
- Update examples with actual working code

### Phase 4: Planning
- Add new todos based on code TODOs/FIXME
- Update priorities based on current project needs
- Identify gaps between implementation and docs

## Output Requirements
- Maintain template structure consistency
- Preserve all [[wikilinks]] and update if needed
- Keep YAML metadata current and accurate
- Ensure single source of truth (no duplication)
- Update effort estimates based on actual completion times

## Usage
```bash
# User provides current working directory  
update-docs --current-directory /path/to/module
```

## Success Criteria
- README.md reflects actual implemented API
- Concepts contain only design rationale
- TODO shows current roadmap status
- No information duplication between files
- All timestamps and versions updated