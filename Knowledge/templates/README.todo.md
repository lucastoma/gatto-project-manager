# README.todo.md : example and explanation

## EXAMPLE

```markdown
---
version: "1.0"
last_updated: 2025-06-13
author: lucastoma
type: roadmap
priority_system: "1-5"
auto_update: true
tags:
  - todo
  - roadmap
  - planning
  - tasks
aliases:
  - "[[Nazwa modułu - TODO]]"
  - "todo"
  - "roadmap"
links:
  - "[[README]]"
  - "[[README.concepts]]"
cssclasses:
  - todo-template
---

<!-- TODO tasks start here -->

# Documentation Workflow Concept Plan : (Notes + TODO + Current Goal) : [[Nazwa modułu]]

## Notes (example)

- The workflow is meant to help both human and AI agents manage code context and documentation efficiently.
- Retain conventional README.md naming for compatibility; highlight entry-point for agents with a banner if needed.
- Synchronization: Concepts → TODO (actionable, cross-linked) → README (final, user-facing); prune outdated info from previous files.
- Automate synchronization and last_updated metadata where possible.
- Gather feedback post-implementation and iterate on workflow.
- Process spec for workflow synchronization created in docs/PROCESS.md
- README.md template updated with entry-point banner and last_updated date.

## Task List (example)

- [x] Review current templates for README, README.concepts, and README.todo
- [x] Document and formalize the synchronization workflow (concept-to-todo-to-readme) as a short process spec in the repo
- [x] Add agent/human entry-point banner to README.md
- [ ] Develop conceptual documentation of automation plan for doc synchronization and last_updated metadata (not implementation)
- [ ] Implement pre-commit hook or CI lint for header duplication and content rules (conceptual, not implementation) [link-to-proper-knowledge-use-README.concept.md-as-source](#md-link)
- [ ] Define and document minimal required section set for each file [link-to-proper-knowledge-use-README.concept.md-as-source](#md-link)
- [ ] Ensure documentation supports both human and AI agent usability (no need to link knowledge to any file, only if needed)
- [ ] Gather team/agent feedback, refine process and automation [link-to-proper-knowledge-use-README.concept.md-as-source](#md-link)

## Current Goal (example)

Document conceptual automation and feedback/iteration process
```

## DESCRIPTION OF FIELDS

### YAML header (minimal)

```yaml
version: "1.0"
type: roadmap
priority_system: "1-5"
auto_update: true
```

### Documentation Workflow Concept Plan

#### Notes (example)

<AI-Agent-task>write down explanation</AI-Agent-task>

#### Task List

<AI-Agent-task>write down explanation</AI-Agent-task>

#### Current Goal

<AI-Agent-task>write down explanation</AI-Agent-task>

## Future Development

### Backlog 📋

#### Pomysły do przemyślenia

- [[Batch processing]] - przetwarzanie grupowe
- [[Caching layer]] - warstwa cache'owania wyników
- [[Metrics collection]] - zbieranie metryk użycia

#### Zgłoszone bugi 🐛

- [ ] **[[Memory leak bug]]** - tracked in [[issue #123]]
- [ ] **[[Performance degradation]]** with files >50MB
- [ ] **[[Thread safety]]** issues in multi-threaded environment

## Breaking Changes Planned ⚠️

### v2.0 (planowane: Q4 2025)

- Constructor will require `config` parameter (currently optional)
- `[[process()]]` will return different error format
- Removal of deprecated `[[validate_old()]]` method

---

## Metadata

**Last review:** 2025-06-10 by [[lucastoma]]
**Next review:** 2025-06-17
**Related projects:** [[ProjectA]], [[ProjectB]]
