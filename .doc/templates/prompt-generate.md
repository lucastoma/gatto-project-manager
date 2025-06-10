---
type: workflow-prompt
version: "1.0"
usage: "Initial creation of README documentation suite"
input_required: "target-directory"
---

# Generate README Documentation Suite

You are creating initial documentation for a code module/directory. Generate 3 files based on provided templates.

## Input
- `target-directory`: [USER PROVIDES] - specific directory path to document
- Do NOT scan subdirectories, focus only on current level

## Task
Create complete documentation suite:

1. **README.md** - functional interface documentation
2. **README.concepts.md** - design concepts and planning  
3. **README.todo.md** - implementation roadmap

## Templates to use:

### README.md Template
```markdown
---
version: "1.0"
last_updated: 2025-06-10
author: lucastoma
interface_stable: false
stop_deep_scan: false
tags: 
  - api
  - module
  - interface
aliases: 
  - "[[Module Name]]"
cssclasses: 
  - readme-template
---

# [[Module Name]]

Brief description of functionality - what it does and why it exists.

## 1. Overview & Quick Start

### Co to jest
This module handles [[main function]]. Part of [[larger system]] for [specific use case].

### Szybki start
```bash
# Basic usage commands
command --input data --output result
```

### Struktura katalogu
```
/target-directory/
├── main_files     # Description
└── sub_components # Description
```

### Wymagania
- Dependencies list
- System requirements

### Najczęstsze problemy
- Common issues and solutions

## 2. API Documentation

### Klasy dostępne

#### [[ClassName]]
**Przeznaczenie:** What this class does

##### Konstruktor
```language
ClassName(params) -> instance
```

##### Główne metody
**[[method_name()]]**
```language
result = instance.method(input: type) -> OutputType
```
- **Input:** exact requirements
- **Output:** exact return format

### Error codes
- `ERR001`: Description and solution

### Dependencies
Required imports and external dependencies

### File locations
- **Main files:** relative paths with line numbers
```

### README.concepts.md Template
```markdown
---
version: "1.0"
last_updated: 2025-06-10
author: lucastoma
type: concepts
implementation_status: planning
tags:
  - concepts
  - planning
aliases:
  - "[[Module - Concepts]]"
---

# Concepts - [[Module Name]]

## Problem do rozwiązania
- **Context:** Current situation
- **Pain points:** What doesn't work
- **Success criteria:** Definition of done

## Podejście koncepcyjne
### Algorithm (high-level)
```
1. Input processing
2. Core transformation  
3. Output generation
```

### Key design decisions
- **Choice rationale:** Why this approach
- **Trade-offs:** What we gain/lose

## Szkic implementacji
### Data structures
```language
InputType = {
    'field': type,
}

OutputType = {
    'result': type,
}
```

### Components to build
- [ ] `[[Component1]]` - purpose
- [ ] `[[Component2]]` - purpose

## Integration points
- **Needs:** Dependencies
- **Provides:** Interface for others

## Next steps
1. **Prototype** core component
2. **Validate** approach with sample data
3. **Implement** in priority order
```

### README.todo.md Template  
```markdown
---
version: "1.0"
last_updated: 2025-06-10
author: lucastoma
type: roadmap
priority_system: "1-3"
tags:
  - todo
  - roadmap
aliases:
  - "[[Module - TODO]]"
---

# TODO - [[Module Name]]

## Priorytet 1 (Critical) 🔴
- [ ] **[[Core implementation]]**
  - **Opis:** Implement main functionality
  - **Effort:** time estimate
  - **Dependencies:** what blocks this

## Priorytet 2 (Important) 🟡  
- [ ] **[[Feature enhancement]]**
  - **Opis:** Additional capabilities
  - **Value:** business impact

## Priorytet 3 (Nice to have) 🟢
- [ ] **[[Polish tasks]]**
  - **Opis:** Quality improvements

## Backlog 📋
- [[Idea1]] - description
- [[Idea2]] - description

## Done ✅
- [x] **[[Planning]]** (2025-06-10) - Initial documentation

## Blocked 🚫
- [ ] **[[Dependent task]]**
  - **Blocker:** External dependency
```

## Output Requirements
- Fill ALL placeholder values based on target-directory analysis
- Use actual file/class names found in directory
- Create realistic API signatures from code inspection
- Generate specific, actionable todos
- Ensure all [[wikilinks]] point to actual components

## Usage
```bash
# User provides target directory
generate-docs --target-directory /path/to/module
```