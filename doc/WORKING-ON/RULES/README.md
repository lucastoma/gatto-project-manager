# GattoNero AI Assistant - System Prompt Rules
## Organizacja Rules dla Rozwoju Projektu

> **Status:** ✅ SYSTEM PROMPT RULES  
> **Ostatnia aktualizacja:** 2024-12-19  
> **Podstawa:** Zweryfikowana dokumentacja CORE + JSX Integration

---

## 🎯 CELE SYSTEM PROMPT RULES

### Dlaczego Two-Tier Rules?
- **Building Rules:** Pełny zestaw dla kompletnego rozwoju funkcjonalności
- **Error-Solving Rules:** Lżejszy zestaw dla szybkich poprawek i bugfixów
- **Modular Approach:** Jasne rozdzielenie CORE + Algorithm modules
- **Documentation Colocation:** Dokumentacja przy kodzie (`.implementation-*`)

### Zakres Zastosowania
- ✅ **Naming Conventions** (`algorithm_XX_name` pattern)
- ✅ **Architecture Rules** (CORE + separate modules)
- ✅ **File Structure** (`app/algorithms/algorithm_XX_name/`)
- ✅ **JSX Integration** (CSV protocol, error handling)
- ✅ **Documentation Standards** (kolokacja, szablony)

---

## 📁 STRUKTURA RULES

```
RULES/
├── README.md                                # Ten plik - opis organizacji
├── system-prompt-building-rules.md         # Pełne rules dla rozwoju
├── system-prompt-error-solving-rules.md    # Lżejsze rules dla błędów
└── template-examples/                       # Szablony i przykłady
    ├── algorithm-folder-template/           # Struktura algorytmu
    ├── implementation-todo-template.md      # Szablon TODO files
    └── implementation-knowledge-template.md # Szablon KNOWLEDGE files
```

---

## 🔄 KIEDY UŻYWAĆ KTÓRYCH RULES?

### Building Rules (Pełny Development)
**Użyj gdy:**
- Tworzysz nowy algorytm od zera
- Implementujesz nową funkcjonalność
- Refaktoryzujesz architekturę
- Rozszerzasz API
- Dodajesz nowe JSX scripts

**Zawiera:**
- Kompletną architekturę projektu
- Szczegółowe naming conventions
- Pełne patterns dla wszystkich komponentów
- Documentation requirements
- Testing guidelines

### Error-Solving Rules (Quick Fixes)
**Użyj gdy:**
- Naprawiasz błędy w istniejącym kodzie
- Debugujesz problemy
- Dokonujesz małych zmian
- Poprawiasz dokumentację
- Optymalizujesz performance

**Zawiera:**
- Podstawowe architecture rules
- Error handling patterns
- Quick debugging guidelines
- Essential file structure
- Key JSX patterns

---

## 🏗️ FOUNDATION DOCUMENTATION

### Bazowa Dokumentacja (Zweryfikowana)
- **CORE:** `gatto-WORKING-01-core.md` ✅ (API, server, processing)
- **JSX Integration:** `gatto-WORKING-01-basic-photoshop-integration.md` ✅ (patterns, protocols)
- **Verification:** `VERIFICATION-SUMMARY-2024-12-19.md` ✅ (testing results)

### Algorithm Documentation
- **Toc:** `gatto-WORKING-03-algorithms-toc.md` (spis wszystkich algorytmów)
- **Individual Algorithms:** `gatto-WORKING-03-algorithms-*.md` (szczegółowe implementacje)

---

## 📋 QUICK REFERENCE

### Naming Convention
```
algorithm_XX_name/          # Folder pattern
├── .implementation-todo    # Hidden TODO file
├── .implementation-knowledge # Hidden KNOWLEDGE file
├── algorithm_main.py       # Main algorithm code
├── README.md              # Brief algorithm description
└── tests/                 # Algorithm tests
```

### JSX Protocol
- **Format:** CSV (nie JSON)
- **Success:** `success,{data}`
- **Error:** `error,{message}`
- **Files:** TIFF format, temp_jsx/ folder

### API Endpoints
- **Color Match:** `/api/colormatch` (3 methods)
- **Palette Analysis:** `/api/analyze_palette` (K-means)

---

## 🎯 NASTĘPNE KROKI

1. **Użyj Building Rules** → Implementuj nową strukturę `app/algorithms/`
2. **Migrate Existing** → Przenieś z `app/processing/` do nowej struktury
3. **Create Templates** → Stwórz `.implementation-*` templates
4. **Update Documentation** → Zaktualizuj CORE docs o nową strukturę

---

*System prompt rules zapewniają spójną organizację rozwoju GattoNero AI Assistant z zachowaniem modularności i jakości dokumentacji.*
