---
version: "1.0"
last_updated: 2025-06-10
author: lucastoma
interface_stable: true
stop_deep_scan: false
tags: 
  - api
  - module
  - interface
aliases: 
  - "[[Nazwa modułu]]"
  - "ClassName"
links:
  - "[[README.concepts]]"
  - "[[README.todo]]"
cssclasses: 
  - readme-template
---

# [[Nazwa modułu/katalogu]]

Krótki opis funkcjonalności w 1-2 zdaniach - co to robi i po co istnieje.

## 1. Overview & Quick Start

### Co to jest
Ten moduł odpowiada za [[główną funkcję]]. Jest częścią [[większego systemu]] i służy do [konkretny przypadek użycia].

### Szybki start
```bash
# Instalacja/setup (jeśli potrzebne)
pip install -r requirements.txt

# Podstawowe uruchomienie
python main.py --input data.json --output result.json

# Lub jako import
from module import ClassName
processor = ClassName()
result = processor.process(data)
```

### Struktura katalogu
```
/current_directory/
├── main.py          # Główny punkt wejścia
├── validator.py     # [[Klasa walidacji]]
├── config/         # Pliki konfiguracyjne
└── tests/          # Testy jednostkowe
```

### Wymagania
- Python 3.8+
- Biblioteki: requests, pydantic (patrz requirements.txt)
- Opcjonalnie: [[Redis]] dla cache'owania

### Najczęstsze problemy
- **Błąd importu:** Sprawdź czy wszystkie dependencje są zainstalowane
- **Timeout:** Zwiększ timeout w konfiguracji
- **Validation error:** Sprawdź format danych wejściowych

---

## 2. API Documentation

### Klasy dostępne

#### [[ClassName]]
**Przeznaczenie:** Waliduje i przetwarza dane użytkownika zgodnie z regułami biznesowymi

##### Konstruktor
```python
ClassName(config_path: str, timeout: int = 30, cache_enabled: bool = True)
```
**Parametry:**
- `config_path` (str, required): Ścieżka do pliku z regułami walidacji (.json)
- `timeout` (int, optional, default=30): Timeout dla operacji w sekundach (1-300)
- `cache_enabled` (bool, optional, default=True): Czy używać cache'a wyników

##### Główne metody

**[[process()]]**
```python
result = instance.process(data: dict, options: list = []) -> ProcessResult
```
- **Input:** `data` musi zawierać klucze: ['user_id', 'email', 'profile_data']
- **Input:** `options` lista z dozwolonych: ['strict_mode', 'auto_fix', 'generate_report']
- **Output:** `ProcessResult` obiekt z polami:
  - `.status` (str): 'success'|'error'|'warning'|'partial'
  - `.data` (dict): przetworzone dane z dodatkowymi polami
  - `.errors` (list[dict]): lista {'code': str, 'message': str, 'field': str}
  - `.warnings` (list[str]): lista ostrzeżeń
  - `.metadata` (dict): statystyki przetwarzania

**[[validate_single()]]**
```python
is_valid = instance.validate_single(item: dict, rule_set: str = 'default') -> ValidationResult
```
- **Input:** `item` dict z danymi do walidacji
- **Input:** `rule_set` nazwa zestawu reguł ('default', 'strict', 'minimal')
- **Output:** `ValidationResult` z polami:
  - `.is_valid` (bool): czy przeszło walidację
  - `.errors` (list): lista błędów walidacji
  - `.score` (float): wynik walidacji 0.0-1.0

**[[get_stats()]]**
```python
stats = instance.get_stats() -> dict
```
- **Output:** Słownik ze statystykami:
  - `processed_count` (int): liczba przetworzonych elementów
  - `success_rate` (float): procent sukcesu
  - `avg_processing_time` (float): średni czas przetwarzania w ms

### Typowe użycie

```python
# Standardowy przepływ
from validator import UserValidator

validator = UserValidator('config/rules.json', timeout=60)

# Przetworzenie pojedynczego użytkownika
user_data = {
    'user_id': 12345,
    'email': 'user@example.com', 
    'profile_data': {'age': 25, 'country': 'PL'}
}

result = validator.process(user_data, options=['strict_mode'])

if result.status == 'success':
    clean_data = result.data
    print(f"Processed user {clean_data['user_id']}")
else:
    for error in result.errors:
        print(f"Error {error['code']}: {error['message']}")

# Sprawdzenie statystyk
stats = validator.get_stats()
print(f"Success rate: {stats['success_rate']}%")
```

### Error codes
- `VAL001`: Invalid email format - email nie pasuje do regex
- `VAL002`: Missing required field - brakuje wymaganego pola
- `VAL003`: Age out of range - wiek poza zakresem 13-120
- `VAL004`: Invalid country code - nieznany kod kraju
- `SYS001`: Config file not found - nie można załadować konfiguracji
- `SYS002`: Timeout exceeded - operacja przekroczyła limit czasu

### Dependencies
**Import:**
```python
from user_validation import UserValidator
from user_validation.exceptions import ValidationError, ConfigError
```

**External dependencies:**
```txt
requests>=2.25.0
pydantic>=1.8.0  
redis>=3.5.0 (optional, for caching)
```

### File locations
- **Main class:** `./user_validation/validator.py` lines 23-156
- **Config schema:** `./config/schema.json`
- **Tests:** `./tests/test_user_validator.py`
- **Examples:** `./examples/basic_usage.py`

### Configuration
Przykład pliku konfiguracyjnego (`config/rules.json`):
```json
{
  "email_regex": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
  "age_range": {"min": 13, "max": 120},
  "required_fields": ["user_id", "email"],
  "cache_ttl": 3600
}
```

---

## template-info
Above structure and this comment in heading `template-info` is description about how to construct README.md in any/all directories

### Rozdział 1 
- klasyczne README - szybki start, overview, podstawowa orientacja

### Rozdział 2 
- szczegółowa dokumentacja API zastępująca potrzebę czytania kodu

**Details:**
- Konkretne API - dokładne sygnatury metod z typami
- Kompletne przykłady - AI widzi jak używać bez czytania kodu
- Error handling - wszystkie możliwe błędy i kody
- Dependencies - dokładnie co importować
- Lokalizacje - gdzie znajdzie kod jeśli jednak musi
- Cel: AI agent może użyć modułu bez oglądania kodu źródłowego.