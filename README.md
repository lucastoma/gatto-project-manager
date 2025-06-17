# Skrypty Agregacji Plików - Dokumentacja

## Przegląd

Katalog `.doc-gen` zawiera skrypty do automatycznej agregacji plików projektowych w grupy tematyczne. Skrypty generują zbiorczy plik Markdown z zawartością wszystkich plików podzielonych na logiczne grupy.

## Pliki w katalogu 

### Skrypty

- **`.comb-scripts-v2.py`** - Wersja 4.0 (stara, uniwersalna)
- **`.comb-scripts-v3.py`** - Wersja 5.0 (nowa, z grupami i YAML)

### Konfiguracja

- **`.comb-scripts-config01.yaml`** - Konfiguracja grup plików dla v3

### Dokumentacja

- **`README.md`** - Ten plik

## Jak używać nowej wersji (v3)

### 1. Wymagania

```bash
pip install PyYAML
```

### 2. Uruchomienie

```bash
# Z katalogu głównego projektu
python .doc-gen\.comb-scripts-v3.py
```

### 3. Wynik

Skrypt utworzy plik `.comb-scripts.md` w katalogu głównym projektu (workspace root).

## Konfiguracja grup (YAML)

Plik `.comb-scripts-config01.yaml` definiuje grupy plików:

```yaml
groups:
  - name: "Nazwa Grupy"
    description: "Opis grupy"
    patterns:
      - "*.py"      # wzorce plików
      - "*.jsx"
    paths:
      - "app/scripts"  # ścieżki do przeszukania
      - "all"          # specjalna wartość = cały workspace
    recursive: true     # czy szukać w podkatalogach
```

### Dostępne grupy (domyślnie)

1. **Dokumentacja Algorytmów** - pliki `*.md` z katalogów dokumentacji algorytmów
2. **Kod Python** - wszystkie pliki `*.py` w całym workspace
3. **Skrypty JSX** - pliki `*.jsx` ze skryptów Photoshop
4. **Konfiguracja i Dokumentacja** - pliki konfiguracyjne z katalogu głównego

## Kluczowe różnice v3 vs v2

### Workspace Root

- **v2**: Używa katalogu, w którym jest uruchamiany skrypt
- **v3**: Automatycznie ustawia workspace root na katalog wyżej niż lokalizacja skryptu

### Organizacja plików

- **v2**: Wszystkie pliki w jednej liście
- **v3**: Pliki podzielone na grupy tematyczne

### Konfiguracja

- **v2**: Konfiguracja w kodzie Python
- **v3**: Konfiguracja w zewnętrznym pliku YAML

### Struktura wyjścia

- **v2**: Prosta lista plików + zawartość
- **v3**: Spis grup + zawartość podzielona na grupy

## Przykład użycia

```bash
# Przejdź do katalogu projektu
cd d:\Unity\Projects\GattoNeroPhotoshop

# Uruchom nowy skrypt
python .doc-gen\.comb-scripts-v3.py

# Sprawdź wynik
type .comb-scripts.md
```

## Dostosowywanie

### Dodanie nowej grupy

Edytuj plik `.comb-scripts-config01.yaml`:

```yaml
groups:
  # ... istniejące grupy ...
  - name: "Moja Nowa Grupa"
    description: "Opis mojej grupy"
    patterns:
      - "*.txt"
      - "*.log"
    paths:
      - "logs"
      - "temp"
    recursive: true
```

### Zmiana nazwy pliku wyjściowego

W pliku YAML:

```yaml
output_file: ".moj-plik-wyjsciowy.md"
```

### Wykluczenie plików

Skrypt automatycznie respektuje reguły z `.gitignore`.

## Rozwiązywanie problemów

### Błąd: "No module named 'yaml'"

```bash
pip install PyYAML
```

### Błąd: "Nie znaleziono pliku konfiguracyjnego"

Upewnij się, że plik `.comb-scripts-config01.yaml` istnieje w katalogu `.doc-gen`.

### Puste grupy

Sprawdź czy ścieżki w konfiguracji YAML są poprawne względem workspace root.

## Migracja z v2 do v3

1. Zainstaluj PyYAML: `pip install PyYAML`
2. Dostosuj konfigurację YAML do swoich potrzeb
3. Uruchom v3: `python .doc-gen\.comb-scripts-v3.py`
4. Porównaj wyniki z v2
5. Po weryfikacji możesz usunąć v2

## Plan Rozwoju - Django GUI

### Faza 1: Podstawy Django GUI ⏳

**Cel**: Stworzenie podstawowego interfejsu webowego z funkcjonalnością quick option

**Kroki implementacji**:
1. ✅ Konfiguracja Django projektu (zakończone)
2. ⏳ Stworzenie modeli Django dla konfiguracji
3. ⏳ Implementacja widoków dla quick option
4. ⏳ Stworzenie formularzy do edycji list glob
5. ⏳ Podstawowy interfejs użytkownika

**Czas realizacji**: 2-3 dni

### Faza 2: Format .gatto-Q ⏳

**Cel**: Implementacja lokalnych plików konfiguracyjnych .gatto-Q

**Kroki implementacji**:
1. ⏳ Definicja struktury pliku .gatto-Q (JSON/YAML)
2. ⏳ Funkcje odczytu/zapisu plików .gatto-Q
3. ⏳ Integracja z istniejącymi skryptami
4. ⏳ Automatyczne wykrywanie plików .gatto-Q
5. ⏳ Migracja ustawień z obecnych formatów

**Czas realizacji**: 1-2 dni

### Faza 3: Uniwersalny Panel Glob ⏳

**Cel**: Stworzenie zaawansowanego komponentu do zarządzania wzorcami plików

**Kroki implementacji**:
1. ⏳ Projekt interfejsu panelu glob
2. ⏳ Implementacja logiki dodawania/usuwania wzorców
3. ⏳ Podgląd na żywo dopasowanych plików
4. ⏳ Walidacja wzorców glob
5. ⏳ Integracja z formularzami konfiguracji

**Czas realizacji**: 2-3 dni

### Faza 4: Zaawansowane Funkcje ⏳

**Cel**: Dodanie funkcji kopiowania do schowka i drag & drop

**Kroki implementacji**:
1. ⏳ Implementacja kopiowania do schowka (JavaScript)
2. ⏳ Funkcjonalność drag & drop dla plików wyjściowych
3. ⏳ Optymalizacja interfejsu użytkownika
4. ⏳ Testy funkcjonalności
5. ⏳ Dokumentacja użytkownika

**Czas realizacji**: 1-2 dni

### Aktualny Stan Projektu

**✅ Zakończone**:
- Podstawowa struktura Django
- Skrypty backend (quick_context_collector.py, advanced_context_collector.py)
- System konfiguracji YAML
- Testy funkcjonalności

**⏳ W trakcie**:
- Planowanie architektury Django GUI

**📋 Do zrobienia**:
- Wszystkie fazy wymienione powyżej

### Architektura Docelowa

```
gatto-project-manager/
├── apps/collector/          # Django app
│   ├── models.py           # Modele dla konfiguracji
│   ├── views.py            # Widoki GUI
│   ├── forms.py            # Formularze Django
│   └── templates/          # Szablony HTML
├── static/                 # CSS, JS, obrazy
├── templates/              # Globalne szablony
└── .gatto-Q               # Lokalne pliki konfiguracyjne
```

### Następne Kroki

1. **Rozpocznij Fazę 1**: Stworzenie podstawowych modeli Django
2. **Testuj iteracyjnie**: Każda faza powinna być testowana przed przejściem do następnej
3. **Zachowaj kompatybilność**: Istniejące skrypty powinny działać równolegle z GUI

## Wsparcie

W przypadku problemów sprawdź:

1. Czy PyYAML jest zainstalowany
2. Czy plik konfiguracyjny YAML ma poprawną składnię
3. Czy ścieżki w konfiguracji istnieją
4. Czy masz uprawnienia do zapisu w katalogu docelowym

### Wsparcie dla Django GUI

Dla problemów z Django GUI:

1. Sprawdź czy Django jest zainstalowany: `python manage.py --version`
2. Uruchom serwer deweloperski: `python manage.py runserver`
3. Sprawdź logi Django w konsoli
4. Upewnij się, że wszystkie zależności z requirements.txt są zainstalowane