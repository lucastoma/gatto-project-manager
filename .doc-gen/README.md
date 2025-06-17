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

## Wsparcie

W przypadku problemów sprawdź:

1. Czy PyYAML jest zainstalowany
2. Czy plik konfiguracyjny YAML ma poprawną składnię
3. Czy ścieżki w konfiguracji istnieją
4. Czy masz uprawnienia do zapisu w katalogu docelowym