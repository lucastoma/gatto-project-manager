# WebView - Interfejs Testowania Algorytmów

**Status:** 🚧 W ROZWOJU  
**Wersja:** 1.0  
**Data:** 19.12.2024  

## Przegląd

WebView to interfejs webowy do testowania i debugowania algorytmów kolorystycznych przed integracją z Photoshop JSX. Umożliwia wizualne testowanie, porównywanie parametrów i izolację problemów w kontrolowanym środowisku.

## Szybki Start

### 1. Uruchom Serwer

```bash
# Uruchom serwer Flask (jeśli nie działa)
python server_manager_enhanced.py start

# Sprawdź status
python server_manager_enhanced.py status
```

### 2. Otwórz WebView

Przejdź do: `http://localhost:5000/webview`

### 3. Testuj Algorytm

1. Wybierz algorytm z listy
2. Wgraj obrazy (master i target)
3. Ustaw parametry
4. Kliknij "Przetestuj"
5. Porównaj wyniki

## Funkcjonalności

### ✅ Zaimplementowane
- Podstawowa struktura katalogów
- Dokumentacja rozwojowa

### 🚧 W Trakcie Implementacji
- Interfejs uploadu obrazów
- Panel parametrów
- Podgląd wyników
- Integracja z Flask server

### ❌ Planowane
- Live logging
- Porównywanie A/B
- Automatyczne testy wizualne
- Historia testów

## Struktura Plików

```
app/webview/
├── README.md                    # Ta dokumentacja
├── README-concept.md            # Architektura techniczna
├── README-todo.md               # Lista zadań
├── routes.py                    # Endpointy webowe
├── static/                      # CSS, JS, obrazy
├── templates/                   # Szablony HTML
├── utils/                       # Narzędzia pomocnicze
└── tests/                       # Testy webview
```

## API Endpoints

### GET /webview
Strona główna z listą algorytmów

### GET /webview/algorithm/{algorithm_id}
Interfejs testowania konkretnego algorytmu

### POST /webview/test/{algorithm_id}
Wysłanie żądania testowania algorytmu

### GET /webview/result/{result_id}
Pobieranie wyników testowania

## Przykłady Użycia

### Testowanie Algorithm_01_Palette

1. Przejdź do `/webview/algorithm/algorithm_01_palette`
2. Wgraj obraz master (źródłowy)
3. Wgraj obraz target (docelowy)
4. Ustaw parametry:
   - `method`: "closest" lub "average"
   - `preserve_luminance`: true/false
   - `color_count`: liczba kolorów (1-256)
5. Kliknij "Przetestuj"
6. Porównaj wynik z oryginałem

### Porównywanie Parametrów

1. Uruchom test z pierwszym zestawem parametrów
2. Zapisz wynik
3. Zmień parametry
4. Uruchom ponownie
5. Porównaj oba wyniki obok siebie

## Troubleshooting

### Problem: Strona nie ładuje się
**Rozwiązanie:**
```bash
# Sprawdź czy serwer działa
python server_manager_enhanced.py status

# Jeśli nie, uruchom ponownie
python server_manager_enhanced.py restart
```

### Problem: Upload obrazów nie działa
**Rozwiązanie:**
- Sprawdź czy obraz jest w formacie JPG/PNG
- Sprawdź czy rozmiar pliku < 10MB
- Sprawdź logi serwera: `logs/development.log`

### Problem: Algorytm zwraca błąd
**Rozwiązanie:**
1. Sprawdź logi w interfejsie webowym
2. Sprawdź logi serwera: `logs/development.log`
3. Przetestuj algorytm przez API: `/api/process`
4. Sprawdź czy parametry są poprawne

### Problem: Wyniki nie wyświetlają się
**Rozwiązanie:**
- Sprawdź czy algorytm zakończył się sukcesem
- Sprawdź czy plik wynikowy został utworzony
- Odśwież stronę (F5)

## Rozwój i Wkład

### Dodawanie Nowego Algorytmu

1. Algorytm musi być zarejestrowany w `app/algorithms/__init__.py`
2. WebView automatycznie wykryje nowy algorytm
3. Opcjonalnie: stwórz dedykowany szablon w `templates/algorithms/`

### Modyfikacja Interfejsu

1. Style CSS: `static/css/`
2. Logika JS: `static/js/`
3. Szablony HTML: `templates/`
4. Endpointy: `routes.py`

### Uruchamianie Testów

```bash
# Wszystkie testy webview
python -m pytest app/webview/tests/

# Konkretny test
python -m pytest app/webview/tests/test_webview_routes.py

# Z pokryciem kodu
python -m pytest app/webview/tests/ --cov=app.webview
```

## Bezpieczeństwo

- Wszystkie uploady są walidowane
- Pliki tymczasowe są automatycznie usuwane
- Parametry są sanityzowane przed wysłaniem
- Brak dostępu do systemu plików poza katalogiem temp

## Wydajność

- Obrazy są automatycznie kompresowane dla podglądu
- Wyniki są cache'owane
- Asynchroniczne przetwarzanie dla dużych obrazów
- Automatyczne czyszczenie starych plików

## Wsparcie

W przypadku problemów:

1. Sprawdź tę dokumentację
2. Sprawdź `README-todo.md` - może problem jest już znany
3. Sprawdź logi: `logs/development.log`
4. Sprawdź testy: czy przechodzą?

## Linki

- [Architektura Techniczna](README-concept.md)
- [Lista Zadań](README-todo.md)
- [Zasady WebView](../../.trae/rules/rules-webview.md)
- [API Documentation](../api/routes.py)