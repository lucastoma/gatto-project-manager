# Zasady Implementacji WebView (System Prompt)

**Status:** 🚧 W ROZWOJU  
**Wersja:** 1.0  
**Data:** 19.12.2024  

## Cel

Ustanowienie jednolitego standardu dla implementacji interfejsu webowego do testowania i debugowania algorytmów przed integracją z Photoshop JSX. WebView ma być narzędziem deweloperskim umożliwiającym wizualne testowanie, porównywanie parametrów i izolację problemów.

---

## 1. Filozofia WebView

WebView to **mostek między algorytmem a JSX** - pozwala na pełne przetestowanie logiki algorytmu w kontrolowanym środowisku przed integracją z Photoshopem. Główne cele:

- **Separacja problemów:** Oddzielenie błędów algorytmu od błędów JSX
- **Wizualne testowanie:** Natychmiastowy podgląd wyników z różnymi parametrami
- **Live debugging:** Śledzenie wykonania w czasie rzeczywistym
- **A/B testing:** Porównywanie różnych zestawów parametrów
- **Dokumentacja wizualna:** Automatyczne generowanie przykładów

**Kluczowe pryncypia:**

- **Rozszerzalność:** WebView rozszerza istniejący Flask server, nie zastępuje go
- **Modularność:** Każdy algorytm ma dedykowany interfejs webowy
- **Spójność:** Wszystkie interfejsy używają tego samego wzorca UI/UX
- **Automatyzacja:** Interfejs automatycznie wykrywa dostępne algorytmy i parametry

---

## 2. Struktura Katalogów WebView

```
/app/webview/
├── __init__.py                     # Inicjalizacja pakietu
├── README.md                       # Dokumentacja główna
├── README-concept.md               # Koncepcja i architektura
├── README-todo.md                  # Lista zadań do wykonania
├── routes.py                       # Endpointy webowe
├── static/                         # Zasoby statyczne
│   ├── css/
│   │   ├── main.css               # Główne style
│   │   └── algorithm-specific.css  # Style specyficzne dla algorytmów
│   ├── js/
│   │   ├── main.js                # Główna logika JS
│   │   ├── upload-handler.js      # Obsługa uploadów
│   │   ├── parameter-manager.js   # Zarządzanie parametrami
│   │   └── result-viewer.js       # Podgląd wyników
│   └── images/
│       └── placeholder.svg        # Placeholder dla obrazów
├── templates/                      # Szablony HTML
│   ├── base.html                  # Szablon bazowy
│   ├── index.html                 # Strona główna
│   ├── algorithm-test.html        # Interfejs testowania algorytmu
│   └── components/
│       ├── upload-form.html       # Komponent uploadu
│       ├── parameter-panel.html   # Panel parametrów
│       └── result-display.html    # Wyświetlanie wyników
├── utils/
│   ├── __init__.py
│   ├── image_processor.py         # Przetwarzanie obrazów dla webview
│   ├── parameter_validator.py     # Walidacja parametrów
│   └── result_formatter.py       # Formatowanie wyników
└── tests/
    ├── __init__.py
    ├── test_webview_routes.py      # Testy endpointów
    ├── test_image_processor.py     # Testy przetwarzania obrazów
    └── test_parameter_validator.py # Testy walidacji
```

---

## 3. Dokumentacja Katalogów

### README.md (Główny)
**Przeznaczenie:** Dokumentacja dla deweloperów używających webview  
**Zawartość:**
- Instrukcje uruchomienia
- Przegląd funkcjonalności
- Przykłady użycia
- Troubleshooting
- API reference

### README-concept.md
**Przeznaczenie:** Architektura i koncepcja techniczna  
**Zawartość:**
- Diagramy architektury
- Przepływ danych
- Integracja z istniejącym systemem
- Wzorce projektowe
- Decyzje techniczne i uzasadnienia

### README-todo.md
**Przeznaczenie:** Zarządzanie rozwojem i priorytetami  
**Zawartość:**
- Lista funkcjonalności do implementacji
- Priorytety (High/Medium/Low)
- Status implementacji (✅/🚧/❌)
- Zależności między zadaniami
- Timeline i milestones

---

## 4. Workflow Implementacji WebView

### Krok 1: Przygotuj Środowisko

Upewnij się, że serwer Flask działa:

```bash
python server_manager_enhanced.py start
python server_manager_enhanced.py status
```

### Krok 2: Stwórz Strukturę WebView

W folderze `app/` stwórz katalog `webview/` z pełną strukturą zgodnie z powyższym schematem.

### Krok 3: Zaimplementuj Dokumentację

Wypełnij wszystkie trzy pliki README zgodnie z ich przeznaczeniem:

1. **README.md** - Instrukcje dla użytkowników
2. **README-concept.md** - Architektura techniczna
3. **README-todo.md** - Plan rozwoju

### Krok 4: Zintegruj z Flask Server

W `app/server.py` dodaj routing do webview:

```python
from app.webview.routes import webview_bp
app.register_blueprint(webview_bp, url_prefix='/webview')
```

### Krok 5: Implementuj Interfejs Algorytmu

Dla każdego algorytmu stwórz dedykowany interfejs testowy z:
- Formularzem uploadu obrazów
- Panelem parametrów
- Podglądem wyników
- Logami w czasie rzeczywistym

### Krok 6: Testuj i Dokumentuj

Uruchom testy webview i zaktualizuj dokumentację:

```bash
python -m pytest app/webview/tests/
```

---

## 5. Złote Zasady WebView

- **ROZSZERZAJ, NIE ZASTĘPUJ:** WebView rozszerza istniejący Flask server, nie tworzy nowego
- **JEDEN INTERFEJS NA ALGORYTM:** Każdy algorytm ma dedykowany interfejs testowy
- **DOKUMENTUJ WIZUALNIE:** Każdy interfejs automatycznie generuje przykłady użycia
- **TESTUJ PRZED JSX:** Zawsze przetestuj algorytm w webview przed integracją z Photoshopem
- **LOGUJ WSZYSTKO:** WebView musi pokazywać logi w czasie rzeczywistym
- **ZACHOWAJ SPÓJNOŚĆ:** Wszystkie interfejsy używają tego samego wzorca UI/UX
- **WALIDUJ PARAMETRY:** Każdy parametr musi być walidowany przed wysłaniem do algorytmu
- **OBSŁUGUJ BŁĘDY:** Interfejs musi elegancko obsługiwać wszystkie błędy algorytmu

---

## 6. Integracja z Istniejącym Systemem

WebView integruje się z:

- **app/algorithms/**: Automatyczne wykrywanie dostępnych algorytmów
- **app/api/routes.py**: Wykorzystanie istniejących endpointów API
- **app/core/**: Użycie loggerów i profilerów
- **tests/**: Rozszerzenie istniejącej struktury testów

WebView **NIE** modyfikuje:
- Logiki algorytmów
- Struktury API
- Systemu testów algorytmów
- Integracji z JSX

---

## 7. Priorytety Implementacji

### Faza 1: Podstawy (High Priority)
- [ ] Struktura katalogów
- [ ] Dokumentacja (3 pliki README)
- [ ] Podstawowy interfejs dla algorithm_01_palette
- [ ] Upload i podgląd obrazów
- [ ] Integracja z Flask server

### Faza 2: Funkcjonalność (Medium Priority)
- [ ] Panel parametrów z walidacją
- [ ] Live logging
- [ ] Porównywanie wyników A/B
- [ ] Interfejsy dla pozostałych algorytmów

### Faza 3: Zaawansowane (Low Priority)
- [ ] Automatyczne testy wizualne
- [ ] Eksport wyników
- [ ] Historia testów
- [ ] Metryki wydajności