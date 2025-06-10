# WebView - Koncepcja i Architektura Techniczna

**Status:** 🚧 W ROZWOJU  
**Wersja:** 1.0  
**Data:** 19.12.2024  

## Koncepcja Ogólna

WebView to **mostek diagnostyczny** między algorytmami a integracją JSX. Głównym celem jest umożliwienie pełnego testowania logiki algorytmu w kontrolowanym środowisku webowym przed wdrożeniem do Photoshopa.

### Problem do Rozwiązania

**Obecny workflow:**
```
Algorytm → API → JSX → Photoshop
         ↑
    Trudne debugowanie
```

**Nowy workflow z WebView:**
```
Algorytm → API → WebView (testowanie)
         ↓
         API → JSX → Photoshop
              ↑
         Pewność działania
```

## Architektura Systemu

### Diagram Komponentów

```
┌─────────────────────────────────────────────────────────────┐
│                    WEBVIEW LAYER                            │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Frontend      │   Backend       │   Integration           │
│                 │                 │                         │
│ ┌─────────────┐ │ ┌─────────────┐ │ ┌─────────────────────┐ │
│ │ HTML/CSS/JS │ │ │ Flask Routes│ │ │ Existing API        │ │
│ │             │ │ │             │ │ │                     │ │
│ │ - Upload    │ │ │ - /webview  │ │ │ - /api/process      │ │
│ │ - Parameters│ │ │ - /test     │ │ │ - Algorithm Registry│ │
│ │ - Results   │ │ │ - /result   │ │ │ - Core Services     │ │
│ │ - Logging   │ │ │             │ │ │                     │ │
│ └─────────────┘ │ └─────────────┘ │ └─────────────────────┘ │
└─────────────────┴─────────────────┴─────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 EXISTING SYSTEM                             │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Algorithms    │   Core          │   API                   │
│                 │                 │                         │
│ ┌─────────────┐ │ ┌─────────────┐ │ ┌─────────────────────┐ │
│ │algorithm_01 │ │ │ Logger      │ │ │ routes.py           │ │
│ │algorithm_02 │ │ │ Profiler    │ │ │ server.py           │ │
│ │algorithm_03 │ │ │ FileHandler │ │ │                     │ │
│ │     ...     │ │ │ HealthMon   │ │ │                     │ │
│ └─────────────┘ │ └─────────────┘ │ └─────────────────────┘ │
└─────────────────┴─────────────────┴─────────────────────────┘
```

### Przepływ Danych

#### 1. Upload i Walidacja
```
User Upload → WebView Frontend → File Validation → Temp Storage
     ↓
Image Preview ← Base64 Encoding ← Image Processing ← File System
```

#### 2. Testowanie Algorytmu
```
Parameter Form → WebView Backend → API Validation → Algorithm Registry
      ↓
Algorithm Execution → Core Services → Result Generation → File System
      ↓
Result Display ← WebView Frontend ← Result Processing ← Result File
```

#### 3. Live Logging
```
Algorithm Logs → Development Logger → WebSocket/SSE → Frontend Display
```

## Wzorce Projektowe

### 1. Adapter Pattern
WebView adaptuje istniejące API do interfejsu webowego:

```python
class WebViewAdapter:
    def __init__(self, api_client):
        self.api = api_client
    
    def process_for_web(self, files, params):
        # Adaptacja parametrów webowych do API
        api_params = self._adapt_params(params)
        result = self.api.process(files, api_params)
        # Adaptacja wyniku API do formatu webowego
        return self._adapt_result(result)
```

### 2. Observer Pattern
Live logging przez obserwację logów:

```python
class LogObserver:
    def __init__(self, websocket):
        self.ws = websocket
    
    def notify(self, log_entry):
        self.ws.send(json.dumps({
            'type': 'log',
            'data': log_entry
        }))
```

### 3. Factory Pattern
Tworzenie interfejsów dla różnych algorytmów:

```python
class AlgorithmInterfaceFactory:
    @staticmethod
    def create_interface(algorithm_id):
        if algorithm_id == 'algorithm_01_palette':
            return PaletteInterface()
        elif algorithm_id == 'algorithm_02_statistical':
            return StatisticalInterface()
        # ...
```

## Integracja z Istniejącym Systemem

### Punkty Integracji

1. **Flask Server Extension**
   ```python
   # app/server.py
   from app.webview.routes import webview_bp
   app.register_blueprint(webview_bp, url_prefix='/webview')
   ```

2. **Algorithm Registry Access**
   ```python
   # app/webview/utils/algorithm_detector.py
   from app.algorithms import ALGORITHM_REGISTRY
   
   def get_available_algorithms():
       return list(ALGORITHM_REGISTRY.keys())
   ```

3. **Core Services Reuse**
   ```python
   # app/webview/utils/image_processor.py
   from app.core.development_logger import get_logger
   from app.core.performance_profiler import get_profiler
   ```

### Zasady Integracji

- **NIE modyfikuj** istniejących algorytmów
- **NIE modyfikuj** istniejącego API
- **UŻYWAJ** istniejących serwisów core
- **ROZSZERZAJ** Flask server przez blueprinty
- **TESTUJ** integrację przez istniejące testy

## Technologie i Biblioteki

### Backend
- **Flask**: Rozszerzenie istniejącego serwera
- **Werkzeug**: Upload i obsługa plików
- **Pillow**: Przetwarzanie obrazów (już używane)
- **WebSockets/SSE**: Live logging

### Frontend
- **Vanilla JavaScript**: Bez dodatkowych frameworków
- **CSS Grid/Flexbox**: Responsywny layout
- **Fetch API**: Komunikacja z backend
- **WebSocket API**: Live updates

### Uzasadnienie Wyborów

1. **Vanilla JS zamiast React/Vue**:
   - Brak dodatkowych zależności
   - Prostota implementacji
   - Szybkość ładowania
   - Łatwość debugowania

2. **Flask Blueprint zamiast osobnego serwera**:
   - Wykorzystanie istniejącej infrastruktury
   - Wspólne logi i monitoring
   - Brak konfliktów portów
   - Łatwiejsza konfiguracja

## Bezpieczeństwo

### Upload Security
```python
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def secure_upload(file):
    # Walidacja rozszerzenia
    if not allowed_file(file.filename):
        raise ValueError("Nieprawidłowy format pliku")
    
    # Walidacja rozmiaru
    if len(file.read()) > MAX_FILE_SIZE:
        raise ValueError("Plik zbyt duży")
    
    # Sanityzacja nazwy
    filename = secure_filename(file.filename)
    
    # Walidacja zawartości
    try:
        Image.open(file)
    except:
        raise ValueError("Plik nie jest prawidłowym obrazem")
```

### Parameter Sanitization
```python
def sanitize_params(params):
    sanitized = {}
    for key, value in params.items():
        # Walidacja kluczy
        if key not in ALLOWED_PARAMS:
            continue
        
        # Sanityzacja wartości
        if isinstance(value, str):
            value = html.escape(value)
        elif isinstance(value, (int, float)):
            value = max(min(value, MAX_VALUES[key]), MIN_VALUES[key])
        
        sanitized[key] = value
    
    return sanitized
```

## Wydajność

### Optymalizacje

1. **Image Compression dla Preview**:
   ```python
   def create_preview(image_path, max_size=(800, 600)):
       with Image.open(image_path) as img:
           img.thumbnail(max_size, Image.Resampling.LANCZOS)
           # Konwersja do base64 dla wyświetlenia
           return image_to_base64(img)
   ```

2. **Async Processing**:
   ```python
   @app.route('/webview/test/<algorithm_id>', methods=['POST'])
   async def test_algorithm(algorithm_id):
       task_id = str(uuid.uuid4())
       # Uruchom w tle
       executor.submit(process_algorithm, task_id, algorithm_id, params)
       return {'task_id': task_id, 'status': 'processing'}
   ```

3. **Result Caching**:
   ```python
   @lru_cache(maxsize=100)
   def get_cached_result(params_hash):
       # Cache wyników dla identycznych parametrów
       pass
   ```

## Monitoring i Debugging

### Metryki
- Czas przetwarzania algorytmów
- Liczba uploadów
- Błędy i wyjątki
- Użycie pamięci

### Logging Levels
```python
# DEBUG: Szczegółowe informacje o przepływie
logger.debug(f"Processing {algorithm_id} with params: {params}")

# INFO: Główne operacje
logger.info(f"Algorithm {algorithm_id} completed successfully")

# WARNING: Potencjalne problemy
logger.warning(f"Large file uploaded: {file_size}MB")

# ERROR: Błędy wymagające uwagi
logger.error(f"Algorithm {algorithm_id} failed", exc_info=True)
```

## Rozszerzalność

### Dodawanie Nowych Algorytmów
System automatycznie wykrywa nowe algorytmy z `ALGORITHM_REGISTRY`:

```python
def get_algorithm_interfaces():
    interfaces = {}
    for alg_id in ALGORITHM_REGISTRY.keys():
        interfaces[alg_id] = {
            'name': get_algorithm_name(alg_id),
            'params': get_algorithm_params(alg_id),
            'template': f'algorithms/{alg_id}.html'
        }
    return interfaces
```

### Dodawanie Nowych Funkcji
1. **Nowy endpoint**: Dodaj do `routes.py`
2. **Nowy template**: Stwórz w `templates/`
3. **Nowa logika JS**: Dodaj do `static/js/`
4. **Nowe style**: Dodaj do `static/css/`

## Testowanie

### Strategie Testowania

1. **Unit Tests**: Testowanie komponentów w izolacji
2. **Integration Tests**: Testowanie integracji z istniejącym API
3. **E2E Tests**: Testowanie pełnego przepływu przez Selenium
4. **Performance Tests**: Testowanie wydajności uploadów i przetwarzania

### Przykład Testu Integracji
```python
def test_algorithm_processing_integration():
    # Przygotuj dane testowe
    test_image = create_test_image()
    params = {'method': 'closest', 'preserve_luminance': True}
    
    # Wywołaj przez WebView
    response = client.post('/webview/test/algorithm_01_palette', 
                          data={'master': test_image, 'params': params})
    
    # Sprawdź wynik
    assert response.status_code == 200
    assert 'task_id' in response.json
    
    # Sprawdź czy algorytm został wywołany
    assert mock_algorithm.process.called
```

## Przyszłe Rozszerzenia

### Faza 2: Zaawansowane Funkcje
- **Batch Processing**: Testowanie wielu obrazów jednocześnie
- **Parameter Presets**: Zapisane zestawy parametrów
- **Result Comparison**: Porównywanie wyników różnych algorytmów
- **Export Results**: Eksport wyników do różnych formatów

### Faza 3: Automatyzacja
- **Automated Testing**: Automatyczne testy regresji
- **Performance Benchmarks**: Automatyczne benchmarki wydajności
- **Visual Regression Tests**: Automatyczne porównywanie wyników wizualnych
- **CI/CD Integration**: Integracja z procesami CI/CD