# GattoNero AI Assistant - Color Matching System

## 📋 Opis Projektu

GattoNero AI Assistant to system do dopasowywania kolorów między obrazami z planowaną integracją z Adobe Photoshop. Aktualnie zawiera działający backend Python z algorytmami dopasowywania kolorów i podstawową infrastrukturę serwera.

## ✅ Co aktualnie działa

### Backend Python
- **Serwer Flask** z API endpoints
- **3 algorytmy dopasowywania kolorów**:
  - Simple Palette Mapping
  - Basic Statistical Transfer  
  - Simple Histogram Matching
- **System zarządzania serwerem** (auto-start/stop)
- **Podstawowe testy** algorytmów
- **Obsługa plików** (upload/download obrazów)

### API Endpoints
- `/api/colormatch` - dopasowywanie kolorów między obrazami
- `/api/analyze_palette` - analiza palety kolorów obrazu
- `/health` - status serwera

## 🚀 Instalacja i Uruchomienie

### Wymagania
- Python 3.7+
- Flask, OpenCV, NumPy, scikit-learn (w requirements.txt)

### Krok 1: Instalacja zależności
```bash
pip install -r requirements.txt
```

### Krok 2: Uruchomienie serwera

**Opcja A: Automatyczne zarządzanie (zalecane)**
```bash
python server_manager.py
```
- Auto-start serwera na porcie 5000
- Sprawdzanie czy serwer już działa
- Graceful shutdown

**Opcja B: Ręczne uruchomienie**
```bash
python run_server.py
```
Serwer uruchomi się na `http://127.0.0.1:5000`

### Krok 3: Testowanie
```bash
# Test podstawowych algorytmów
python test_basic.py

# Test API przez curl
python test_curl.py
```

## 📁 Struktura Projektu

```
GattoNeroPhotoshop/
├── app/                      # Główny kod aplikacji
│   ├── api/
│   │   └── routes.py         # API endpoints (/api/colormatch, /api/analyze_palette)
│   ├── core/
│   │   └── file_handler.py   # Obsługa plików
│   ├── processing/
│   │   ├── color_matching.py # 3 algorytmy dopasowywania kolorów
│   │   └── palette_analyzer.py # Analiza palety kolorów
│   ├── scripts/              # Skrypty JSX (planowane dla Photoshop)
│   ├── server.py            # Główny serwer Flask
│   └── utils.py             # Funkcje pomocnicze
├── doc/
│   ├── IDEAS general/        # Dokumentacja koncepcyjna
│   └── WORKING-ON/          # Aktualna dokumentacja robocza
├── test_results/            # Wyniki testów
├── server_manager.py        # Zarządzanie serwerem (auto-start/stop)
├── test_basic.py           # Testy algorytmów
├── test_runner.py          # Runner testów z raportowaniem
├── test_curl.py            # Testy API
├── run_server.py           # Ręczne uruchomienie serwera
├── requirements.txt        # Zależności Python
└── README.md              # Ten plik
```

## 🛠️ API Endpoints

### `/api/colormatch` (POST)
Dopasowuje kolory między dwoma obrazami używając wybranego algorytmu.

**Parametry:**
- `source_image`: Obraz źródłowy (multipart/form-data)
- `target_image`: Obraz docelowy (multipart/form-data)  
- `method`: Algorytm (`simple_palette_mapping`, `basic_statistical_transfer`, `simple_histogram_matching`)

**Przykład odpowiedzi:**
```json
{
  "status": "success",
  "result_image": "base64_encoded_image",
  "method_used": "simple_palette_mapping",
  "processing_time": 0.45
}
```

### `/api/analyze_palette` (POST)
Analizuje obraz i zwraca dominujące kolory.

**Parametry:**
- `source_image`: Plik obrazu (multipart/form-data)
- `k`: Liczba kolorów (opcjonalny, domyślnie 8)

### `/health` (GET)
Sprawdza status serwera - zwraca `{"status": "healthy"}`.

## 🎨 Jak działają algorytmy dopasowywania kolorów

### 1. Simple Palette Mapping
- Wyodrębnia dominujące kolory z obu obrazów (K-Means)
- Mapuje każdy piksel na najbliższy kolor z palety docelowej
- Szybki, ale może dawać ostre przejścia

### 2. Basic Statistical Transfer
- Oblicza średnią i odchylenie standardowe dla każdego kanału RGB
- Normalizuje obraz źródłowy do statystyk obrazu docelowego
- Zachowuje naturalne przejścia kolorów

### 3. Simple Histogram Matching
- Dopasowuje histogram obrazu źródłowego do docelowego
- Używa funkcji transformacji dla każdego kanału koloru
- Dobry balans między jakością a szybkością

**Proces przetwarzania:**
1. Upload dwóch obrazów przez API
2. Wybór algorytmu dopasowywania
3. Przetwarzanie obrazu (OpenCV + NumPy)
4. Zwrócenie wyniku jako base64

## 🧪 Testowanie

### Test algorytmów
```bash
# Test wszystkich 3 algorytmów z przykładowymi obrazami
python test_basic.py
```
Wyniki zapisywane w `test_results/` z timestampem i metrykami wydajności.

### Test API
```bash
# Test endpoints przez curl
python test_curl.py

# Ręczny test color matching
curl -X POST -F "source_image=@obraz1.png" -F "target_image=@obraz2.png" -F "method=simple_palette_mapping" http://127.0.0.1:5000/api/colormatch
```

### Test zarządzania serwerem
```bash
# Test auto-start/stop
python test_runner.py
```

## 🐛 Rozwiązywanie problemów

**Serwer nie startuje:**
- Sprawdź zależności: `pip install -r requirements.txt`
- Sprawdź czy port 5000 nie jest zajęty
- Użyj `python server_manager.py` dla auto-diagnostyki

**Błędy algorytmów:**
- Sprawdź format obrazów (obsługiwane: PNG, JPG, TIFF)
- Upewnij się że obrazy nie są uszkodzone
- Sprawdź logi w `test_results/`

**Problemy z API:**
- Sprawdź czy serwer odpowiada: `curl http://127.0.0.1:5000/health`
- Sprawdź rozmiar plików (limit ~10MB)
- Sprawdź format multipart/form-data

## 🔮 Przyszły rozwój

### Planowane ulepszenia algorytmów
- Zaawansowane algorytmy dopasowywania (LAB color space)
- Optymalizacja wydajności (GPU acceleration)
- Adaptacyjne algorytmy (machine learning)
- Obsługa większej liczby formatów obrazów

### Integracja z Photoshop
- Skrypty JSX dla automatyzacji
- CEP Panel dla UI
- ExtendScript API integration
- Batch processing

### Dodatkowe funkcje
- Web interface dla testowania
- REST API documentation
- Docker containerization
- Performance benchmarking

## 📊 Aktualny status

**✅ Ukończone:**
- Backend Python z 3 algorytmami
- API endpoints
- System testów
- Zarządzanie serwerem

**🚧 W trakcie:**
- Dokumentacja algorytmów
- Optymalizacja wydajności

**📋 Planowane:**
- Integracja z Photoshop
- Zaawansowane algorytmy
- Web interface

---

**Wersja:** 0.5.0 (Backend MVP)  
**Data:** Styczeń 2025  
**Status:** 🚧 Backend gotowy, Photoshop w planach
