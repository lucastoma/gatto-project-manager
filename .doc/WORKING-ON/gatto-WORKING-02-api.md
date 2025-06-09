# GattoNero AI Assistant - WORKING DOCUMENTATION
## Część 2: API & Photoshop Integration - Działające Interfejsy

> **Status:** ✅ DZIAŁAJĄCE API  
> **Ostatnia aktualizacja:** 2024  
> **Poprzedni:** `gatto-WORKING-01-core.md`

---

## 🌐 REST API SPECIFICATION

### Base Configuration
- **Host:** `127.0.0.1`
- **Port:** `5000`
- **Protocol:** HTTP
- **Base URL:** `http://127.0.0.1:5000`
- **Content-Type:** `multipart/form-data` (uploads), `application/json` (responses)

---

## 📡 ENDPOINTS DOCUMENTATION

### ✅ `/api/analyze_palette` (POST)

#### Opis
Analiza palety kolorów z przesłanego obrazu przy użyciu algorytmu K-means.

#### Request
```http
POST /api/analyze_palette HTTP/1.1
Host: 127.0.0.1:5000
Content-Type: multipart/form-data

--boundary
Content-Disposition: form-data; name="image"; filename="test.jpg"
Content-Type: image/jpeg

[binary image data]
--boundary--
```

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | File | ✅ | Plik obrazu (JPEG, PNG, TIFF) |
| `k` | Integer | ❌ | Liczba kolorów w palecie (default: 8) |

#### Response (Success)
```json
{
  "status": "success",
  "palette": [
    {"r": 255, "g": 128, "b": 64, "hex": "#ff8040"},
    {"r": 120, "g": 200, "b": 100, "hex": "#78c864"},
    // ... więcej kolorów
  ],
  "colors_count": 8,
  "processing_time": 0.15
}
```

#### Response (Error)
```json
{
  "status": "error",
  "message": "No image file provided",
  "error_code": "MISSING_FILE"
}
```

#### Curl Example
```bash
curl -X POST \
  http://127.0.0.1:5000/api/analyze_palette \
  -F "image=@test_image.jpg" \
  -F "k=12"
```

---

### ✅ `/api/colormatch` (POST)

#### Opis
Color matching między obrazem wzorcowym (master) a docelowym (target) przy użyciu wybranej metody.

#### Request
```http
POST /api/colormatch HTTP/1.1
Host: 127.0.0.1:5000
Content-Type: multipart/form-data

--boundary
Content-Disposition: form-data; name="master"; filename="master.tif"
Content-Type: image/tiff

[binary master image]
--boundary
Content-Disposition: form-data; name="target"; filename="target.tif"
Content-Type: image/tiff

[binary target image]
--boundary
Content-Disposition: form-data; name="method"

2
--boundary--
```

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `master` | File | ✅ | Obraz wzorcowy (źródło kolorów) |
| `target` | File | ✅ | Obraz docelowy (do przekształcenia) |
| `method` | Integer | ✅ | Metoda (1, 2, lub 3) |
| `k` | Integer | ❌ | Liczba kolorów dla metody 1 (default: 16) |

#### Dostępne Metody
| Method | Name | Description | Speed | Quality |
|--------|------|-------------|-------|----------|
| `1` | Simple Palette Mapping | K-means RGB clustering | 🟡 Medium | 🟢 Stylized |
| `2` | Basic Statistical Transfer | LAB statistics matching | 🟢 Fast | 🟢 Natural |
| `3` | Simple Histogram Matching | Luminance histogram | 🟢 Fast | 🟢 Exposure |

#### Response (Success)
```json
{
  "status": "success",
  "method": 2,
  "method_name": "Basic Statistical Transfer",
  "result_file": "test_simple_1749375027_matched.tif",
  "result_path": "app/temp_jsx/test_simple_1749375027_matched.tif",
  "processing_time": 0.01,
  "input_files": {
    "master": "master_1749375027.tif",
    "target": "target_1749375027.tif"
  }
}
```

#### Response (Error)
```json
{
  "status": "error",
  "message": "Invalid method. Use 1, 2, or 3",
  "error_code": "INVALID_METHOD",
  "available_methods": [1, 2, 3]
}
```

#### Curl Example
```bash
curl -X POST \
  http://127.0.0.1:5000/api/colormatch \
  -F "master=@master.tif" \
  -F "target=@target.tif" \
  -F "method=2"
```

---

## 🔧 ERROR HANDLING

### Standard Error Codes
| Code | Description | HTTP Status |
|------|-------------|-------------|
| `MISSING_FILE` | Brak wymaganego pliku | 400 |
| `INVALID_FORMAT` | Nieprawidłowy format obrazu | 400 |
| `INVALID_METHOD` | Nieprawidłowa metoda | 400 |
| `PROCESSING_ERROR` | Błąd podczas przetwarzania | 500 |
| `FILE_SAVE_ERROR` | Błąd zapisu wyniku | 500 |
| `INTERNAL_ERROR` | Wewnętrzny błąd serwera | 500 |

### Error Response Format
```json
{
  "status": "error",
  "message": "Human readable error message",
  "error_code": "MACHINE_READABLE_CODE",
  "details": {
    "additional": "context",
    "if": "needed"
  }
}
```

---

## 🎨 PHOTOSHOP INTEGRATION

### CEP Panel Architecture
**Lokalizacja:** `app/scripts/`

#### ✅ Główne Skrypty

##### `client.jsx` - Main CEP Panel
```javascript
// Główny interfejs użytkownika
// HTML/CSS/JavaScript + ExtendScript bridge
// Komunikacja z Python API
```

##### `color_matcher.jsx` - Color Matching Interface
```javascript
// Dedykowany interfejs dla color matching
// Wybór warstw, parametrów metody
// Preview i apply funkcjonalności
```

##### `palette_analyzer.jsx` - Palette Analysis
```javascript
// Analiza palet kolorów
// Wizualizacja wyników
// Export palet do swatches
```

##### `test_simple.jsx` - Integration Tests
```javascript
// Testy integracyjne PS ↔ Python
// Walidacja komunikacji
// Debug utilities
```

### Workflow Integration

#### 1. Export Phase (PS → Python)
```javascript
// 1. Użytkownik wybiera warstwy/obrazy w PS
var masterLayer = app.activeDocument.activeLayer;
var targetLayer = getSelectedLayer();

// 2. Export do TIFF (bezstratny)
var masterFile = exportToTIFF(masterLayer, "master_" + timestamp + ".tif");
var targetFile = exportToTIFF(targetLayer, "target_" + timestamp + ".tif");

// 3. Przygotowanie danych dla API
var formData = new FormData();
formData.append("master", masterFile);
formData.append("target", targetFile);
formData.append("method", selectedMethod);
```

#### 2. Processing Phase (Python)
```python
# 1. Odbiór plików przez Flask
master_file = request.files['master']
target_file = request.files['target']
method = int(request.form['method'])

# 2. Przetwarzanie algorytmem
result_path = process_color_matching(master_file, target_file, method)

# 3. Zwrócenie ścieżki wyniku
return jsonify({
    "status": "success",
    "result_file": result_path
})
```

#### 3. Import Phase (Python → PS)
```javascript
// 1. Odbiór odpowiedzi z API
var response = JSON.parse(httpResponse);
var resultFile = response.result_file;

// 2. Import wyniku do PS
var resultDoc = app.open(new File(resultFile));

// 3. Opcjonalne: kopiowanie do oryginalnego dokumentu
copyLayerToDocument(resultDoc, originalDoc);

// 4. Cleanup plików tymczasowych
cleanupTempFiles([masterFile, targetFile]);
```

---

## 📁 FILE MANAGEMENT

### Temporary Files Structure
```
app/temp_jsx/
├── master_1749375027.tif          # Obraz wzorcowy
├── target_1749375027.tif          # Obraz docelowy  
├── test_simple_1749375027_matched.tif # Wynik color matching
└── palette_source_1749372754913.tif   # Analiza palety
```

### Naming Convention
- **Pattern:** `{type}_{timestamp}[_{suffix}].{ext}`
- **Types:** `master`, `target`, `palette_source`
- **Suffixes:** `matched`, `analyzed`, `processed`
- **Timestamp:** Unix timestamp dla unikalności

### File Lifecycle
1. **Upload:** CEP → multipart form → Flask
2. **Processing:** Temporary storage w `app/temp_jsx/`
3. **Result:** Nowy plik z wynikiem
4. **Download:** CEP pobiera wynik
5. **Cleanup:** Automatyczne lub manualne usunięcie

---

## ⚡ PERFORMANCE METRICS

### API Response Times (Rzeczywiste)
| Endpoint | Method | Image Size | Avg Time | Status |
|----------|--------|------------|----------|--------|
| `/api/analyze_palette` | - | 1MP | 150ms | ✅ |
| `/api/colormatch` | 1 | 1MP | 190ms | ✅ |
| `/api/colormatch` | 2 | 1MP | 10ms | ✅ ⚡ |
| `/api/colormatch` | 3 | 1MP | 20ms | ✅ |

### Throughput
- **Concurrent requests:** 1 (single-threaded Flask)
- **Max file size:** 50MB (configurable)
- **Supported formats:** JPEG, PNG, TIFF
- **Memory usage:** ~2x image size

---

## 🔒 SECURITY CONSIDERATIONS

### Input Validation
```python
# File type validation
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# File size limits
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Filename sanitization
import werkzeug.utils
safe_filename = werkzeug.utils.secure_filename(filename)
```

### Network Security
- **Localhost only:** Bind do 127.0.0.1
- **No authentication:** Development mode
- **CORS:** Disabled (same-origin)
- **HTTPS:** Not implemented (localhost)

---

## 🧪 API TESTING

### Manual Testing
```bash
# Test server health
curl http://127.0.0.1:5000/api/analyze_palette

# Test palette analysis
curl -X POST \
  -F "image=@test_image.jpg" \
  http://127.0.0.1:5000/api/analyze_palette

# Test color matching
curl -X POST \
  -F "master=@master.tif" \
  -F "target=@target.tif" \
  -F "method=2" \
  http://127.0.0.1:5000/api/colormatch
```

### Automated Testing
**Plik:** `test_basic.py`
```python
# Test wszystkich metod color matching
for method in [1, 2, 3]:
    response = test_method(method)
    assert response['status'] == 'success'
    assert os.path.exists(response['result_path'])
```

### Integration Testing
**Plik:** `test_curl.py`
```python
# HTTP integration tests
# Multipart form testing
# Error handling validation
```

---

## 📊 MONITORING & DEBUGGING

### Server Logs
```
 * Serving Flask app 'app.api.routes'
 * Debug mode: off
 * Running on http://127.0.0.1:5000
127.0.0.1 - - [timestamp] "POST /api/colormatch HTTP/1.1" 200 -
```

### Request Debugging
```python
# Enable debug mode for detailed logs
app.run(debug=True)

# Custom logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks
```python
# Server status check
def check_server_health():
    try:
        response = requests.get('http://127.0.0.1:5000/api/analyze_palette')
        return response.status_code in [200, 400, 405]
    except:
        return False
```

---

## 🚀 DEPLOYMENT CONSIDERATIONS

### Development Server (Current)
```python
# Flask development server
app.run(host='127.0.0.1', port=5000, debug=False)
```

### Production Recommendations
```bash
# WSGI server (future)
gunicorn --bind 127.0.0.1:5000 app.api.routes:app

# Process management
supervisord configuration

# Reverse proxy
nginx configuration for static files
```

---

## 📝 API CHANGELOG

### v1.0 (Current)
- ✅ `/api/analyze_palette` - Palette analysis
- ✅ `/api/colormatch` - Color matching (methods 1-3)
- ✅ Multipart file uploads
- ✅ JSON responses
- ✅ Error handling

### v1.1 (Planned)
- [ ] `/api/methods` - List available methods
- [ ] `/api/status` - Server health endpoint
- [ ] Progress reporting for long operations
- [ ] Batch processing support

---

## 🔗 RELATED DOCUMENTATION

- **Core System:** `gatto-WORKING-01-core.md`
- **Server Management:** `METHODOLOGY.md`
- **Testing Guide:** `TESTING_GUIDE.md`
- **Concepts:** `color-matching-IDEAS-*.md`

---

*Ten dokument opisuje rzeczywiście działające API i integrację z Photoshopem. Wszystkie endpointy zostały przetestowane i są gotowe do użycia.*