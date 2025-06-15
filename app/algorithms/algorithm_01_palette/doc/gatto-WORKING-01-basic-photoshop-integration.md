# GattoNero AI Assistant - Basic Photoshop Integration
## Podstawowa Integracja JSX dla 3 Algorytmów Color Matching

> **Status:** ✅ BASIC JSX INTEGRATION  
> **Ostatnia aktualizacja:** 2024-12-19  
> **Podstawa:** Przetestowane skrypty `palette_analyzer.jsx`, `color_matcher.jsx`, `test_simple.jsx`

---

## 🎯 FILOZOFIA BASIC INTEGRATION

### Dlaczego BASIC?
- **Prostota:** Minimum kodu, maksimum funkcjonalności
- **Skuteczność:** Przetestowane rozwiązania, sprawdzone protokoły
- **CSV over JSON:** Prostszy parsing, mniej błędów
- **Jeden plik = jedna funkcja:** Modularność i łatwość debugowania

### Zakres Funkcjonalny
- ✅ **3 Algorytmy Color Matching** (Palette, Statistical, Histogram)
- ✅ **Analiza Palety Kolorów** (K-means clustering)
- ✅ **File Management** (TIFF export/import)
- ✅ **Error Handling** (Robust error reporting)

---

## 📁 STRUKTURA SKRYPTÓW JSX

### Verified Scripts
```
app/scripts/
├── palette_analyzer.jsx    # ✅ Analiza palety kolorów (CSV protocol)
├── color_matcher.jsx       # ✅ Color matching 3 metod (CSV protocol)  
└── test_simple.jsx         # ✅ Basic connectivity test
```

### Usunięte/Niepoprawne
- ❌ `client.jsx` - USUNIĘTY (niepoprawny protokół JSON)

---

## 🔄 PROTOKÓŁ WYMIANY DANYCH

### Format CSV (Ustalony Standard)
**Dlaczego CSV?**
- Prostszy parsing niż JSON
- Mniej podatny na błędy składni
- Szybszy transfer danych
- Łatwiejszy debugging

### API Response Formats

#### `/api/analyze_palette` Response:
```csv
success,{count},{r,g,b,r,g,b,...}
```
**Przykład:**
```csv
success,3,255,128,64,100,200,50,75,175,225
```

#### `/api/colormatch` Response:
```csv
success,method{X},{filename}
```
**Przykład:**
```csv
success,method1,test_simple_1749392883_matched.tif
```

#### Error Response (obie metody):
```csv
error,{error_message}
```

---

## 🎨 PATTERN: Color Matching (color_matcher.jsx)

### Główny Workflow
```jsx
1. Configuration Dialog → wybór master/target docs + metoda
2. Export Documents → TIFF files w temp_jsx/
3. HTTP Request → curl POST multipart/form-data
4. Parse CSV Response → success,method{X},{filename}
5. Import Result → otwórz wynikowy plik w PS
6. Cleanup → usuń pliki tymczasowe
```

### Kluczowe Funkcje

#### showConfigurationDialog()
```jsx
// Centralne okno wyboru:
// - Master document (dropdown)
// - Target document (dropdown)  
// - Method (1: Palette, 2: Statistical, 3: Histogram)
// - K colors parameter (dla metody 1)
```

#### parseColorMatchResponse()
```jsx
// CSV Parser:
// Input:  "success,method1,result_file.tif"
// Output: { status: "success", method: "method1", filename: "result_file.tif" }
```

#### executeCurl()
```jsx
// HTTP Request:
// Windows: cmd batch file + stdout capture
// macOS: AppleScript shell command
// Parametry: master_image, target_image, method, k
```

---

## 🎨 PATTERN: Palette Analysis (palette_analyzer.jsx)

### Główny Workflow
```jsx
1. Active Layer Selection → bieżąca warstwa
2. K Colors Input → prompt użytkownika (1-50)
3. Export Layer → TIFF file w temp_jsx/
4. HTTP Request → curl POST multipart/form-data
5. Parse CSV Response → success,{count},{r,g,b,...}
6. Create Color Swatches → nowa paleta w PS
7. Cleanup → usuń pliki tymczasowe
```

### Kluczowe Funkcje

#### parseSimpleResponse()
```jsx
// CSV Parser dla palety:
// Input:  "success,3,255,128,64,100,200,50,75,175,225"
// Output: [[255,128,64], [100,200,50], [75,175,225]]
```

#### saveLayerToPNG()
```jsx
// Export pojedynczej warstwy:
// - Ukryj wszystkie inne warstwy
// - Zapisz jako TIFF
// - Przywróć widoczność warstw
```

#### createColorSwatches()
```jsx
// Wizualizacja palety:
// - Nowy dokument 400x100px
// - Prostokąty kolorów
// - Nazwa z wartościami RGB
```

---

## 🛠️ ZASADY KONSTRUKCJI JSX

### 1. Error Handling Pattern
```jsx
try {
    // Main workflow
    var result = processImage();
    alert("SUCCESS: " + result);
} catch (e) {
    alert("ERROR: " + e.message);
} finally {
    // Cleanup files
    cleanupFile(tempFile);
}
```

### 2. File Management Pattern
```jsx
// Temporary files w temp_jsx/
var tempFolder = new Folder(projectRoot + "/temp_jsx");
if (!tempFolder.exists) tempFolder.create();

// Timestamp naming
var fileName = prefix + "_" + Date.now() + ".tif";

// Cleanup after use
function cleanupFile(file) {
    if (file && file.exists) {
        try { file.remove(); } catch (e) { /* ignore */ }
    }
}
```

### 3. Document Export Pattern
```jsx
// TIFF Save Options (standard)
var tiffOptions = new TiffSaveOptions();
tiffOptions.imageCompression = TIFFEncoding.NONE; // Bezstratnie
tiffOptions.layers = false; // Spłaszczony obraz

doc.saveAs(filePath, tiffOptions, true, Extension.LOWERCASE);
```

### 4. HTTP Request Pattern (Windows)
```jsx
// curl command przez CMD batch file
var cmdFile = new File(tempFolder + "/command.cmd");
var stdoutFile = new File(tempFolder + "/output.txt");

cmdFile.open("w");
cmdFile.writeln("@echo off");
cmdFile.writeln(curlCommand);
cmdFile.close();

app.system('cmd /c ""' + cmdFile.fsName + '" > "' + stdoutFile.fsName + '""');

// Wait for response with timeout
var maxWaitTime = 15000; // 15 sekund
// ... polling logic ...
```

---

## 📊 PARAMETRY I KONFIGURACJA

### Server Configuration
```jsx
var SERVER_URL = "http://127.0.0.1:5000/api/colormatch"; // lub analyze_palette
```

### Method Parameters
- **Method 1 (Palette):** `k` colors (4-32, default: 8)
- **Method 2 (Statistical):** brak dodatkowych parametrów
- **Method 3 (Histogram):** brak dodatkowych parametrów

### File Paths
```jsx
var projectRoot = new File($.fileName).parent.parent; // GattoNeroPhotoshop/
var tempFolder = projectRoot + "/temp_jsx/";          // temp files
var resultsFolder = projectRoot + "/results/";        // wyniki
```

---

## ⚡ OPTYMALIZACJE I BEST PRACTICES

### Performance
- **TIFF Format:** Bezstratny, szybki zapis/odczyt
- **Single Layer Export:** Tylko aktywna warstwa (palette_analyzer)
- **Timeout Handling:** 15s limit dla HTTP requests
- **Immediate Cleanup:** Usuwanie plików tymczasowych

### User Experience
- **Configuration Dialog:** Wszystkie parametry w jednym oknie
- **Progress Feedback:** Alert messages o postępie
- **Error Messages:** Szczegółowe informacje o błędach
- **File Validation:** Sprawdzanie istnienia plików

### Security
- **Path Validation:** Kontrola ścieżek plików
- **Input Sanitization:** Walidacja parametrów użytkownika
- **File Cleanup:** Automatyczne usuwanie temp files
- **Error Isolation:** Try-catch dla każdej operacji

---

## 🧪 TESTING WORKFLOW

### test_simple.jsx
```jsx
// Basic connectivity test:
// 1. Alert message
// 2. File write test (desktop log)
// 3. Exception handling verification
```

### Verification Steps
1. **JSX Engine:** `test_simple.jsx` - podstawowy test działania
2. **HTTP Connection:** `palette_analyzer.jsx` - test API komunikacji  
3. **Full Workflow:** `color_matcher.jsx` - test kompletnego procesu

---

## 🎯 ROZWÓJ I ROZSZERZENIA

### Priorytet 1: Stabilność
- [ ] Batch processing (multiple files)
- [ ] Progress bars dla długich operacji
- [ ] Configuration persistence (user preferences)
- [ ] Advanced error recovery

### Priorytet 2: UI/UX
- [ ] Drag & drop file support
- [ ] Preview thumbnails w dialog
- [ ] Real-time parameter preview
- [ ] Keyboard shortcuts

### Priorytet 3: Integration
- [ ] Photoshop Actions integration
- [ ] Bridge integration
- [ ] Preset management system
- [ ] Automated workflows

---

## 📝 TEMPLATE JSX SCRIPT

### Minimal Working Example
```jsx
#target photoshop

var SERVER_URL = "http://127.0.0.1:5000/api/endpoint";

function main() {
    try {
        // 1. Validate input
        if (app.documents.length === 0) {
            throw new Error("Open a document first");
        }
        
        // 2. Setup paths
        var projectRoot = new File($.fileName).parent.parent;
        var tempFolder = new Folder(projectRoot + "/temp_jsx");
        if (!tempFolder.exists) tempFolder.create();
        
        // 3. Export file
        var tempFile = exportDocument(app.activeDocument, tempFolder);
        
        // 4. HTTP request
        var response = executeCurl(tempFile);
        
        // 5. Parse response
        var result = parseCSVResponse(response);
        
        // 6. Process result
        processResult(result);
        
        alert("SUCCESS!");
        
    } catch (e) {
        alert("ERROR: " + e.message);
    } finally {
        cleanupFile(tempFile);
    }
}

main();
```

---

*Ten dokument opisuje podstawową integrację JSX dla systemu GattoNero AI Assistant, opartą na przetestowanych skryptach i ustalonych protokołach komunikacji.*
