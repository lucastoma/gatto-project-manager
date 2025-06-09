# GattoNero AI Assistant - WORKING DOCUMENTATION
## Część 4: Photoshop Integration - Chapter 1: Overview & Architecture

> **Status:** ✅ DZIAŁAJĄCA INTEGRACJA  
> **Ostatnia aktualizacja:** 2024  
> **Spis treści:** `gatto-WORKING-04-photoshop-toc.md`

---

## 🎨 PHOTOSHOP INTEGRATION OVERVIEW

### System Architecture
```
GattoNero AI Assistant - Photoshop Integration
┌─────────────────────────────────────────────────────────────┐
│                    Adobe Photoshop CC 2019+                │
├─────────────────┬─────────────────┬─────────────────────────┤
│   CEP Panel     │  ExtendScript   │    Photoshop Core      │
│  (HTML/CSS/JS)  │     (.jsx)      │   (Layer Operations)   │
├─────────────────┼─────────────────┼─────────────────────────┤
│ • User Interface│ • Layer Ops     │ • Document Management  │
│ • Form Controls │ • File Export   │ • Layer Management     │
│ • Status Display│ • File Import   │ • Color Operations     │
│ • Progress UI   │ • API Calls     │ • File I/O             │
└─────────────────┴─────────────────┴─────────────────────────┘
           │                 │                    │
           ▼                 ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Communication Layer                     │
├─────────────────┬─────────────────┬─────────────────────────┤
│   CSInterface   │  System Calls   │    File Operations     │
│    (CEP API)    │     (curl)      │     (TIFF Export)      │
└─────────────────┴─────────────────┴─────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 Python API Server                          │
├─────────────────┬─────────────────┬─────────────────────────┤
│ Color Matching  │ Palette Analysis│    File Processing     │
│   Algorithms    │   Algorithms    │     & Management       │
└─────────────────┴─────────────────┴─────────────────────────┘
```

### Communication Flow
```
User Interaction Flow:
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│   User   │───▶│CEP Panel │───▶│ExtendScrpt│───▶│ API Call │
│  Action  │    │(UI Event)│    │(Function)│    │  (curl)  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                      │
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────▼────┐
│ Result   │◀───│Layer     │◀───│File      │◀───│ Python   │
│ Display  │    │Import    │    │Export    │    │API Server│
└──────────┘    └──────────┘    └──────────┘    └──────────┘

Detailed Process:
1. User selects layers in CEP Panel
2. CEP Panel validates selection
3. CEP Panel calls ExtendScript function via CSInterface
4. ExtendScript exports selected layers to TIFF files
5. ExtendScript builds curl command with file paths
6. ExtendScript executes curl to send files to Python API
7. Python API processes color matching
8. Python API returns result file path
9. ExtendScript imports result file as new layer
10. CEP Panel updates UI with success status
```

---

## 📁 COMPLETE FILE STRUCTURE

### Project Root Structure
```
d:\Unity\Projects\GattoNeroPhotoshop\
├── app/                               # Python API Server
│   ├── server.py                      # Main API server
│   ├── processing.py                  # Core processing logic
│   └── scripts/                       # CEP Extension Files
│       ├── CSXS/                      # ✅ CEP Configuration
│       │   └── manifest.xml           # Extension manifest
│       ├── index.html                 # ✅ CORRECTED: Main CEP Panel
│       ├── host/                      # ✅ CORRECTED: ExtendScript files
│       │   ├── main.jsx               # Main ExtendScript logic
│       │   ├── layer-operations.jsx   # Layer management
│       │   ├── file-operations.jsx    # File export/import
│       │   └── api-communication.jsx  # API calls via curl
│       ├── css/                       # ✅ Panel Styling
│       │   ├── main.css               # Main styles
│       │   └── components.css         # Component styles
│       ├── js/                        # ✅ CEP Panel Logic
│       │   ├── main.js                # Main application logic
│       │   ├── ui.js                  # UI management
│       │   └── utils.js               # Utility functions
│       └── assets/                    # ✅ Resources
│           ├── icons/                 # UI icons
│           └── images/                # Interface graphics
├── doc/                               # Documentation
├── source/                            # Test images
├── results/                           # Processing results
└── uploads/                           # API upload cache
```

### ⚠️ **KRYTYCZNE KOREKTY** (na podstawie code review):

#### 1. Manifest.xml - MainPath/ScriptPath
```xml
<!-- ❌ BŁĘDNA KONFIGURACJA -->
<Resources>
    <MainPath>./client.jsx</MainPath>        <!-- NIE! -->
    <ScriptPath>./js/main.js</ScriptPath>    <!-- NIE! -->
</Resources>

<!-- ✅ POPRAWNA KONFIGURACJA -->
<Resources>
    <MainPath>./index.html</MainPath>        <!-- Panel UI -->
    <ScriptPath>./host/main.jsx</ScriptPath> <!-- ExtendScript -->
</Resources>
```

#### 2. File Organization
```
✅ NOWA STRUKTURA (poprawiona):
app/scripts/
├── index.html              # Panel interface (MainPath)
├── host/                   # ExtendScript files
│   ├── main.jsx           # Main ExtendScript (ScriptPath)
│   ├── layer-ops.jsx      # Layer operations
│   ├── file-ops.jsx       # File operations
│   └── api-comm.jsx       # API communication (curl)
├── js/                    # CEP Panel JavaScript
│   ├── main.js            # CEP panel logic
│   ├── ui.js              # UI management
│   └── utils.js           # Utilities
└── css/                   # Styling
    ├── main.css           # Main styles
    └── components.css     # Components
```

---

## 🔄 COMMUNICATION PROTOCOLS

### 1. CEP Panel ↔ ExtendScript Communication
```javascript
// CEP Panel (main.js) - Wywołanie funkcji ExtendScript
function performColorMatching() {
    var params = {
        masterId: parseInt(elements.masterLayerSelect.value),
        targetId: parseInt(elements.targetLayerSelect.value),
        method: app.currentMethod,
        k: getMethodParams()
    };
    
    // ✅ POPRAWNA komunikacja CEP → ExtendScript
    csInterface.evalScript(
        'runColorMatching(' + JSON.stringify(params) + ')', 
        function(result) {
            handleColorMatchingResult(result);
        }
    );
}

// ExtendScript (main.jsx) - Funkcja wywoływana z CEP
function runColorMatching(paramsJson) {
    try {
        var params = JSON.parse(paramsJson);
        
        // Export layers
        var masterPath = exportLayerToTIFF(params.masterId);
        var targetPath = exportLayerToTIFF(params.targetId);
        
        // ✅ POPRAWNA komunikacja z API przez curl
        var response = sendColorMatchingRequest(
            masterPath, targetPath, params.method, params
        );
        
        // Import result
        var resultLayer = importTIFFAsLayer(response.result_path);
        
        return JSON.stringify({
            status: 'success', 
            layerName: resultLayer.name,
            processingTime: response.processing_time
        });
        
    } catch (error) {
        return JSON.stringify({
            status: 'error', 
            message: error.message
        });
    }
}
```

### 2. ExtendScript ↔ Python API Communication
```javascript
// ❌ BŁĘDNE podejście (CEP Panel - nie zadziała z lokalnymi plikami)
function sendToAPI_WRONG() {
    var xhr = new XMLHttpRequest();
    var formData = new FormData();
    formData.append("master", localFile);  // CORS BLOCK!
    xhr.send(formData);
}

// ✅ POPRAWNE podejście (ExtendScript - curl)
function sendColorMatchingRequest(masterPath, targetPath, method, params) {
    try {
        // Build curl command
        var curlCommand = 'curl -X POST ' +
            'http://127.0.0.1:5000/api/colormatch ' +
            '-F "master=@' + masterPath + '" ' +
            '-F "target=@' + targetPath + '" ' +
            '-F "method=' + method + '"';
        
        if (method === 1 && params.k) {
            curlCommand += ' -F "k=' + params.k + '"';
        }
        
        // Execute curl
        var response = system.callSystem(curlCommand);
        
        if (!response || response.indexOf('"status":"success"') === -1) {
            throw new Error("API request failed");
        }
        
        return JSON.parse(response);
        
    } catch (error) {
        throw new Error("Color matching request failed: " + error.message);
    }
}
```

---

## 🎯 TECHNOLOGY STACK

### Frontend (CEP Panel)
```javascript
Technology Stack:
┌─────────────────┬─────────────────┬─────────────────┐
│     HTML5       │      CSS3       │   JavaScript    │
├─────────────────┼─────────────────┼─────────────────┤
│ • Semantic HTML │ • CSS Variables │ • ES5 Compatible│
│ • Form Controls │ • Flexbox/Grid  │ • Event Handling│
│ • Progress Bars │ • Animations    │ • State Mgmt    │
│ • Status Display│ • Responsive    │ • Error Handling│
└─────────────────┴─────────────────┴─────────────────┘

CEP APIs:
• CSInterface - Communication with ExtendScript
• CEP Events - Application integration
• CSEvent - Custom events
• SystemPath - File system access
```

### Backend (ExtendScript)
```javascript
ExtendScript Capabilities:
┌─────────────────┬─────────────────┬─────────────────┐
│   Photoshop API │   File System   │   System Calls  │
├─────────────────┼─────────────────┼─────────────────┤
│ • Document Ops  │ • File I/O      │ • curl Commands │
│ • Layer Mgmt    │ • Path Handling │ • Process Exec  │
│ • Color Ops     │ • TIFF Export   │ • Response Parse│
│ • Selection Ops │ • Import Ops    │ • Error Handling│
└─────────────────┴─────────────────┴─────────────────┘

Key Features:
• Direct Photoshop integration
• No browser security restrictions
• Full file system access
• System command execution
```

### API Communication
```javascript
Communication Methods:
┌─────────────────┬─────────────────┬─────────────────┐
│   Data Format   │   Transport     │   File Format   │
├─────────────────┼─────────────────┼─────────────────┤
│ • JSON Messages │ • HTTP POST     │ • TIFF Images   │
│ • Form Data     │ • curl Commands │ • Lossless Comp │
│ • Error Objects │ • Local Server  │ • RGB Color     │
│ • Status Reports│ • Port 5000     │ • High Quality  │
└─────────────────┴─────────────────┴─────────────────┘
```

---

## ⚙️ SYSTEM REQUIREMENTS

### Photoshop Compatibility
```
Supported Versions:
• Adobe Photoshop CC 2019+ (Version 20.0+)
• Adobe Photoshop 2020
• Adobe Photoshop 2021
• Adobe Photoshop 2022
• Adobe Photoshop 2023
• Adobe Photoshop 2024

CEP Version Requirements:
• CEP 9.0+ (for CC 2019+)
• CEP 10.0+ (for 2020+)
• CEP 11.0+ (for 2021+)

Operating System:
• Windows 10/11 (64-bit)
• macOS 10.14+ (64-bit)
```

### Python API Requirements
```
Python Environment:
• Python 3.8+
• NumPy 1.19+
• scikit-image 0.17+
• OpenCV 4.5+
• Flask 2.0+
• Pillow 8.0+

System Resources:
• RAM: 8GB minimum, 16GB recommended
• Storage: 2GB free space for temp files
• Network: Local server (127.0.0.1:5000)
```

---

## 🔍 ARCHITECTURE BENEFITS

### Separation of Concerns
```
Layer Responsibilities:
┌─────────────────┬─────────────────┬─────────────────┐
│   CEP Panel     │  ExtendScript   │   Python API    │
├─────────────────┼─────────────────┼─────────────────┤
│ • User Interface│ • PS Integration│ • Algorithms    │
│ • Input Valid.  │ • File Ops      │ • Processing    │
│ • Status Display│ • Layer Ops     │ • Image Analysis│
│ • Error Display │ • API Comm      │ • Math Ops      │
└─────────────────┴─────────────────┴─────────────────┘

Benefits:
✅ Clear responsibility boundaries
✅ Independent development/testing
✅ Modular updates and maintenance
✅ Technology-appropriate solutions
```

### Scalability & Maintenance
```
Advantages:
✅ CEP Panel - Modern web technologies
✅ ExtendScript - Native Photoshop integration
✅ Python API - Advanced image processing
✅ Modular design - Easy to extend
✅ Clear interfaces - Simple debugging
✅ File-based comm - Reliable data transfer
```

---

## 🚀 NEXT STEPS

Po przeczytaniu tego rozdziału, przejdź do:

1. **[Chapter 2 - CEP Extension Setup](./gatto-WORKING-04-photoshop-chapter2.md)** - Konfiguracja manifest.xml i struktura rozszerzenia
2. **[Chapter 3 - CEP Panel Interface](./gatto-WORKING-04-photoshop-chapter3.md)** - HTML/CSS interface panelu
3. **[Chapter 4 - ExtendScript Core Functions](./gatto-WORKING-04-photoshop-chapter4.md)** - ⭐ **KLUCZOWY** - funkcje warstw i komunikacji API

---

*Ten rozdział przedstawia wysokopoziomową architekturę integracji GattoNero z Photoshopem oraz kluczowe korekty zidentyfikowane w code review.*
