# GattoNero AI Assistant - WORKING DOCUMENTATION
## Część 4: Photoshop Integration - Chapter 2: CEP Extension Setup

> **Status:** ✅ POPRAWIONA KONFIGURACJA  
> **Ostatnia aktualizacja:** 2024  
> **Spis treści:** `gatto-WORKING-04-photoshop-toc.md`

---

## 🔧 CEP MANIFEST CONFIGURATION

### ✅ POPRAWIONY manifest.xml (po code review)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<ExtensionManifest Version="7.0" ExtensionBundleId="com.gattonero.colormatching">
  <ExtensionList>
    <Extension Id="com.gattonero.colormatching.panel" Version="1.0.0">
      <HostList>
        <Host Name="PHXS" Version="[20.0,99.9]" />  <!-- Photoshop CC 2019+ -->
        <Host Name="PHSP" Version="[20.0,99.9]" />  <!-- Photoshop CC 2019+ -->
      </HostList>
      <LocaleList>
        <Locale Code="All" />
      </LocaleList>
      <RequiredRuntimeList>
        <RequiredRuntime Name="CSXS" Version="9.0" />
      </RequiredRuntimeList>
    </Extension>
  </ExtensionList>
  
  <ExecutionEnvironment>
    <HostList>
      <Host Name="PHXS" Version="[20.0,99.9]" />
      <Host Name="PHSP" Version="[20.0,99.9]" />
    </HostList>
    <LocaleList>
      <Locale Code="All" />
    </LocaleList>
    <RequiredRuntimeList>
      <RequiredRuntime Name="CSXS" Version="9.0" />
    </RequiredRuntimeList>
  </ExecutionEnvironment>
  
  <DispatchInfoList>
    <Extension Id="com.gattonero.colormatching.panel">
      <DispatchInfo>
        <Resources>
          <!-- ✅ POPRAWKA: MainPath wskazuje na HTML, nie JSX -->
          <MainPath>./index.html</MainPath>
          <!-- ✅ POPRAWKA: ScriptPath wskazuje na ExtendScript -->
          <ScriptPath>./host/main.jsx</ScriptPath>
        </Resources>
        <Lifecycle>
          <AutoVisible>true</AutoVisible>
        </Lifecycle>
        <UI>
          <Type>Panel</Type>
          <Menu>GattoNero Color Matching</Menu>
          <Geometry>
            <Size>
              <Width>320</Width>
              <Height>600</Height>
            </Size>
            <MinSize>
              <Width>280</Width>
              <Height>400</Height>
            </MinSize>
            <MaxSize>
              <Width>800</Width>
              <Height>1200</Height>
            </MaxSize>
          </Geometry>
        </UI>
      </DispatchInfo>
    </Extension>
  </DispatchInfoList>
</ExtensionManifest>
```

### 🚨 KRYTYCZNE KOREKTY

#### Przed (BŁĘDNE):
```xml
<Resources>
    <MainPath>./client.jsx</MainPath>        <!-- ❌ JSX nie może być MainPath -->
    <ScriptPath>./js/main.js</ScriptPath>    <!-- ❌ JS nie może być ScriptPath -->
</Resources>
```

#### Po (POPRAWNE):
```xml
<Resources>
    <MainPath>./index.html</MainPath>        <!-- ✅ HTML dla interfejsu -->
    <ScriptPath>./host/main.jsx</ScriptPath> <!-- ✅ JSX dla ExtendScript -->
</Resources>
```

#### Wyjaśnienie:
- **MainPath** = główny plik interfejsu (HTML) ładowany w panelu CEP
- **ScriptPath** = plik ExtendScript (.jsx) ładowany równolegle z panelem
- **CEP Panel** (HTML/CSS/JS) + **ExtendScript** (.jsx) = komunikują się przez CSInterface

---

## 📁 POPRAWIONA STRUKTURA ROZSZERZENIA

### Nowa organizacja plików (po korekcie):
```
app/scripts/                           # Extension Root
├── CSXS/                              # ✅ CEP Configuration
│   └── manifest.xml                   # Extension manifest
├── index.html                         # ✅ POPRAWKA: MainPath (Panel UI)
├── host/                              # ✅ POPRAWKA: ExtendScript files
│   ├── main.jsx                       # ✅ POPRAWKA: ScriptPath (główny)
│   ├── layer-operations.jsx           # Layer management functions
│   ├── file-operations.jsx            # File export/import functions
│   ├── api-communication.jsx          # API calls via curl
│   └── utils.jsx                      # ExtendScript utilities
├── css/                               # ✅ Panel Styling
│   ├── main.css                       # Main styles
│   ├── components.css                 # Component styles
│   └── themes.css                     # Color themes
├── js/                                # ✅ CEP Panel Logic
│   ├── main.js                        # Main application logic
│   ├── ui.js                          # UI management
│   ├── api.js                         # CEP side API handling
│   ├── events.js                      # Event management
│   └── utils.js                       # Utility functions
├── assets/                            # ✅ Resources
│   ├── icons/                         # UI icons
│   │   ├── logo-16.png
│   │   ├── logo-32.png
│   │   └── buttons/
│   └── images/                        # Interface graphics
│       ├── background.jpg
│       └── patterns/
└── debug/                             # ✅ Development Tools
    ├── debug.html                     # Debug panel
    ├── debug.js                       # Debug utilities
    └── test-data/                     # Test resources
```

### Zmiana nazw plików (wyjaśnienie):
```
STARA NAZWA → NOWA NAZWA (powód zmiany)
─────────────────────────────────────────────────
client.jsx → index.html (MainPath musi być HTML)
main.js → js/main.js (organizacja, CEP panel logic)
[nowy] → host/main.jsx (ScriptPath, ExtendScript)
[rozdzielone] → host/*.jsx (organizacja ExtendScript)
```

---

## 🎯 WERSJONOWANIE I KOMPATYBILNOŚĆ

### CEP Version Compatibility Matrix
```
CEP Version │ Photoshop Version │ Status │ Recommended
────────────┼───────────────────┼────────┼──────────
CEP 9.0     │ CC 2019 (20.0)    │   ✅   │ Minimum
CEP 10.0    │ CC 2020 (21.0)    │   ✅   │ Good
CEP 11.0    │ CC 2021 (22.0)    │   ✅   │ Better
CEP 12.0    │ CC 2022 (23.0)    │   ✅   │ Recommended
CEP 13.0    │ CC 2023 (24.0)    │   ✅   │ Latest
```

### Host Names Reference
```xml
<!-- Photoshop Host Names -->
<Host Name="PHXS" ... />  <!-- Photoshop Standard -->
<Host Name="PHSP" ... />  <!-- Photoshop Extended -->

<!-- Version Range Examples -->
Version="[20.0,99.9]"     <!-- CC 2019+ (recommended) -->
Version="[22.0,99.9]"     <!-- CC 2021+ (if using newer features) -->
Version="24.0"            <!-- Exact version (not recommended) -->
```

### Extension ID Guidelines
```
Bundle ID Structure:
com.{company}.{product}
├── com.gattonero.colormatching          # Bundle ID
└── com.gattonero.colormatching.panel    # Extension ID

Best Practices:
✅ Use reverse domain notation
✅ Keep IDs unique across all extensions
✅ Use descriptive names
❌ Don't use spaces or special characters
❌ Don't use Adobe-reserved prefixes
```

---

## 🔄 LIFECYCLE CONFIGURATION

### Panel Lifecycle Options
```xml
<Lifecycle>
    <AutoVisible>true</AutoVisible>           <!-- Show on Photoshop start -->
    <StartOn>
        <Event>applicationActivate</Event>    <!-- Start on PS activation -->
    </StartOn>
</Lifecycle>

<!-- Alternative configurations: -->

<!-- Manual start only -->
<Lifecycle>
    <AutoVisible>false</AutoVisible>
</Lifecycle>

<!-- Start on document open -->
<Lifecycle>
    <AutoVisible>false</AutoVisible>
    <StartOn>
        <Event>documentAfterActivate</Event>
    </StartOn>
</Lifecycle>
```

### Panel UI Configuration
```xml
<UI>
    <Type>Panel</Type>                        <!-- Panel type -->
    <Menu>GattoNero Color Matching</Menu>     <!-- Menu name -->
    <Geometry>
        <Size>
            <Width>320</Width>                <!-- Default width -->
            <Height>600</Height>              <!-- Default height -->
        </Size>
        <MinSize>
            <Width>280</Width>                <!-- Minimum width -->
            <Height>400</Height>              <!-- Minimum height -->
        </MinSize>
        <MaxSize>
            <Width>800</Width>                <!-- Maximum width -->
            <Height>1200</Height>             <!-- Maximum height -->
        </MaxSize>
    </Geometry>
    <Icons>
        <Icon Type="Normal">./assets/icons/logo-16.png</Icon>
        <Icon Type="RollOver">./assets/icons/logo-16-hover.png</Icon>
        <Icon Type="DarkNormal">./assets/icons/logo-16-dark.png</Icon>
        <Icon Type="DarkRollOver">./assets/icons/logo-16-dark-hover.png</Icon>
    </Icons>
</UI>
```

---

## 🛡️ PERMISSIONS & SECURITY

### Script Access Permissions
```xml
<!-- Enable ExtendScript execution -->
<RequiredRuntimeList>
    <RequiredRuntime Name="CSXS" Version="9.0" />
</RequiredRuntimeList>

<!-- Additional permissions (if needed) -->
<CEFCommandLine>
    <Parameter>--allow-file-access-from-files</Parameter>
    <Parameter>--allow-file-access</Parameter>
    <Parameter>--enable-nodejs</Parameter>          <!-- For Node.js integration -->
</CEFCommandLine>
```

### Development vs Production Settings
```xml
<!-- Development (Debug Mode) -->
<ExtensionManifest Version="7.0" ExtensionBundleId="com.gattonero.colormatching.dev">
    <!-- CEFCommandLine for debugging -->
    <CEFCommandLine>
        <Parameter>--enable-developer-tools</Parameter>
        <Parameter>--remote-debugging-port=8088</Parameter>
    </CEFCommandLine>
    
<!-- Production (Release Mode) -->
<ExtensionManifest Version="7.0" ExtensionBundleId="com.gattonero.colormatching">
    <!-- No debug parameters -->
```

---

## 🚀 INSTALLATION PROCEDURES

### Method 1: Development Installation (Manual)
```bash
# Windows - Copy to CEP extensions folder
xcopy /E /I "app\scripts" "C:\Program Files (x86)\Common Files\Adobe\CEP\extensions\GattoNero"

# Enable debug mode (Windows Registry)
reg add "HKEY_CURRENT_USER\Software\Adobe\CSXS.9" /v PlayerDebugMode /t REG_SZ /d 1

# Restart Photoshop
# Access via: Window > Extensions > GattoNero Color Matching
```

```bash
# macOS - Copy to CEP extensions folder
cp -r app/scripts/ "/Library/Application Support/Adobe/CEP/extensions/GattoNero/"

# Enable debug mode (macOS Terminal)
defaults write com.adobe.CSXS.9 PlayerDebugMode 1

# Restart Photoshop
# Access via: Window > Extensions > GattoNero Color Matching
```

### Method 2: Production Installation (ZXP Package)
```bash
# Create signed ZXP package (requires certificate)
zxpsign -sign app/scripts/ GattoNero.zxp certificate.p12 password timestamp_server

# Install via Adobe Extension Manager
# Or distribute as .zxp file for user installation
```

### Method 3: Development with Hot Reload
```bash
# Enable CEP debug mode
reg add "HKEY_CURRENT_USER\Software\Adobe\CSXS.9" /v PlayerDebugMode /t REG_SZ /d 1
reg add "HKEY_CURRENT_USER\Software\Adobe\CSXS.9" /v LogLevel /t REG_SZ /d 6

# Create symbolic link for development
mklink /D "C:\Program Files (x86)\Common Files\Adobe\CEP\extensions\GattoNero" "D:\Unity\Projects\GattoNeroPhotoshop\app\scripts"

# Access Chrome DevTools at: http://localhost:8088
```

---

## 🔍 VALIDATION & TESTING

### Manifest Validation
```bash
# Validate manifest.xml syntax
xmllint --noout --schema manifest.xsd app/scripts/CSXS/manifest.xml

# Check CEP version compatibility
# Use Adobe CEP Test Suite (if available)
```

### Extension Loading Test
```javascript
// Test ExtendScript availability (in CEP Panel console)
csInterface.evalScript('alert("ExtendScript is working!");');

// Test extension load status
csInterface.getExtensions(function(extensions) {
    console.log('Loaded extensions:', extensions);
});
```

### Folder Permissions Test
```bash
# Verify write permissions to temp folder
# Test file creation in uploads directory
echo "test" > "uploads/test.txt"
```

---

## 🐛 COMMON SETUP ISSUES

### Issue 1: Extension Not Visible
```
Problem: Extension doesn't appear in Window > Extensions
Diagnosis:
1. Check manifest.xml syntax errors
2. Verify file paths in MainPath/ScriptPath
3. Check CEP version compatibility
4. Verify PlayerDebugMode setting

Solution:
1. Validate XML with xmllint
2. Ensure paths exist and are correct
3. Update CEP version in manifest
4. Set PlayerDebugMode = 1
```

### Issue 2: ExtendScript Not Loading
```
Problem: CEP Panel loads but ExtendScript functions fail
Diagnosis:
1. Check ScriptPath points to valid .jsx file
2. Verify ExtendScript syntax
3. Check file permissions

Solution:
1. Correct ScriptPath in manifest.xml
2. Test ExtendScript syntax separately
3. Set proper file permissions
```

### Issue 3: File Path Issues
```
Problem: "File not found" errors
Diagnosis:
1. Check absolute vs relative paths
2. Verify folder structure matches manifest
3. Check file extensions (.jsx vs .js)

Solution:
1. Use correct relative paths from extension root
2. Reorganize files to match manifest structure
3. Ensure proper file extensions
```

---

## 🚀 NEXT STEPS

Po skonfigurowaniu CEP Extension, przejdź do:

1. **[Chapter 3 - CEP Panel Interface](./gatto-WORKING-04-photoshop-chapter3.md)** - Tworzenie HTML/CSS interfejsu
2. **[Chapter 4 - ExtendScript Core Functions](./gatto-WORKING-04-photoshop-chapter4.md)** - Implementacja funkcji ExtendScript
3. **[Chapter 7 - Deployment & Troubleshooting](./gatto-WORKING-04-photoshop-chapter7.md)** - Zaawansowane rozwiązywanie problemów

---

*Ten rozdział przedstawia poprawioną konfigurację CEP Extension na podstawie code review oraz procedury instalacji i debugowania.*
