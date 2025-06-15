// GattoNero Color Matcher - v1.6
#target photoshop

function writeToLog(message) {
    try {
        var logFile = new File(Folder.desktop + "/gatto_nero_log.txt");
        logFile.open("a");
        logFile.encoding = "UTF-8";
        logFile.writeln(new Date().toTimeString().substr(0, 8) + ": " + message);
        logFile.close();
    } catch (e) {}
}

writeToLog("--- Script execution started (v1.5) ---");

var SERVER_URL = "http://127.0.0.1:5000/api/colormatch";

function main() {
    if (app.documents.length < 2) {
        alert("Otwórz co najmniej dwa dokumenty (master i target), aby uruchomić skrypt.");
        writeToLog("Error: Less than 2 documents open. Script terminated.");
        return;
    }
    
    writeToLog("Showing configuration dialog.");
    var config = showConfigurationDialog();

    if (config === null) {
        writeToLog("User cancelled the dialog or a critical error occured inside dialog function. Script terminated.");
        return;
    }
    
    writeToLog("Configuration received successfully. Starting process...");
    // Usunięto logowanie całego obiektu config, by nie zaśmiecać, mamy to w logach DEBUG
    
    var tempFolder = new Folder(config.projectRoot + "/temp_jsx");
    if (!tempFolder.exists) {
        tempFolder.create();
        writeToLog("Created temp folder: " + tempFolder.fsName);
    }

    var masterFile = null;
    var targetFile = null;
    
    try {
        alert("Rozpoczynam przetwarzanie... Sprawdź plik gatto_nero_log.txt na pulpicie, aby śledzić postęp.");
        
        masterFile = saveDocumentToTIFF(config.masterDoc, tempFolder, "master");
        targetFile = saveDocumentToTIFF(config.targetDoc, tempFolder, "target");

        var response = executeCurl(masterFile, targetFile, config);
        writeToLog("Raw server response: " + response);
        
        var result = parseColorMatchResponse(response);
        writeToLog("Parsed response successfully. Filename: " + result.filename);
        
        openResultFile(result.filename, config.projectRoot);
        
    } catch (e) {
        writeToLog("!!! SCRIPT CRASHED !!! Error: " + e.message + " (Line: " + e.line + ")");
        alert("Wystąpił krytyczny błąd: \n" + e.message + "\n\nSprawdź plik gatto_nero_log.txt na pulpicie po więcej szczegółów.");
    } finally {
        writeToLog("Cleaning up temporary files.");
        cleanupFile(masterFile);
        cleanupFile(targetFile);
        writeToLog("--- Script execution finished ---");
    }
}

function showConfigurationDialog() {
    var docList = [];
    for (var i = 0; i < app.documents.length; i++) {
        docList.push(app.documents[i].name);
    }

    var dialog = new Window("dialog", "GattoNero Color Matcher v1.5");
    dialog.orientation = "column";
    dialog.alignChildren = ["fill", "top"];

    var masterPanel = dialog.add("panel", undefined, "1. Obraz WZORCOWY (Master)");
    masterPanel.alignChildren = "left";
    masterPanel.add("statictext", undefined, "Dokument:");
    var masterDropdown = masterPanel.add("dropdownlist", undefined, docList);
    masterDropdown.selection = 0;

    var targetPanel = dialog.add("panel", undefined, "2. Obraz DOCELOWY (Target)");
    targetPanel.alignChildren = "left";
    targetPanel.add("statictext", undefined, "Dokument:");
    var targetDropdown = targetPanel.add("dropdownlist", undefined, docList);
    targetDropdown.selection = (docList.length > 1) ? 1 : 0;

    var methodPanel = dialog.add("panel", undefined, "3. Metoda i Główne Parametry");
    methodPanel.alignChildren = "left";
    methodPanel.add("statictext", undefined, "Metoda dopasowania:");
    var methodDropdown = methodPanel.add("dropdownlist", undefined, ["1: Palette Mapping", "2: Statistical Transfer", "3: Histogram Matching"]);
    methodDropdown.selection = 0;

    var kGroup = methodPanel.add("group");
    kGroup.add("statictext", undefined, "Liczba kolorów w palecie (dla Metody 1):");
    var kInput = kGroup.add("edittext", undefined, "16");
    kInput.characters = 3;
    
    var advancedOptionsPanel = dialog.add("panel", undefined, "4. Opcje Zaawansowane (dla Palette Mapping)");
    advancedOptionsPanel.orientation = "column";
    advancedOptionsPanel.alignChildren = "left";
    
    var ditheringGroup = advancedOptionsPanel.add('group');
    ditheringGroup.add("statictext", undefined, "Wygładzanie krawędzi:");
    var ditheringDropdown = ditheringGroup.add("dropdownlist", undefined, ["none: Szybko, ostre krawędzie", "floyd_steinberg: Wolniej, gładkie przejścia"]);
    ditheringDropdown.selection = 0;
    
    advancedOptionsPanel.add('statictext', undefined, 'Ochrona tonów skrajnych:');
    var injectExtremesCheckbox = advancedOptionsPanel.add("checkbox", undefined, "Dodaj czysty czarny/biały do palety");
    injectExtremesCheckbox.value = false;
    
    var preserveGroup = advancedOptionsPanel.add('group');
    var preserveExtremesCheckbox = preserveGroup.add("checkbox", undefined, "Chroń cienie i światła w obrazie docelowym");
    preserveExtremesCheckbox.value = false;

    var thresholdGroup = advancedOptionsPanel.add('group');
    thresholdGroup.add("statictext", undefined, "Próg ochrony (0-255):");
    var thresholdInput = thresholdGroup.add("edittext", undefined, "10");
    thresholdInput.characters = 3;
    thresholdInput.enabled = false;

    preserveExtremesCheckbox.onClick = function() {
        thresholdInput.enabled = this.value;
    };
    
    // === NOWE PARAMETRY EDGE BLENDING ===
    var edgeBlendingPanel = dialog.add("panel", undefined, "5. Wygładzanie Krawędzi (Edge Blending)");
    edgeBlendingPanel.orientation = "column";
    edgeBlendingPanel.alignChildren = "left";
    
    var enableEdgeBlendingCheckbox = edgeBlendingPanel.add("checkbox", undefined, "Włącz wygładzanie krawędzi");
    enableEdgeBlendingCheckbox.value = false;
    
    var edgeDetectionGroup = edgeBlendingPanel.add('group');
    edgeDetectionGroup.add("statictext", undefined, "Próg detekcji krawędzi (0-100):");
    var edgeDetectionThresholdInput = edgeDetectionGroup.add("edittext", undefined, "25");
    edgeDetectionThresholdInput.characters = 3;
    edgeDetectionThresholdInput.enabled = false;
    
    var blurRadiusGroup = edgeBlendingPanel.add('group');
    blurRadiusGroup.add("statictext", undefined, "Promień rozmycia (0.5-5.0):");
    var edgeBlurRadiusInput = blurRadiusGroup.add("edittext", undefined, "1.0");
    edgeBlurRadiusInput.characters = 4;
    edgeBlurRadiusInput.enabled = false;
    
    var blurStrengthGroup = edgeBlendingPanel.add('group');
    blurStrengthGroup.add("statictext", undefined, "Siła rozmycia (0.0-1.0):");
    var edgeBlurStrengthInput = blurStrengthGroup.add("edittext", undefined, "0.5");
    edgeBlurStrengthInput.characters = 4;
    edgeBlurStrengthInput.enabled = false;
    
    enableEdgeBlendingCheckbox.onClick = function() {
        edgeDetectionThresholdInput.enabled = this.value;
        edgeBlurRadiusInput.enabled = this.value;
        edgeBlurStrengthInput.enabled = this.value;
    };
    
    var buttonGroup = dialog.add("group");
    buttonGroup.orientation = "row";
    buttonGroup.alignChildren = ["fill", "center"];
    var runButton = buttonGroup.add("button", undefined, "Uruchom", { name: "ok" });
    var cancelButton = buttonGroup.add("button", undefined, "Anuluj", { name: "cancel" });
    
    var result = null;

    runButton.onClick = function() {
        // === POCZĄTEK LOGOWANIA CHIRURGICZNEGO ===
        try {
            writeToLog("DEBUG: 'Uruchom' clicked. Starting validation.");

            if (masterDropdown.selection.index === targetDropdown.selection.index) {
                alert("Dokument Master i Target muszą być różne.");
                writeToLog("DEBUG: Validation FAILED. Master and Target are the same.");
                return;
            }

            var kValue = parseInt(kInput.text);
            if (isNaN(kValue) || kValue < 4 || kValue > 64) {
                alert("Liczba kolorów musi być w zakresie 4-64.");
                writeToLog("DEBUG: Validation FAILED. Invalid k value: " + kInput.text);
                return;
            }
            writeToLog("DEBUG: kValue is OK: " + kValue);

            var thresholdValue = 0; 
            if (preserveExtremesCheckbox.value) {
                writeToLog("DEBUG: Preserve extremes is checked. Reading threshold value.");
                thresholdValue = parseInt(thresholdInput.text);
                if (isNaN(thresholdValue) || thresholdValue < 0 || thresholdValue > 255) {
                    alert("Gdy opcja ochrony jest włączona, jej próg musi być w zakresie 0-255.");
                    writeToLog("DEBUG: Validation FAILED. Invalid threshold value: " + thresholdInput.text);
                    return;
                }
                writeToLog("DEBUG: thresholdValue is OK: " + thresholdValue);
            } else {
                writeToLog("DEBUG: Preserve extremes is NOT checked.");
            }
            
            // === WALIDACJA PARAMETRÓW EDGE BLENDING ===
            var edgeBlendingEnabled = enableEdgeBlendingCheckbox.value;
            var edgeDetectionThreshold = 25;
            var edgeBlurRadius = 1.0;
            var edgeBlurStrength = 0.5;
            
            if (edgeBlendingEnabled) {
                writeToLog("DEBUG: Edge blending is enabled. Validating parameters.");
                
                edgeDetectionThreshold = parseFloat(edgeDetectionThresholdInput.text);
                if (isNaN(edgeDetectionThreshold) || edgeDetectionThreshold < 0 || edgeDetectionThreshold > 100) {
                    alert("Próg detekcji krawędzi musi być w zakresie 0-100.");
                    writeToLog("DEBUG: Validation FAILED. Invalid edge detection threshold: " + edgeDetectionThresholdInput.text);
                    return;
                }
                
                edgeBlurRadius = parseFloat(edgeBlurRadiusInput.text);
                if (isNaN(edgeBlurRadius) || edgeBlurRadius < 0.5 || edgeBlurRadius > 5.0) {
                    alert("Promień rozmycia musi być w zakresie 0.5-5.0.");
                    writeToLog("DEBUG: Validation FAILED. Invalid edge blur radius: " + edgeBlurRadiusInput.text);
                    return;
                }
                
                edgeBlurStrength = parseFloat(edgeBlurStrengthInput.text);
                if (isNaN(edgeBlurStrength) || edgeBlurStrength < 0.0 || edgeBlurStrength > 1.0) {
                    alert("Siła rozmycia musi być w zakresie 0.0-1.0.");
                    writeToLog("DEBUG: Validation FAILED. Invalid edge blur strength: " + edgeBlurStrengthInput.text);
                    return;
                }
                
                writeToLog("DEBUG: Edge blending parameters validated successfully.");
            } else {
                writeToLog("DEBUG: Edge blending is NOT enabled.");
            }
            
            writeToLog("DEBUG: All validation passed. Creating result object.");

            result = {
                masterDoc: app.documents[masterDropdown.selection.index],
                targetDoc: app.documents[targetDropdown.selection.index],
                method: methodDropdown.selection.text.split(":")[0],
                k: kValue,
                ditheringMethod: (ditheringDropdown.selection.text.split(":")[0]).replace(/^[\s\u00A0]+|[\s\u00A0]+$/g, ''),
                injectExtremes: injectExtremesCheckbox.value,
                preserveExtremes: preserveExtremesCheckbox.value,
                extremesThreshold: thresholdValue,
                // === NOWE PARAMETRY EDGE BLENDING ===
                enableEdgeBlending: edgeBlendingEnabled,
                edgeDetectionThreshold: edgeDetectionThreshold,
                edgeBlurRadius: edgeBlurRadius,
                edgeBlurStrength: edgeBlurStrength,
                projectRoot: new File($.fileName).parent.parent.parent,
                is_preview: false // Ta opcja nie jest już używana w UI, ale może być w przyszłości
            };
            
            writeToLog("DEBUG: Result object created successfully. Closing dialog.");
            dialog.close();

        } catch (e) {
            var errorMessage = "KRYTYCZNY BŁĄD w przycisku 'Uruchom': " + e.message + " (linia: " + e.line + ")";
            writeToLog("!!! " + errorMessage);
            alert(errorMessage);
            // Nie zamykamy okna, ale błąd jest zalogowany
        }
        // === KONIEC LOGOWANIA CHIRURGICZNEGO ===
    };

    cancelButton.onClick = function() {
        writeToLog("DEBUG: 'Anuluj' button clicked.");
        result = null;
        dialog.close();
    };

    dialog.show();
    writeToLog("DEBUG: Dialog closed. Returning result. Is it null? " + (result === null));
    return result;
}

function executeCurl(masterFile, targetFile, config) {
    var url = SERVER_URL;
    var curlExecutable = "C:/Windows/System32/curl.exe";
    
    var command = '"' + curlExecutable + '" -s -X POST ' +
                  '-F "master_image=@' + masterFile.fsName + '" ' +
                  '-F "target_image=@' + targetFile.fsName + '" ' +
                  '-F "method=' + config.method + '" ' +
                  '-F "k=' + config.k + '" ' +
                  '-F "dithering_method=' + config.ditheringMethod + '" ' +
                  '-F "inject_extremes=' + config.injectExtremes + '" ' +
                  '-F "preserve_extremes=' + config.preserveExtremes + '" ' +
                  '-F "extremes_threshold=' + config.extremesThreshold + '" ' +
                  '-F "enable_edge_blending=' + config.enableEdgeBlending + '" ' +
                  '-F "edge_detection_threshold=' + config.edgeDetectionThreshold + '" ' +
                  '-F "edge_blur_radius=' + config.edgeBlurRadius + '" ' +
                  '-F "edge_blur_strength=' + config.edgeBlurStrength + '" ' +
                  url;

    writeToLog("Executing command: " + command);

    var result = "";
    var tempFolder = masterFile.parent;

    if ($.os.indexOf("Windows") > -1) {
        var cmdFile = new File(tempFolder + "/colormatch_curl.cmd");
        var stdoutFile = new File(tempFolder + "/curl_stdout.txt");
        var stderrFile = new File(tempFolder + "/curl_stderr.txt");
        try {
            cmdFile.open("w");
            cmdFile.encoding = "UTF-8";
            cmdFile.writeln("@echo off");
            cmdFile.writeln(command);
            cmdFile.close();

            if (stdoutFile.exists) stdoutFile.remove();
            if (stderrFile.exists) stderrFile.remove();
            
            app.system('cmd /c ""' + cmdFile.fsName + '" 1> "' + stdoutFile.fsName + '" 2> "' + stderrFile.fsName + '""');
            
            var maxWaitTime = 30000;
            var waitInterval = 500;
            var totalWait = 0;
            while (totalWait < maxWaitTime && (!stdoutFile.exists || stdoutFile.length === 0) && (!stderrFile.exists || stderrFile.length > 0)) {
                $.sleep(waitInterval);
                totalWait += waitInterval;
            }

            var errorOutput = "";
            if (stderrFile.exists && stderrFile.length > 0) {
                stderrFile.open("r"); errorOutput = stderrFile.read(); stderrFile.close();
                writeToLog("CURL stderr: " + errorOutput);
            }

            var stdOutput = "";
            if (stdoutFile.exists && stdoutFile.length > 0) {
                stdoutFile.open("r"); stdOutput = stdoutFile.read(); stdoutFile.close();
                writeToLog("CURL stdout: " + stdOutput);
            }
            
            if (errorOutput) { throw new Error("Błąd wykonania CURL: " + errorOutput); }
            if (!stdOutput) { throw new Error("Nie otrzymano odpowiedzi od serwera (pusty stdout)."); }
            
            result = stdOutput;
        } finally {
            cleanupFile(cmdFile); cleanupFile(stdoutFile); cleanupFile(stderrFile);
        }
    } else {
        result = app.doScript('do shell script "' + command + '"', Language.APPLESCRIPT);
    }
    return result;
}

function parseColorMatchResponse(response) {
    try {
        var cleaned_response = response.replace(/(\r\n|\n|\r)/gm, "").replace(/^\s+|\s+$/g, "");
        var parts = cleaned_response.split(",");
        if (parts.length < 3 || parts[0] !== "success") {
             throw new Error("Nieprawidłowa odpowiedź serwera: " + cleaned_response);
        }
        return { status: parts[0], method: parts[1], filename: parts[2] };
    } catch (e) {
        throw new Error("Błąd parsowania odpowiedzi: " + e.message + ". Oryginalna odpowiedź: " + response);
    }
}

function openResultFile(filename, projectRoot) {
    var resultsFolder = new Folder(projectRoot + "/results");
    var resultFile = new File(resultsFolder.fsName + "/" + filename);
    
    var max_wait_ms = 20000;
    var interval_ms = 500;
    var elapsed_ms = 0;
    writeToLog("Waiting for result file: " + resultFile.fsName);

    while (elapsed_ms < max_wait_ms) {
        if (resultFile.exists) {
            writeToLog("File found after " + elapsed_ms + "ms.");
            var resultDoc = app.open(resultFile);
            resultDoc.name = "ColorMatch_" + filename;
            alert("Gotowe! Color Matching zakończony.\n\nWynik został otwarty w nowym dokumencie.");
            return;
        }
        $.sleep(interval_ms);
        elapsed_ms += interval_ms;
    }
    throw new Error("Plik wynikowy nie istnieje (nawet po " + (max_wait_ms / 1000) + "s): " + resultFile.fsName);
}

function saveDocumentToTIFF(doc, folderPath, prefix) {
    writeToLog("Saving document '" + doc.name + "' to TIFF...");
    var activeDoc = app.activeDocument;
    app.activeDocument = doc;
    var filePath = new File(folderPath + "/" + prefix + "_" + Date.now() + ".tif");
    var tiffOptions = new TiffSaveOptions();
    tiffOptions.imageCompression = TIFFEncoding.NONE;
    tiffOptions.layers = false;
    doc.saveAs(filePath, tiffOptions, true, Extension.LOWERCASE);
    app.activeDocument = activeDoc;
    writeToLog("Saved successfully to: " + filePath.fsName);
    return filePath;
}

function cleanupFile(file) {
    if (file && file.exists) {
        try { 
            file.remove();
            writeToLog("Cleaned up temp file: " + file.fsName);
        } catch (e) {}
    }
}

main();