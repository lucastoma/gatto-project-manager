// GattoNero Color Matcher - v1.2 with Advanced Logging
#target photoshop

// << ZMIANA: Prosta i niezawodna funkcja do logowania na pulpicie.
function writeToLog(message) {
    try {
        var logFile = new File(Folder.desktop + "/gatto_nero_log.txt");
        logFile.open("a"); // "a" oznacza dopisywanie do pliku (append)
        logFile.encoding = "UTF-8";
        logFile.writeln(new Date().toTimeString().substr(0, 8) + ": " + message);
        logFile.close();
    } catch (e) {
        // Ignoruj błędy zapisu do logu, aby nie przerywać głównego skryptu
    }
}

// << ZMIANA: Rozpoczynamy logowanie od razu.
writeToLog("--- Script execution started ---");


// --- KONFIGURACJA ---
var SERVER_URL = "http://127.0.0.1:5000/api/colormatch";

// --- GŁÓWNA FUNKCJA ---
function main() {
    if (app.documents.length < 2) {
        alert("Otwórz co najmniej dwa dokumenty (master i target), aby uruchomić skrypt.");
        writeToLog("Error: Less than 2 documents open. Script terminated.");
        return;
    }
    
    writeToLog("Showing configuration dialog.");
    var config = showConfigurationDialog();
    if (config === null) {
        writeToLog("User cancelled the dialog. Script terminated.");
        return; // Użytkownik anulował
    }
    writeToLog("Configuration received: Method " + config.method + ", Preview: " + config.is_preview);

    var tempFolder = new Folder(config.projectRoot + "/temp_jsx");
    if (!tempFolder.exists) {
        tempFolder.create();
        writeToLog("Created temp folder: " + tempFolder.fsName);
    }

    var masterFile = null;
    var targetFile = null;
    
    try {
        alert("Rozpoczynam przetwarzanie... Sprawdź plik gatto_nero_log.txt na pulpicie, aby śledzić postęp.");
        
        writeToLog("Saving master document: " + config.masterDoc.name);
        masterFile = saveDocumentToTIFF(config.masterDoc, tempFolder, "master");
        writeToLog("Master file saved to: " + masterFile.fsName);
        
        writeToLog("Saving target document: " + config.targetDoc.name);
        targetFile = saveDocumentToTIFF(config.targetDoc, tempFolder, "target");
        writeToLog("Target file saved to: " + targetFile.fsName);

        writeToLog("Executing server request (curl).");
        var response = executeCurl(masterFile, targetFile, config);
        writeToLog("Raw server response: " + response);
        
        writeToLog("Parsing server response.");
        var result = parseColorMatchResponse(response);
        writeToLog("Parsed response successfully. Filename: " + result.filename);
        
        writeToLog("Opening result file.");
        openResultFile(result.filename, config.projectRoot, config.is_preview);
        
    } catch (e) {
        writeToLog("!!! SCRIPT CRASHED !!! Error: " + e.message);
        alert("Wystąpił krytyczny błąd: \n" + e.message + "\n\nSprawdź plik gatto_nero_log.txt na pulpicie po więcej szczegółów.");
    } finally {
        writeToLog("Cleaning up temporary files.");
        cleanupFile(masterFile);
        cleanupFile(targetFile);
        writeToLog("--- Script execution finished ---");
    }
}

function executeCurl(masterFile, targetFile, config) {
    var url = config.is_preview ? "http://127.0.0.1:5000/api/colormatch/preview" : SERVER_URL;
    
    // << ZMIANA: Używamy PEŁNEJ ŚCIEŻKI do curl i usuwamy flagę -s (silent), aby był bardziej "gadatliwy"
    var curlExecutable = "C:/Windows/System32/curl.exe";
    
    var command = '"' + curlExecutable + '" -s -X POST ' + // << DODANO -s (silent)
                  '-F "master_image=@' + masterFile.fsName + '" ' +
                  '-F "target_image=@' + targetFile.fsName + '" ' +
                  '-F "method=' + config.method + '" ' +
                  '-F "k=' + config.k + '" ' +
                  '-F "distance_metric=' + config.distanceMetric + '" ' +
                  '-F "use_dithering=' + config.useDithering + '" ' +
                  '-F "preserve_luminance=' + config.preserveLuminance + '" ' +
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
            
            var maxWaitTime = 15000;
            var waitInterval = 500;
            var totalWait = 0;
            while (totalWait < maxWaitTime && (!stdoutFile.exists || stdoutFile.length === 0) && (!stderrFile.exists || stderrFile.length === 0)) {
                $.sleep(waitInterval);
                totalWait += waitInterval;
            }

            var errorOutput = "";
            if (stderrFile.exists && stderrFile.length > 0) {
                stderrFile.open("r");
                errorOutput = stderrFile.read();
                stderrFile.close();
                writeToLog("CURL stderr: " + errorOutput); // << ZMIANA: Logujemy błąd
            }

            var stdOutput = "";
            if (stdoutFile.exists && stdoutFile.length > 0) {
                stdoutFile.open("r");
                stdOutput = stdoutFile.read();
                stdoutFile.close();
                writeToLog("CURL stdout: " + stdOutput); // << ZMIANA: Logujemy wyjście
            }
            
            if (errorOutput) {
                throw new Error("Błąd wykonania CURL (szczegóły w logu): " + errorOutput);
            }
            
            result = stdOutput;

        } finally {
            cleanupFile(cmdFile);
            cleanupFile(stdoutFile);
            cleanupFile(stderrFile);
        }
    } else { // macOS
        result = app.doScript('do shell script "' + command + '"', Language.APPLESCRIPT);
    }
    
    if (result.replace(/^\s+|\s+$/g, "") === "") {
        throw new Error("Nie otrzymano odpowiedzi od serwera (stdout był pusty).");
    }
    return result;
}

// --- Pozostałe funkcje (bez istotnych zmian) ---
// (showConfigurationDialog, saveDocumentToTIFF, openResultFile, cleanupFile)
function showConfigurationDialog() {
    var docList = [];
    for (var i = 0; i < app.documents.length; i++) {
        docList.push(app.documents[i].name);
    }

    var dialog = new Window("dialog", "GattoNero Color Matcher");
    dialog.orientation = "column";
    dialog.alignChildren = ["fill", "top"];

    // --- Panel Master ---
    var masterPanel = dialog.add("panel", undefined, "1. Wybierz obraz WZORCOWY (Master)");
    masterPanel.alignChildren = "left";
    masterPanel.add("statictext", undefined, "Dokument:");
    var masterDropdown = masterPanel.add("dropdownlist", undefined, docList);
    masterDropdown.selection = 0;

    // --- Panel Target ---
    var targetPanel = dialog.add("panel", undefined, "2. Wybierz obraz DOCELOWY (Target)");
    targetPanel.alignChildren = "left";
    targetPanel.add("statictext", undefined, "Dokument:");
    var targetDropdown = targetPanel.add("dropdownlist", undefined, docList);
    targetDropdown.selection = (docList.length > 1) ? 1 : 0;

    // --- Panel Metody ---
    var methodPanel = dialog.add("panel", undefined, "3. Wybierz metodę i parametry");
    methodPanel.alignChildren = "left";
    methodPanel.add("statictext", undefined, "Metoda dopasowania:");
    var methodDropdown = methodPanel.add("dropdownlist", undefined, [
        "1: Palette Mapping", 
        "2: Statistical Transfer", 
        "3: Histogram Matching"
    ]);
    methodDropdown.selection = 0;

    var kGroup = methodPanel.add("group");
    kGroup.add("statictext", undefined, "Liczba kolorów w palecie (dla Metody 1):");
    var kInput = kGroup.add("edittext", undefined, "16"); // Default to 16
    kInput.characters = 3;

    // --- Panel Opcji Zaawansowanych ---
    var advancedOptionsPanel = dialog.add("panel", undefined, "4. Opcje Zaawansowane");
    advancedOptionsPanel.alignChildren = "left";

    advancedOptionsPanel.add("statictext", undefined, "Metryka odległości:");
    var distanceMetricDropdown = advancedOptionsPanel.add("dropdownlist", undefined, [
        "weighted_rgb: Percepcyjna (domyślna)", 
        "rgb: Szybka (RGB)", 
        "lab: Percepcyjna (LAB)"
    ]);
    distanceMetricDropdown.selection = 0; // Default to weighted_rgb

    var ditheringCheckbox = advancedOptionsPanel.add("checkbox", undefined, "Włącz rozpraszanie (Dithering)");
    ditheringCheckbox.value = false;

    var preserveLuminanceCheckbox = advancedOptionsPanel.add("checkbox", undefined, "Zachowaj jasność oryginału");
    preserveLuminanceCheckbox.value = false;

    // --- Przyciski ---
    var buttonGroup = dialog.add("group");
    buttonGroup.orientation = "row";
    buttonGroup.alignChildren = ["fill", "center"];
    buttonGroup.add("button", undefined, "Anuluj", { name: "cancel" });
    var previewButton = buttonGroup.add("button", undefined, "Generuj Podgląd", { name: "preview" });
    var runButton = buttonGroup.add("button", undefined, "Uruchom", { name: "ok" });

    previewButton.onClick = function() {
        var kValue = parseInt(kInput.text);
        if (isNaN(kValue) || kValue < 4 || kValue > 64) { // Updated range for K
            alert("Liczba kolorów musi być w zakresie 4-64.");
            return;
        }
        if (masterDropdown.selection.index === targetDropdown.selection.index) {
            alert("Dokument Master i Target muszą być różne.");
            return;
        }
        result = {
            masterDoc: app.documents[masterDropdown.selection.index],
            targetDoc: app.documents[targetDropdown.selection.index],
            method: methodDropdown.selection.text.split(":")[0],
            k: kValue,
            distanceMetric: distanceMetricDropdown.selection.text.split(":")[0],
            useDithering: ditheringCheckbox.value,
            preserveLuminance: preserveLuminanceCheckbox.value,
            projectRoot: new File($.fileName).parent.parent,
            is_preview: true
        };
        dialog.close();
    };

    runButton.onClick = function() {
        var kValue = parseInt(kInput.text);
        if (isNaN(kValue) || kValue < 4 || kValue > 64) { // Updated range for K
            alert("Liczba kolorów musi być w zakresie 4-64.");
            return;
        }
        if (masterDropdown.selection.index === targetDropdown.selection.index) {
            alert("Dokument Master i Target muszą być różne.");
            return;
        }
        result = {
            masterDoc: app.documents[masterDropdown.selection.index],
            targetDoc: app.documents[targetDropdown.selection.index],
            method: methodDropdown.selection.text.split(":")[0],
            k: kValue,
            distanceMetric: distanceMetricDropdown.selection.text.split(":")[0],
            useDithering: ditheringCheckbox.value,
            preserveLuminance: preserveLuminanceCheckbox.value,
            projectRoot: new File($.fileName).parent.parent.parent,
            is_preview: false
        };
        dialog.close();
    };

    dialog.show();
    return result;
}
function saveDocumentToTIFF(doc, folderPath, prefix) {
    var activeDoc = app.activeDocument;
    app.activeDocument = doc;

    var filePath = new File(folderPath + "/" + prefix + "_" + Date.now() + ".tif");
    var tiffOptions = new TiffSaveOptions();
    tiffOptions.imageCompression = TIFFEncoding.NONE;
    tiffOptions.layers = false;

    doc.saveAs(filePath, tiffOptions, true, Extension.LOWERCASE);
    
    app.activeDocument = activeDoc;
    return filePath;
}
function parseColorMatchResponse(response) {
    try {
        // Najpierw usuwamy wszystkie możliwe znaki nowej linii z całego tekstu
        var cleaned_response = response.replace(/(\r\n|\n|\r)/gm, "");
        
        // Następnie usuwamy białe znaki z początku i końca
        cleaned_response = cleaned_response.replace(/^\s+|\s+$/g, "");

        var parts = cleaned_response.split(",");
        
        if (parts.length < 1) throw new Error("Pusta odpowiedź serwera");
        
        var status = parts[0];
        if (status === "error") {
            throw new Error("Błąd serwera: " + (parts.length > 1 ? parts.slice(1).join(',') : "Nieznany błąd"));
        }
        if (status !== "success" || parts.length < 3) {
            throw new Error("Nieprawidłowa odpowiedź serwera (oczekiwano 'success,method,filename'): " + cleaned_response);
        }
        
        // Zwracamy obiekt z idealnie czystą nazwą pliku
        return { status: status, method: parts[1], filename: parts[2] };

    } catch (e) {
        throw new Error("Błąd parsowania odpowiedzi: " + e.message + "\nOryginalna odpowiedź: " + response);
    }
}
function openResultFile(filename, projectRoot, is_preview) {
    var resultsFolder = new Folder(projectRoot + "/results");
    var resultFile = new File(resultsFolder.fsName + "/" + filename);
    
    // --- PANCERNA PĘTLA OCZEKIWANIA NA PLIK ---
    var max_wait_ms = 20000; // Maksymalny czas oczekiwania: 20 sekund
    var interval_ms = 500;   // Sprawdzaj co pół sekundy
    var elapsed_ms = 0;
    var fileFound = false;

    writeToLog("Waiting for result file: " + resultFile.fsName);

    while (elapsed_ms < max_wait_ms) {
        if (resultFile.exists) {
            fileFound = true;
            writeToLog("File found after " + elapsed_ms + "ms.");
            break; // Znaleziono plik, wyjdź z pętli
        }
        
        $.sleep(interval_ms); // Czekaj
        elapsed_ms += interval_ms;
    }

    if (!fileFound) {
        throw new Error("Plik wynikowy nie istnieje (nawet po " + (max_wait_ms / 1000) + " sekundach oczekiwania): " + resultFile.fsName);
    }
    // --- KONIEC PĘTLI OCZEKIWANIA ---

    // Otwórz plik, gdy już na pewno istnieje
    if (is_preview) {
        var resultDoc = app.open(resultFile);
        resultDoc.name = "ColorMatch_Preview_" + filename;
        alert("Podgląd wygenerowany! Plik otwarty:\n" + filename + "\n\nZamknij podgląd, aby kontynuować.");
    } else {
        var resultDoc = app.open(resultFile);
        resultDoc.name = "ColorMatch_" + filename;
        alert("Gotowe! Color Matching zakończony.\n\nWynik został otwarty w nowym dokumencie.");
    }
}
function cleanupFile(file) {
    if (file && file.exists) {
        try {
            file.remove();
        } catch (e) { /* ignoruj błędy */ }
    }
}

// --- URUCHOMIENIE ---
main();