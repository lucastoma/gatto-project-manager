// GattoNero Color Matcher - v1.1
// [REFAKTORYZACJA] Wprowadzono centralne okno dialogowe konfiguracji,
// eliminując mylący i podatny na błędy proces wyboru plików.
#target photoshop

// --- KONFIGURACJA ---
var SERVER_URL = "http://127.0.0.1:5000/api/colormatch";

// --- GŁÓWNA FUNKCJA ---
function main() {
    if (app.documents.length < 2) {
        alert("Otwórz co najmniej dwa dokumenty (master i target), aby uruchomić skrypt.");
        return;
    }
    
    // [REFAKTORYZACJA] Cała konfiguracja odbywa się w jednym oknie dialogowym
    var config = showConfigurationDialog();
    if (config === null) {
        return; // Użytkownik anulował
    }

    var tempFolder = new Folder(config.projectRoot + "/temp_jsx");
    if (!tempFolder.exists) tempFolder.create();

    var masterFile = null;
    var targetFile = null;
    
    try {
        // KROK 1: Zapisz wybrane dokumenty do plików tymczasowych
        alert("Rozpoczynam przetwarzanie...\nMaster: " + config.masterDoc.name + "\nTarget: " + config.targetDoc.name);
        
        masterFile = saveDocumentToTIFF(config.masterDoc, tempFolder, "master");
        targetFile = saveDocumentToTIFF(config.targetDoc, tempFolder, "target");

        // KROK 2: Wyślij do serwera
        var response = executeCurl(masterFile, targetFile, config.method, config.k);
        
        // KROK 3: Parsuj odpowiedź
        var result = parseColorMatchResponse(response);
        
        // KROK 4: Otwórz wynikowy plik
        openResultFile(result.filename, config.projectRoot);
        
        alert("Gotowe! Color Matching zakończony.\n\nWynik został otwarty w nowym dokumencie.");

    } catch (e) {
        alert("Wystąpił błąd: \n" + e.message);
    } finally {
        // Posprzątaj po sobie
        cleanupFile(masterFile);
        cleanupFile(targetFile);
    }
}

// [NOWA FUNKCJA] Centralne okno dialogowe
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
    var kInput = kGroup.add("edittext", undefined, "8");
    kInput.characters = 3;

    // --- Przyciski ---
    var buttonGroup = dialog.add("group");
    buttonGroup.orientation = "row";
    buttonGroup.alignment = "right";
    buttonGroup.add("button", undefined, "Anuluj", { name: "cancel" });
    buttonGroup.add("button", undefined, "Uruchom", { name: "ok" });

    if (dialog.show() === 1) { // 1 = OK
        var kValue = parseInt(kInput.text);
        if (isNaN(kValue) || kValue < 4 || kValue > 32) {
            alert("Liczba kolorów musi być w zakresie 4-32.");
            return null;
        }
        if (masterDropdown.selection.index === targetDropdown.selection.index) {
            alert("Dokument Master i Target muszą być różne.");
            return null;
        }

        return {
            masterDoc: app.documents[masterDropdown.selection.index],
            targetDoc: app.documents[targetDropdown.selection.index],
            method: methodDropdown.selection.text.split(":")[0],
            k: kValue,
            projectRoot: new File($.fileName).parent.parent
        };
    }

    return null; // Użytkownik kliknął Anuluj
}

// [REFAKTORYZACJA] Uproszczona funkcja zapisu, przyjmuje cały dokument
function saveDocumentToTIFF(doc, folderPath, prefix) {
    var activeDoc = app.activeDocument;
    app.activeDocument = doc; // Upewnij się, że pracujemy na właściwym dokumencie

    var filePath = new File(folderPath + "/" + prefix + "_" + Date.now() + ".tif");
    var tiffOptions = new TiffSaveOptions();
    tiffOptions.imageCompression = TIFFEncoding.NONE; // Bezstratnie i szybko
    tiffOptions.layers = false; // Zapisz spłaszczony obraz

    doc.saveAs(filePath, tiffOptions, true, Extension.LOWERCASE);
    
    app.activeDocument = activeDoc; // Przywróć aktywny dokument
    return filePath;
}

// Pozostałe funkcje (executeCurl, parseColorMatchResponse, openResultFile, cleanupFile)
// pozostają takie same jak w poprzedniej wersji. Poniżej ich kopia dla kompletności.

function parseColorMatchResponse(response) {
    try {
        response = response.replace(/^\s+|\s+$/g, ""); // trim
        var parts = response.split(",");
        if (parts.length < 1) throw new Error("Pusta odpowiedź serwera");
        
        var status = parts[0];
        if (status === "error") {
            throw new Error("Błąd serwera: " + (parts.length > 1 ? parts.slice(1).join(',') : "Nieznany błąd"));
        }
        if (status !== "success" || parts.length < 3) {
            throw new Error("Nieprawidłowa odpowiedź serwera");
        }
        return { status: status, method: parts[1], filename: parts[2] };
    } catch (e) {
        throw new Error("Błąd parsowania odpowiedzi: " + e.message + "\nOdpowiedź: " + response);
    }
}

function executeCurl(masterFile, targetFile, method, k) {
    var command = 'curl -s -X POST ' +
                  '-F "master_image=@' + masterFile.fsName + '" ' +
                  '-F "target_image=@' + targetFile.fsName + '" ' +
                  '-F "method=' + method + '" ' +
                  '-F "k=' + k + '" ' +
                  SERVER_URL;

    var result = "";
    var tempFolder = masterFile.parent;

    if ($.os.indexOf("Windows") > -1) {
        var cmdFile = new File(tempFolder + "/colormatch_curl.cmd");
        var stdoutFile = new File(tempFolder + "/curl_stdout.txt");
        try {
            cmdFile.open("w");
            cmdFile.encoding = "UTF-8";
            cmdFile.writeln("@echo off");
            cmdFile.writeln(command);
            cmdFile.close();
            if (stdoutFile.exists) stdoutFile.remove();
            app.system('cmd /c ""' + cmdFile.fsName + '" > "' + stdoutFile.fsName + '""');
            
            var maxWaitTime = 15000; // 15 sekund
            var waitInterval = 500;
            var totalWait = 0;
            while (totalWait < maxWaitTime && (!stdoutFile.exists || stdoutFile.length === 0)) {
                $.sleep(waitInterval);
                totalWait += waitInterval;
            }

            if (stdoutFile.exists && stdoutFile.length > 0) {
                stdoutFile.open("r");
                result = stdoutFile.read();
                stdoutFile.close();
            }
        } finally {
            cleanupFile(cmdFile);
            cleanupFile(stdoutFile);
        }
    } else {
        result = app.doScript('do shell script "' + command + '"', Language.APPLESCRIPT);
    }
    
    if (result.replace(/^\s+|\s+$/g, "") === "") {
        throw new Error("Nie otrzymano odpowiedzi od serwera.");
    }
    return result;
}

function openResultFile(filename, projectRoot) {
    var resultsFolder = new Folder(projectRoot + "/results");
    var resultFile = new File(resultsFolder.fsName + "/" + filename);
    
    alert("DEBUG: Szukam pliku:\n" + resultFile.fsName + "\nExists: " + resultFile.exists);
    
    if (!resultFile.exists) {
        throw new Error("Plik wynikowy nie istnieje: " + resultFile.fsName);
    }
    var resultDoc = app.open(resultFile);
    resultDoc.name = "ColorMatch_" + filename;
    
    alert("SUCCESS! Plik otwarty:\n" + filename);
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
