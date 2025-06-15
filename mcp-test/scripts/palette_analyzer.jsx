// GattoNero Palette Analyzer - Prosty format CSV
#target photoshop

// --- KONFIGURACJA ---
var SERVER_URL = "http://127.0.0.1:5000/api/analyze_palette";

function main() {
    if (app.documents.length === 0) {
        alert("Otwórz dokument, aby uruchomić skrypt.");
        return;
    }

    var doc = app.activeDocument;
    if (doc.layers.length === 0) {
        alert("Dokument nie zawiera żadnych warstw.");
        return;
    }

    var activeLayer = doc.activeLayer;
    
    // Zapytaj użytkownika o liczbę kolorów
    var k = prompt("Ile dominujących kolorów chcesz znaleźć?", 8, "Analizator Palety");
    if (k === null) {
        return; // Użytkownik anulował
    }
    k = parseInt(k);
    if (isNaN(k) || k < 1 || k > 50) {
        alert("Podaj liczbę między 1 a 50.");
        return;
    }

    alert("Analizuję paletę kolorów warstwy: \"" + activeLayer.name + "\"\nLiczba kolorów: " + k + "\n\nKliknij OK, aby rozpocząć analizę.");

    // Solidne ścieżki do folderów
    var scriptFile = new File($.fileName);
    var projectRoot = scriptFile.parent.parent; 
    var tempFolder = new Folder(projectRoot + "/temp_jsx");
    if (!tempFolder.exists) tempFolder.create();

    var sourceFile = null;
    
    try {
        // Zapisz aktywną warstwę do pliku TIFF
        sourceFile = saveLayerToPNG(doc, activeLayer, tempFolder, "palette_source");

        // Wyślij do serwera i otrzymaj paletę
        var response = executeCurl(sourceFile, k);
        
        // NOWY PROSTY PARSER - zamiast JSON używamy CSV
        var palette = parseSimpleResponse(response);
        
        // Wizualizuj paletę w dokumencie
        visualizePalette(doc, activeLayer, palette);
        
        alert("Gotowe! Paleta kolorów została wygenerowana.");

    } catch (e) {
        alert("Wystąpił błąd: \n" + e.message);
    } finally {
        // Posprzątaj po sobie
        cleanupFile(sourceFile);
    }
}

function parseSimpleResponse(response) {
    /**
     * Parsuje prostą odpowiedź w formacie:
     * success,4,255,0,0,0,255,255,0,255,0,0,0,255
     * lub
     * error,komunikat błędu
     */
    try {
        // Usuń białe znaki
        response = response.replace(/^\s+|\s+$/g, "");
        
        // Podziel po przecinkach
        var parts = response.split(",");
        
        if (parts.length < 1) {
            throw new Error("Pusta odpowiedź serwera");
        }
        
        var status = parts[0];
        
        if (status === "error") {
            var errorMessage = parts.length > 1 ? parts[1] : "Nieznany błąd";
            throw new Error("Błąd serwera: " + errorMessage);
        }
        
        if (status !== "success") {
            throw new Error("Nieznany status: " + status);
        }
        
        if (parts.length < 2) {
            throw new Error("Brak informacji o liczbie kolorów");
        }
        
        var colorCount = parseInt(parts[1]);
        if (isNaN(colorCount) || colorCount < 1) {
            throw new Error("Nieprawidłowa liczba kolorów: " + parts[1]);
        }
        
        // Sprawdź czy mamy odpowiednią liczbę wartości RGB
        var expectedValues = 2 + (colorCount * 3); // status + count + (r,g,b)*colorCount
        if (parts.length < expectedValues) {
            throw new Error("Za mało wartości kolorów. Oczekiwano: " + expectedValues + ", otrzymano: " + parts.length);
        }
        
        // Parsuj kolory
        var palette = [];
        for (var i = 0; i < colorCount; i++) {
            var r = parseInt(parts[2 + i * 3]);
            var g = parseInt(parts[3 + i * 3]);
            var b = parseInt(parts[4 + i * 3]);
            
            if (isNaN(r) || isNaN(g) || isNaN(b)) {
                throw new Error("Nieprawidłowe wartości RGB dla koloru " + (i + 1));
            }
            
            palette.push([r, g, b]);
        }
        
        return palette;
        
    } catch (e) {
        throw new Error("Błąd parsowania odpowiedzi: " + e.message + "\nOdpowiedź: " + response);
    }
}

// --- FUNKCJE POMOCNICZE ---

function saveLayerToPNG(doc, layer, folderPath, prefix) {
    var originalVisibility = [];
    var activeLayer = doc.activeLayer;

    // Zapisz obecny stan widoczności warstw
    for (var i = 0; i < doc.layers.length; i++) {
        originalVisibility.push({ layer: doc.layers[i], visible: doc.layers[i].visible });
    }

    var filePath = null;

    try {
        // Ukryj wszystkie warstwy oprócz analizowaneи
        for (var i = 0; i < originalVisibility.length; i++) {
            originalVisibility[i].layer.visible = false;
        }
        layer.visible = true;

        filePath = new File(folderPath + "/" + prefix + "_" + Date.now() + ".tif");
        var tiffOptions = new TiffSaveOptions();
        tiffOptions.imageCompression = TIFFEncoding.NONE;
        tiffOptions.byteOrder = ByteOrder.IBM;

        doc.saveAs(filePath, tiffOptions, true, Extension.LOWERCASE);
    } catch(e) {
        throw new Error("Błąd podczas zapisu warstwy do pliku TIFF: " + e.message);
    } finally {
        // Przywróć stan widoczności warstw
        for (var i = 0; i < originalVisibility.length; i++) {
            originalVisibility[i].layer.visible = originalVisibility[i].visible;
        }
        doc.activeLayer = activeLayer;
    }
    return filePath;
}

function executeCurl(sourceFile, k) {
    var command = 'curl -s -X POST ' +
                  '-F "source_image=@' + sourceFile.fsName + '" ' +
                  '-F "k=' + k + '" ' +
                  SERVER_URL;

    var result = "";
    var tempFolder = sourceFile.parent;

    if ($.os.indexOf("Windows") > -1) {
        var cmdFile = new File(tempFolder + "/photoshop_curl.cmd");
        var stdoutFile = new File(tempFolder + "/curl_stdout.txt");
        
        try {
            cmdFile.open("w");
            cmdFile.encoding = "UTF-8";
            cmdFile.writeln("@echo off");
            cmdFile.writeln(command);
            cmdFile.close();
            
            if (stdoutFile.exists) stdoutFile.remove();
            
            app.system('cmd /c ""' + cmdFile.fsName + '" > "' + stdoutFile.fsName + '""');
            
            // Oczekiwanie na odpowiedź serwera
            var maxWaitTime = 10000; // 10 sekund
            var waitInterval = 500;   // sprawdzaj co 0.5 sekundy
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

    // Własna implementacja trim() dla starszych wersji JSX
    var trimmedResult = result.replace(/^\s+|\s+$/g, "");
    if (trimmedResult === "") {
        throw new Error("Nie otrzymano odpowiedzi od serwera lub odpowiedź jest pusta. Upewnij się, że serwer jest uruchomiony.");
    }
    return result;
}

function visualizePalette(doc, sourceLayer, palette) {
    try {
        // Utwórz nową grupę warstw
        var layerSet = doc.layerSets.add();
        layerSet.name = "Analiza Palety - " + sourceLayer.name;
        
        // Utwórz nową warstwę w grupie dla kolorów
        doc.activeLayer = layerSet;
        var paletteLayer = doc.artLayers.add();
        paletteLayer.name = "Paleta Kolorów";
        
        // Konfiguracja wizualizacji - ładniejszy układ w siatce
        var squareSize = 80;  // większe kwadraty
        var spacing = 15;     // większy odstęp
        var startX = 100;     // pozycja startowa X
        var startY = 100;     // pozycja startowa Y
        var columns = 4;      // liczba kolumn w siatce
        
        // Iteruj przez kolory w palecie - układ w siatce
        for (var i = 0; i < palette.length; i++) {
            var color = palette[i];
            var r = color[0];
            var g = color[1];
            var b = color[2];
            
            // Ustaw kolor pierwszego planu w Photoshopie
            var foregroundColor = new SolidColor();
            foregroundColor.rgb.red = r;
            foregroundColor.rgb.green = g;
            foregroundColor.rgb.blue = b;
            app.foregroundColor = foregroundColor;
            
            // Oblicz pozycję kwadratu w siatce
            var x = startX + (i % columns) * (squareSize + spacing);
            var y = startY + Math.floor(i / columns) * (squareSize + spacing + 60); // +60 na etykiety
            
            // Utwórz zaznaczenie prostokątne
            var selectionArray = [
                [x, y],
                [x + squareSize, y],
                [x + squareSize, y + squareSize],
                [x, y + squareSize]
            ];
            doc.selection.select(selectionArray);
            
            // Wypełnij zaznaczenie kolorem
            doc.selection.fill(foregroundColor);
        }
        
        // Usuń zaznaczenie
        doc.selection.deselect();
        
        // Dodaj etykiety pod kwadratami - każda w nowej linii
        addColorLabels(doc, layerSet, palette, startX, startY, squareSize, spacing, columns);
        
    } catch (e) {
        throw new Error("Błąd podczas wizualizacji palety: " + e.message);
    }
}

function addColorLabels(doc, layerSet, palette, startX, startY, squareSize, spacing, columns) {
    try {
        for (var i = 0; i < palette.length; i++) {
            var color = palette[i];
            var r = color[0];
            var g = color[1];
            var b = color[2];
            
            // Konwertuj RGB na HEX
            var hex = "#" + 
                      ("0" + r.toString(16)).slice(-2) + 
                      ("0" + g.toString(16)).slice(-2) + 
                      ("0" + b.toString(16)).slice(-2);
            
            // Oblicz pozycję tekstu - środek kwadratu
            var x = startX + (i % columns) * (squareSize + spacing) + squareSize/2;
            var y = startY + Math.floor(i / columns) * (squareSize + spacing + 60); // +60 na etykiety
            
            // Numer koloru (nad kodem HEX)
            var numberLayer = doc.artLayers.add();
            numberLayer.kind = LayerKind.TEXT;
            numberLayer.name = "Numer " + (i + 1);
            
            var numberItem = numberLayer.textItem;
            numberItem.contents = (i + 1).toString();
            numberItem.position = [x, y + squareSize + 5];  // pod kwadratem
            numberItem.size = 14;
            numberItem.justification = Justification.CENTER;
            
            // Ustaw kolor tekstu na czarny
            var blackColor = new SolidColor();
            blackColor.rgb.red = 0;
            blackColor.rgb.green = 0;
            blackColor.rgb.blue = 0;
            numberItem.color = blackColor;
            
            // Kod HEX (pod numerem)
            var hexLayer = doc.artLayers.add();
            hexLayer.kind = LayerKind.TEXT;
            hexLayer.name = "HEX " + (i + 1);
            
            var hexItem = hexLayer.textItem;
            hexItem.contents = hex.toUpperCase();
            hexItem.position = [x, y + squareSize + 20];  // nieco niżej
            hexItem.size = 10;
            hexItem.justification = Justification.CENTER;
            hexItem.color = blackColor;
            
            // RGB (na samym dole)
            var rgbLayer = doc.artLayers.add();
            rgbLayer.kind = LayerKind.TEXT;
            rgbLayer.name = "RGB " + (i + 1);
            
            var rgbItem = rgbLayer.textItem;
            rgbItem.contents = "R:" + r + " G:" + g + " B:" + b;
            rgbItem.position = [x, y + squareSize + 35];  // jeszcze niżej
            rgbItem.size = 8;
            rgbItem.justification = Justification.CENTER;
            rgbItem.color = blackColor;
            
            // Przenieś wszystkie warstwy tekstowe do grupy
            numberLayer.move(layerSet, ElementPlacement.INSIDE);
            hexLayer.move(layerSet, ElementPlacement.INSIDE);
            rgbLayer.move(layerSet, ElementPlacement.INSIDE);
        }
    } catch (e) {
        // Jeśli dodawanie etykiet się nie powiedzie, nie przerywaj całego procesu
        alert("Ostrzeżenie: Nie udało się dodać etykiet tekstowych: " + e.message);
    }
}

function cleanupFile(file) {
    if (file && file.exists) {
        try {
            file.remove();
        } catch (e) {
            // Ignoruj błędy usuwania
        }
    }
}

// Konwersja liczby na hex (pomocnicza funkcja)
function toHex(n) {
    var hex = n.toString(16);
    return hex.length === 1 ? "0" + hex : hex;
}

// --- URUCHOMIENIE ---
main();
