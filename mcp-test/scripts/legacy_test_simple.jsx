// Prosty test JSX
#target photoshop

try {
    alert("Test JSX działa!");
    
    // Test logowania
    var desktop = Folder.desktop;
    var logFile = new File(desktop + "/jsx_test.txt");
    logFile.open("w");
    logFile.writeln("JSX test działa: " + new Date());
    logFile.close();
    
    alert("Log zapisany na pulpicie!");
    
} catch (e) {
    alert("Błąd: " + e.message);
}
