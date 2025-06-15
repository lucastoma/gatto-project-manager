# GattoNero AI Assistant - Color Matching System

## 📋 Opis Projektu

GattoNero AI Assistant to system do dopasowywania kolorów między obrazami z planowaną integracją z Adobe Photoshop. Aktualnie zawiera działający backend Python z algorytmami dopasowywania kolorów i podstawową infrastrukturę serwera.

<ai-to-improve>

1. opisać że działać musimy w venv - to jest chyba oczywiste ale trzeba to podkreśli i to bardzo
2. że nie uruchamia sie bezpośrednio servera, chyba że dla siłowego wymuszenie błedów i ich wykazania przy starcie
3. używamy menadzera serwera, które ma wiele opcji - mam help który informuje o opcjach

i tutaj wklejam help

```
PS D:\projects\gatto-ps-ai> python server_manager_enhanced.py help
usage: server_manager_enhanced.py [-h] {help,start,stop,restart,status,watch,logs} ...

Enhanced Server Manager v2.2.0 - Advanced Flask Server Management for GattoNero AI Assistant

Features:
- Unified watchdog system via 'watch' command
- Configuration-driven setup from 'server_config.json'
- Advanced auto-restart with exponential backoff
- Graceful shutdown with '--force' option
- Structured, TTY-aware logging with log file redirection
- Production-ready deployment capabilities
- Intelligent Python environment detection (VENV vs. SYSTEM)

Usage:
    python server_manager_enhanced.py start [--auto-restart] [--port PORT]
    python server_manager_enhanced.py stop [--force]
    python server_manager_enhanced.py status [--detailed]
    python server_manager_enhanced.py restart [--auto-restart]
    python server_manager_enhanced.py watch [--interval SECONDS]
    python server_manager_enhanced.py logs [--tail LINES] [--file server|manager|errors]

positional arguments:
  {help,start,stop,restart,status,watch,logs}
                        Dostępne komendy
    help                Wyświetla tę wiadomość pomocy.
    start               Uruchamia serwer w tle.
    stop                Zatrzymuje serwer.
    restart             Restartuje serwer.
    status              Pokazuje status serwera.
    watch               Monitoruje serwer na żywo.
    logs                Wyświetla ostatnie logi.

options:
  -h, --help            show this help message and exit

-------------------------------------------------
 GattoNero AI - Przewodnik Szybkiego Startu
-------------------------------------------------
1. Uruchom serwer w tle:
   python server_manager_enhanced.py start

2. Sprawdź, czy działa:
   python server_manager_enhanced.py status

3. Uruchom testy lub pracuj z API/Photoshopem:
   python test_basic.py

4. Zatrzymaj serwer po pracy:
   python server_manager_enhanced.py stop
-------------------------------------------------
Użyj `[komenda] --help` aby zobaczyć opcje dla konkretnej komendy.
PS D:\projects\gatto-ps-ai>
```

dodatkowo .START-ME-FIRST-WORKSPACE.cmd

warto uruchomić zaczynając pracę z workspace (zawiere np. mcp skonfigurowane pod rokspace na docker)

to jest kluczowy element ogólny to co powyzej

elementem biznesowym rozwiązania są algorytmy przetwarzania obrazu

obecnie jest jeden algorytm

[algorytm-mapowania-palety](app/algorithms/algorithm_01_palette)

opcja cpu
opcja gpu - openCL (dla uniwersalności) - obecnie jest zaimplementowane ale jeszcze ostateczne testy real life ne gui (webview)

dodatkowo jest gui (webview) - strona glówna i transfer

[webview-dir](app/webview)

docelowo cały system to współpraca z photoshope na początku proste jsx potem bardziej zaawansowany system

algorytm 2 to placeholder
[algorytm-2](app/algorithms/algorithm_02_statistical)

algorytm 3 to placeholder
[algorytm-3](app/algorithms/algorithm_03_histogram)

testy

</ai-to-improve>
