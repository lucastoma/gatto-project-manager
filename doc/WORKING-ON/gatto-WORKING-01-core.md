# **GattoNero AI Assistant – Kompletna Dokumentacja Systemu i SOP**

**Status:** ✅ SYSTEM W PEŁNI OPERACYJNY – ZWERYFIKOWANO 08.06.2025  
**Ostatnia aktualizacja:** 08.06.2025  
**Wersja managera:** `server_manager_enhanced.py v2.2.0`

---

## 1. **Architektura i Stack Technologiczny**

### Struktura Projektu (Aktualna)

Struktura została zrefaktoryzowana, aby wspierać modularne algorytmy i solidną infrastrukturę.

```
GattoNeroPhotoshop/
├── app/
│   ├── algorithms/               # ✅ Nowy modularny system algorytmów
│   │   ├── algorithm_01_palette/
│   │   ├── ...
│   ├── api/
│   │   └── routes.py             # ✅ Endpointy API
│   ├── core/                     # ✅ Rdzeń infrastruktury (logger, profiler, monitor)
│   │   ├── development_logger.py
│   │   ├── performance_profiler.py
│   │   └── health_monitor_simple.py
│   ├── scripts/                  # ✅ Skrypty integracyjne dla Adobe Photoshop
│   └── server.py                 # ✅ Główna aplikacja serwera Flask
│
├── logs/                         # ✅ Automatycznie tworzone logi (serwera, managera)
├── results/                      # ✅ Wyniki działania algorytmów
├── uploads/                      # ✅ Tymczasowe pliki
│
├── run_server.py                 # ✅ Skrypt uruchamiający aplikację Flask
├── server_manager_enhanced.py    # ✅ **GŁÓWNE NARZĘDZIE DO ZARZĄDZANIA SERWEREM**
├── server_config.json            # ✅ Konfiguracja serwera i managera
│
├── test_basic.py                 # ✅ Podstawowe testy funkcjonalne API
└── test_algorithm_integration.py # ✅ Testy integracji modularnych algorytmów
```

### Stack Technologiczny (Zweryfikowany)

- **Backend:** Python 3.x + Flask  
- **Computer Vision:** OpenCV (cv2)  
- **Machine Learning:** scikit-learn (K-means)  
- **Narzędzia systemowe:** psutil, requests  
- **Frontend / Integracja:** Adobe CEP (ExtendScript .jsx, HTML/JS)

---

## 2. **Niezawodny Cykl Pracy z Serwerem (SOP)**

Poniżej znajduje się procedura gwarantująca stabilne i przewidywalne środowisko pracy.

### Krok 1: Uruchomienie Serwera w Tle

W głównym folderze projektu uruchom:

```sh
python server_manager_enhanced.py start
```

- **Co się dzieje?** Manager uruchamia serwer Flask w odłączonym procesie, sprawdza poprawność startu i zwalnia terminal.

### Krok 2: Weryfikacja Statusu

Aby sprawdzić, czy serwer działa:

```sh
python server_manager_enhanced.py status
```

- **Poprawny wynik:** Dwie linie `[SUCCESS]`: RUNNING (PID: ...) i RESPONDING.

### Krok 3: Praca i Testowanie

- **Szybki test funkcjonalny:**  
	`python test_basic.py`
- **Test integracji nowych algorytmów:**  
	`python test_algorithm_integration.py`
- **Praca z Photoshopem:** Serwer gotowy na zapytania ze skryptów `.jsx`.

### Krok 4: Zatrzymanie Serwera

Po zakończeniu pracy zatrzymaj serwer:

```sh
python server_manager_enhanced.py stop
```

### Krok 5: Diagnostyka (Gdy coś pójdzie nie tak)

Sprawdź logi błędów:

```sh
python server_manager_enhanced.py logs --file errors
```

- Komenda pokaże dokładny błąd Pythona, który spowodował awarię.

---

## 3. **Kompletny Opis Managera Serwera (`server_manager_enhanced.py`)**

To narzędzie jest centrum dowodzenia. Poniżej wszystkie możliwości:

### `start` – Uruchamianie serwera

```sh
python server_manager_enhanced.py start [opcje]
```

**Opcje:**
- `--auto-restart` – Watchdog automatycznie restartuje serwer po awarii.
- `--no-wait` – Natychmiast zwalnia terminal, nie czeka na pełny start.
- `--port PORT` – Uruchamia serwer na innym porcie.

### `stop` – Zatrzymywanie serwera

```sh
python server_manager_enhanced.py stop [opcje]
```

**Opcje:**
- `--force` – Natychmiastowe zatrzymanie procesu (gdy standardowe nie działa).

### `restart` – Restartowanie serwera

```sh
python server_manager_enhanced.py restart [opcje]
```

**Opcje:**
- `--auto-restart` – Włącza watchdoga po restarcie.

### `status` – Sprawdzanie statusu

```sh
python server_manager_enhanced.py status [opcje]
```

**Opcje:**
- `--detailed` – Dodatkowe informacje: pamięć, CPU, uptime.

### `logs` – Przeglądanie logów

```sh
python server_manager_enhanced.py logs [opcje]
```

**Opcje:**
- `--file [manager|server|errors]` – Wybór pliku logu.
	- `manager`: Logi managera.
	- `server`: Wyjście serwera Flask.
	- `errors`: **Najważniejsze do debugowania**.
- `--tail N` – Ostatnie N linii (domyślnie 20).

### `watch` – Monitoring na żywo

```sh
python server_manager_enhanced.py watch [opcje]
```

**Opcje:**
- `--interval N` – Interwał odświeżania w sekundach (domyślnie 5).

---

## 4. **Konfiguracja (`server_config.json`)**

Manager i serwer są w pełni konfigurowalne przez plik `server_config.json`. Jeśli plik nie istnieje, zostanie utworzony automatycznie.

**Kluczowe opcje:**
- `server.python_executable` – Ścieżka do interpretera Pythona (można ustawić ręcznie).
- `server.startup_command` – Komenda startowa serwera (domyślnie `["<python_exe>", "run_server.py"]`).
- `logging.log_dir` – Folder na logi.

---

Dzięki tej dokumentacji masz solidny fundament do implementacji i testowania kolejnych zaawansowanych algorytmów.