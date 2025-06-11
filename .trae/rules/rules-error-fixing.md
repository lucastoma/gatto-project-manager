# Zasady Diagnozowania Błędów

## Sprawdzanie Logów Serwera

Najważniejszym narzędziem do diagnozowania problemów jest przeglądanie logów. Użyj skryptu `server_manager_enhanced.py`, aby łatwo uzyskać do nich dostęp.

- **Logi błędów serwera:** Zawsze zaczynaj od sprawdzania tego pliku. Zawiera on pełne ślady stosu (traceback) dla awarii w kodzie Pythona.

  ```bash
  python server_manager_enhanced.py logs --file errors
  ```

- **Główne logi serwera:** Pokazują ogólną aktywność serwera, przychodzące żądania i informacje o pomyślnych operacjach.

  ```bash
  python server_manager_enhanced.py logs --file server
  ```

- **Logi menedżera serwera:** Zawierają informacje o procesach startu, stopu i restartu.

  ```bash
  python server_manager_enhanced.py logs --file manager
  ```

- **Określanie długości logu:** Możesz kontrolować, ile ostatnich linii chcesz zobaczyć, używając flagi `--tail`.
  ```bash
  # Pokaż ostatnie 100 linii z pliku błędów
  python server_manager_enhanced.py logs --file errors --tail 100
  ```

## Logi Skryptów JSX (Photoshop)

Skrypty JSX (`.jsx`) również prowadzą własny, uproszczony plik logu, który jest bardzo przydatny do śledzenia komunikacji między Photoshopem a serwerem.

- **Lokalizacja:** Plik `gatto_nero_log.txt` jest tworzony bezpośrednio na Pulpicie.
- **Zawartość:** Zapisywane są w nim kluczowe kroki, takie jak:
  - Moment uruchomienia skryptu.
  - Konfiguracja wybrana przez użytkownika w oknie dialogowym.
  - Dokładna komenda `curl` wysyłana do serwera.
  - Surowa odpowiedź otrzymana od serwera.
  - Informacje o ewentualnych błędach krytycznych i powodzeniu operacji.
