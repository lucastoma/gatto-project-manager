# Zasady Testowania

## Dashboard Deweloperski

Serwer udostępnia wbudowany dashboard deweloperski, który jest podstawowym narzędziem do "testowania na żywo" i monitorowania kondycji aplikacji.

- **Adres:** Dostępny pod adresem `http://127.0.0.1:5000/development/dashboard`
- **Funkcje:**
  - Wyświetla ogólny status systemu (Health Status).
  - Pokazuje metryki wydajności, takie jak średni czas trwania operacji i zużycie pamięci.
  - Umożliwia generowanie raportów wydajności.
  - Zapewnia szybki dostęp do endpointów API zwracających surowe dane o zdrowiu i wydajności.

## Testy Automatyczne

Projekt jest przygotowany do uruchamiania testów automatycznych.

- **Komenda:** Menedżer serwera posiada komendę `run_tests`.
  ```bash
  # Ta komenda nie jest jeszcze zaimplementowana, ale jest na nią miejsce
  # python server_manager_enhanced.py run_tests
  # (Obecnie komenda istnieje, ale brak pliku testowego `test_algorithm_integration.py` w dostarczonym kontekście)
  ```

## Testy Manualne

- **Skrypt `test_simple.jsx`**: Prosty skrypt do weryfikacji, czy środowisko JSX w Photoshopie działa poprawnie i potrafi zapisać plik na pulpicie.
- **Interfejs WebView**: Główne narzędzie do manualnego testowania algorytmów przed ich finalną integracją. Zostało opisane w `rules-webview.md`.
