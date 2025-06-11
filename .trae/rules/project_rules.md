# Gatto Nero AI - Główne Zasady Projektu

## Filozofia: Bezpiecznie = Szybko

Cały projekt, od kodu serwera po skrypty zarządzające, kieruje się filozofią "Bezpiecznie = Szybko". Oznacza to, że priorytetem jest stabilność, monitorowanie i klarowność kodu. Inwestycja w te obszary pozwala na szybsze wykrywanie błędów, łatwiejsze wdrażanie nowych funkcji i ogólnie sprawniejszy rozwój.

## Struktura Dokumentacji

Ta dokumentacja została podzielona na logiczne moduły, aby ułatwić nawigację:

- **`project_rules.md`**: (Ten plik) Ogólne zasady i przegląd projektu.
- **`rules-server.md`**: Instrukcje dotyczące uruchamiania, zatrzymywania i monitorowania serwera Flask.
- **`rules-error-fixing.md`**: Procedury sprawdzania logów i diagnozowania błędów.
- **`rules-test.md`**: Wytyczne dotyczące testowania algorytmów i funkcjonalności.
- **`rules-webview.md`**: Opis interfejsu webowego do testowania.

## Generowanie Dokumentacji i Łączenie Skryptów

Projekt wykorzystuje skrypty w Pythonie do automatycznego łączenia plików w większe całości, co jest przydatne przy analizie i dostarczaniu kodu.

- **Skrypty `.comb-*.py`**: Służą do agregacji plików (np. `.py`, `.md`, `.jsx`) w jeden duży plik `.md`.
- **Skrypt `.comb-scripts-v3.py`**: Nowsza wersja, która używa plików konfiguracyjnych YAML do definiowania grup plików, co pozwala na bardziej elastyczne i zorganizowane tworzenie zbiorczych dokumentów.
- **`config-selector.py`**: Interaktywne narzędzie do wybierania i uruchamiania różnych konfiguracji dla skryptu `comb-scripts-v3.py`.
