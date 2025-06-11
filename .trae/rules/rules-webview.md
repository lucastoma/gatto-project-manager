# Zasady Pracy z WebView

## Cel

Interfejs WebView służy do interaktywnego testowania i debugowania algorytmów przetwarzania obrazu w przeglądarce, zanim zostaną one zintegrowane ze skryptami JSX dla Photoshopa. Pozwala to na szybkie prototypowanie i weryfikację działania parametrów.

## Dostępne Interfejsy

Po uruchomieniu serwera, interfejsy testowe są dostępne pod poniższymi adresami:

- **Główna strona WebView:** `http://127.0.0.1:5000/webview/`
- **Testowanie Ekstrakcji Palety (stary panel):** `http://127.0.0.1:5000/webview/algorithm_01`
- **Testowanie Transferu Palety (nowy panel):** `http://127.0.0.1:5000/webview/algorithm_01/transfer`

## Sposób Użycia

1.  Uruchom serwer (`python server_manager_enhanced.py start`).
2.  Otwórz w przeglądarce jeden z powyższych adresów.
3.  Użyj formularzy na stronie, aby:
    - Wgrać obraz wzorcowy (master) i docelowy (target).
    - Skonfigurować parametry algorytmu, takie jak liczba kolorów, metoda ditheringu czy wygładzanie krawędzi.
4.  Kliknij przycisk przetwarzania.
5.  Wynikowy obraz zostanie wyświetlony na stronie, a także zapisany w folderze `app/webview/static/results`.

## Pliki Statyczne

- Logika front-endowa interfejsu znajduje się w plikach `main.js` i `main.css`.
- Szablony HTML znajdują się w folderze `app/webview/templates`.
