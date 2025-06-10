# Zasady Obsługi Błędów i Diagnostyki

**Status:** ✅ FINALNE I OBOWIĄZUJĄCE  
**Wersja:** 1.0  
**Data:** 09.06.2025  

**Cel:**  
Ustanowienie jednolitego standardu diagnozowania, naprawiania i weryfikowania błędów w projekcie GattoNero.

---

## 1. Filozofia Obsługi Błędów

Błędy są naturalną częścią procesu tworzenia oprogramowania. Traktujemy je jako dane diagnostyczne wskazujące słabe punkty systemu. Nasz proces opiera się na:

- **Szybkiej identyfikacji:** Błąd musi być natychmiast widoczny i łatwy do zlokalizowania.
- **Skutecznej naprawie:** Poprawka eliminuje przyczynę błędu, nie tylko objawy.
- **Zapobieganiu regresji:** Każda poprawka jest potwierdzona testami, by nie wprowadzać nowych błędów.

---

## 2. Workflow Diagnostyki i Naprawy Błędu

### Krok 1: Identyfikacja Błędu

Zlokalizuj, w której warstwie systemu pojawia się problem:

- **A) Klient (Photoshop):** alert("ERROR: ...") oznacza błąd po stronie serwera.
- **B) Serwer (Terminal/API):** Serwer nie startuje, status NOT RESPONDING lub błąd połączenia – problem z procesem serwera.
- **C) Testy (Konsola):** Skrypt testowy zwraca FAILED/ERROR – błąd w logice lub odpowiedzi API.

### Krok 2: Uruchom Testy Jednostkowe

Przed jakąkolwiek naprawą, uruchom testy jednostkowe dla modułu, w którym występuje błąd:

```bash
# Uruchom wszystkie testy algorytmu
python -m pytest app/algorithms/algorithm_XX_nazwa/tests/

# Uruchom konkretny test parametru (jeśli błąd dotyczy parametru)
python -m pytest app/algorithms/algorithm_XX_nazwa/tests/test_parameter_XX_nazwa.py

# Uruchom kompleksowe testy algorytmu
python -m pytest app/algorithms/algorithm_XX_nazwa/tests/test_algorithm_comprehensive.py

# Uruchom wszystkie testy w projekcie
python -m pytest app/
```

Jeśli testy przechodzą, ale błąd nadal występuje, problem może być w integracji między modułami. Sprawdź status testów w `tests/README.md` - może brakuje testów dla konkretnego parametru.

### Krok 3: Lokalizacja Źródła

Najważniejszy krok: zawsze zaczynaj od sprawdzenia logów serwera:

```bash
python server_manager_enhanced.py logs --file errors
```

W logach znajdziesz traceback wskazujący plik i linię kodu powodującą problem.

### Krok 4: Analiza Błędu

Przeczytaj traceback od dołu do góry. Ostatnia linia to typ błędu (np. `ValueError`), powyżej – ścieżka wywołań prowadząca do błędu.

### Krok 5: Replikacja Błędu (Test)

Przed naprawą napisz test jednostkowy w odpowiednim pliku testowym, który odtwarza błąd:

- **Dla błędów parametrów:** Dodaj test w `test_parameter_XX_nazwa.py`
- **Dla błędów algorytmu:** Dodaj test w `test_algorithm_comprehensive.py`
- **Dla nowych przypadków:** Stwórz nowy plik testowy zgodnie z numerowaną konwencją

*Przykład:* Jeśli błąd to `TypeError` w algorytmie, napisz test wywołujący metodę z błędnym typem danych i sprawdź, czy zgłasza oczekiwany wyjątek.

### Krok 6: Naprawa Błędu

Mając test potwierdzający błąd, wprowadź poprawkę w najniższej możliwej warstwie (np. w logice algorytmu, nie w API).

### Krok 7: Weryfikacja Poprawki

- Uruchom test z Kroku 5 – musi przejść (PASSED).
- Uruchom cały zestaw kluczowych testów, by upewnić się, że nie wprowadziłeś regresji:

```bash
python -m pytest app/algorithms/algorithm_XX_nazwa/tests/
python test_algorithm_integration.py
python test_basic.py
```

### Krok 8: Aktualizuj Dokumentację

Jeśli dodałeś nowy test parametru, zaktualizuj status w `tests/README.md` (✅ Implemented).

Jeśli wszystkie testy przejdą, błąd został poprawnie naprawiony.

---

## 3. Złote Zasady Obsługi Błędów

- **Zaczynaj od logów błędów:**  
	`python server_manager_enhanced.py logs --file errors` to podstawowe narzędzie diagnostyczne.
- **Replikuj błąd testem:**  
	Przed naprawą napisz test w odpowiednim pliku numerowanym jednoznacznie potwierdzający istnienie błędu.
- **Używaj struktury testów:**  
	Organizuj testy według numerowanej konwencji `test_parameter_XX_nazwa.py`.
- **Naprawiaj u źródła:**  
	Poprawki wprowadzaj w najniższej możliwej warstwie.
- **Loguj z kontekstem:**  
	Wszystkie błędy łapane w `try...except` muszą być logowane z `exc_info=True`.
- **Użytkownik dostaje prosty komunikat:**  
	Klient widzi tylko prosty alert, pełna diagnostyka trafia do logów serwera.
- **Aktualizuj status:**  
	Po naprawie zaktualizuj status testów w `tests/README.md`.
- **Testy potwierdzają naprawę:**  
	Przejście wszystkich testów po poprawce jest ostatecznym potwierdzeniem poprawności i bezpieczeństwa zmiany.