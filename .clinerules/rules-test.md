# Zasady Testowania i Zarządzania Danymi Testowymi

**Status:** ✅ FINALNE I OBOWIĄZUJĄCE  
**Wersja:** 1.2  
**Data:** 09.06.2025  

## Cel

Ustanowienie jednolitych, czystych i wydajnych standardów dla wszystkich testów w projekcie **GattoNero**.

---

## 1. Filozofia Testowania

Testy w naszym projekcie muszą być **szybkie, niezależne i powtarzalne**. Oznacza to, że:

- Nie przechowujemy dużych plików testowych w repozytorium. Obrazy i dane są generowane programistycznie.
- Każdy test działa w izolowanym, tymczasowym środowisku.
- Po zakończeniu testu żadne pliki-śmieci nie mogą pozostać na dysku, dzięki mechanizmowi automatycznego sprzątania.

---

## 2. Przygotowanie Środowiska – Uruchomienie Serwera

**Warunek konieczny:** Przed uruchomieniem jakichkolwiek testów (zarówno automatycznych skryptów, jak i manualnych w Photoshopie), serwer API musi działać w tle.

Najprostszą i najbezpieczniejszą metodą jest użycie komendy `start`. Komenda ta jest "inteligentna" – sama sprawdza, czy serwer już działa.

- Jeśli serwer nie działa, zostanie uruchomiony w tle.
- Jeśli serwer już działa, komenda nic nie zrobi i poinformuje o tym w konsoli.

**Jako stały element rozpoczynania pracy, zawsze wykonuj:**

```bash
python server_manager_enhanced.py start
```

Aby upewnić się, że wszystko jest w porządku, możesz dodatkowo zweryfikować status:

```bash
python server_manager_enhanced.py status
```

---

## 3. Uniwersalny Mechanizm: `BaseAlgorithmTestCase`

Aby ustandaryzować powyższe zasady, w projekcie zaimplementowano uniwersalną klasę bazową `BaseAlgorithmTestCase`. Wszystkie nowe klasy testowe dla algorytmów muszą po niej dziedziczyć.

### Lokalizacja i Cel

- Klasa `BaseAlgorithmTestCase` jest jedynym źródłem prawdy dla mechanizmu testowego i znajduje się w pliku:  
	`tests/base_test_case.py`
- Jej głównym celem jest dostarczenie gotowych narzędzi do:
	- **Automatycznego tworzenia środowiska (`setUp`)**: Przed każdym testem tworzony jest unikalny folder tymczasowy.
	- **Automatycznego sprzątania (`tearDown`)**: Po każdym teście folder tymczasowy wraz z całą zawartością jest bezwarunkowo usuwany.
	- **Generowania danych testowych (`create_test_image`)**: Udostępnia prostą metodę do tworzenia plików z obrazami w locie, bez potrzeby przechowywania ich w repozytorium.

Dzięki temu, pisząc testy, deweloper może w pełni skupić się na logice testu, a nie na zarządzaniu plikami.

---

## 4. Workflow Pisania Nowego Testu

Dzięki klasie bazowej, pisanie testów dla nowych algorytmów staje się niezwykle proste i czyste:

1. Stwórz plik `tests.py` w module swojego nowego algorytmu (np. `app/algorithms/algorithm_04/tests.py`).
2. Dodaj na początku pliku `import sys` i `sys.path.append('.')`, aby zapewnić poprawne działanie importów.
3. Zaimprotuj klasę `BaseAlgorithmTestCase` z `tests.base_test_case`.
4. Stwórz swoją klasę testową, która dziedziczy po `BaseAlgorithmTestCase`.
5. Wewnątrz swoich metod testowych, użyj `self.create_test_image()` do generowania potrzebnych plików.

### Przykład: Test dla nowego algorytmu

```python
# w pliku /app/algorithms/algorithm_04_new_method/tests.py
import os
import sys
sys.path.append('.') # Zapewnia, że importy z korzenia projektu działają

from tests.base_test_case import BaseAlgorithmTestCase
from app.algorithms.algorithm_04_new_method.algorithm import NewMethodAlgorithm

class TestNewMethodAlgorithm(BaseAlgorithmTestCase):

		def test_processing_with_solid_colors(self):
				"""
				Testuje, czy algorytm poprawnie przetwarza obrazy o jednolitych kolorach.
				"""
				# Krok 1: Wygeneruj pliki testowe za pomocą metody z klasy bazowej.
				master_path = self.create_test_image('master.png', color=[255, 0, 0])
				target_path = self.create_test_image('target.png', color=[0, 0, 255])
				
				# Krok 2: Uruchom logikę algorytmu
				algorithm = NewMethodAlgorithm()
				output_dir = os.path.dirname(master_path)
				result_path = os.path.join(output_dir, 'result.png')
				
				algorithm.process(master_path, target_path, output_path=result_path)
				
				# Krok 3: Sprawdź wynik (asercja)
				self.assertTrue(os.path.exists(result_path), "Plik wynikowy nie został utworzony.")
				# tearDown() zostanie wywołane automatycznie i posprząta wszystkie pliki.

		def test_handles_random_noise(self):
				"""
				Testuje, czy algorytm nie zawiesza się na losowych danych.
				"""
				master_path = self.create_test_image('master_noise.png') # bez koloru = losowy
				target_path = self.create_test_image('target_noise.png')
				
				# ... logika testu ...
				pass
```

---

## 5. Złote Zasady Testowania (System Rules)

- **STARTUJ SERWER PRZED TESTAMI:** Przed uruchomieniem testów, zawsze wykonaj `python server_manager_enhanced.py start`, aby upewnić się, że środowisko jest gotowe.
- **DZIEDZICZ Z BAZY:** Każda nowa klasa testowa dla algorytmu musi dziedziczyć po `BaseAlgorithmTestCase`.
- **GENERUJ, NIE PRZECHOWUJ:** Wszystkie dane testowe, zwłaszcza obrazy, muszą być generowane programistycznie za pomocą `self.create_test_image()` wewnątrz metod testowych.
- **NIE SPRZĄTAJ RĘCZNIE:** Nigdy nie pisz własnej logiki usuwania plików w testach. Mechanizm `tearDown` z klasy bazowej zajmuje się tym automatycznie.
- **TESTUJ JEDNO ZJAWISKO:** Każda metoda testowa (`test_*`) powinna weryfikować jeden, konkretny aspekt działania algorytmu.
- **UŻYWAJ ASERCJI:** Każdy test musi kończyć się przynajmniej jedną asercją (np. `self.assertTrue(...)`, `self.assertEqual(...)`), która jednoznacznie określa, czy test zakończył się sukcesem.
