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

## 4. Struktura Katalogów Testowych

Projekt wykorzystuje zorganizowaną strukturę testów z numerowanym katalogiem parametrów:

```
app/algorithms/algorithm_XX_nazwa/
├── tests/
│   ├── README.md                           # Dokumentacja testów
│   ├── __init__.py                         # Inicjalizacja pakietu
│   ├── base_test_case.py                   # Klasa bazowa testów
│   ├── test_algorithm_comprehensive.py     # Kompleksowe testy algorytmu
│   ├── test_parameter_01_num_colors.py     # Test parametru num_colors
│   ├── test_parameter_02_distance_metric.py # Test parametru distance_metric
│   ├── test_parameter_03_distance_cache.py  # Test parametru distance_cache
│   └── test_parameter_XX_nazwa.py          # Inne testy parametrów (01-18)
└── algorithm.py                            # Główna logika algorytmu
```

### Konwencja Nazewnictwa Testów Parametrów

Testy parametrów muszą używać numerowanej konwencji:
- Format: `test_parameter_[XX]_[nazwa_parametru].py`
- Numeracja: 01-18 (zgodnie z katalogiem parametrów)
- Przykłady: `test_parameter_01_num_colors.py`, `test_parameter_09_dithering.py`

---

## 5. Workflow Pisania Nowego Testu

Dzięki klasie bazowej i zorganizowanej strukturze, pisanie testów staje się standardowe:

### A) Testy Algorytmu (Comprehensive)

1. Stwórz lub edytuj `test_algorithm_comprehensive.py` w folderze `tests/`.
2. Dodaj na początku pliku `import sys` i `sys.path.append('.')`, aby zapewnić poprawne działanie importów.
3. Zaimportuj klasę `BaseAlgorithmTestCase` z `tests.base_test_case`.
4. Stwórz swoją klasę testową, która dziedziczy po `BaseAlgorithmTestCase`.
5. Wewnątrz swoich metod testowych, użyj `self.create_test_image()` do generowania potrzebnych plików.

### B) Testy Parametrów (Numbered)

1. Stwórz plik `test_parameter_XX_nazwa.py` w folderze `tests/` (gdzie XX to numer 01-18).
2. Użyj tej samej struktury co w testach comprehensive.
3. Skup się na testowaniu konkretnego parametru w różnych scenariuszach.
4. Zaktualizuj `tests/README.md` z statusem implementacji.

### Przykład A: Test Comprehensive

```python
# w pliku /app/algorithms/algorithm_04_new_method/tests/test_algorithm_comprehensive.py
import os
import sys
sys.path.append('.') # Zapewnia, że importy z korzenia projektu działają

from tests.base_test_case import BaseAlgorithmTestCase
from app.algorithms.algorithm_04_new_method.algorithm import NewMethodAlgorithm

class TestNewMethodAlgorithmComprehensive(BaseAlgorithmTestCase):

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
```

### Przykład B: Test Parametru

```python
# w pliku /app/algorithms/algorithm_04_new_method/tests/test_parameter_01_num_colors.py
import os
import sys
sys.path.append('.')

from tests.base_test_case import BaseAlgorithmTestCase
from app.algorithms.algorithm_04_new_method.algorithm import NewMethodAlgorithm

class TestParameter01NumColors(BaseAlgorithmTestCase):

		def test_num_colors_range_2_to_256(self):
				"""
				Testuje parametr num_colors w zakresie 2-256.
				"""
				master_path = self.create_test_image('master.png')
				target_path = self.create_test_image('target.png')
				
				algorithm = NewMethodAlgorithm()
				
				# Test różnych wartości num_colors
				for num_colors in [2, 8, 16, 32, 64, 128, 256]:
						result_path = os.path.join(self.temp_dir, f'result_{num_colors}.png')
						algorithm.process(master_path, target_path, 
															output_path=result_path, 
															num_colors=num_colors)
						self.assertTrue(os.path.exists(result_path))
```

---

## 6. Uruchamianie Testów

### Testy Comprehensive
```bash
# Uruchomienie wszystkich testów comprehensive
python -m unittest app/algorithms/algorithm_XX_nazwa/tests/test_algorithm_comprehensive.py
```

### Testy Parametrów
```bash
# Uruchomienie konkretnego testu parametru
python -m unittest app/algorithms/algorithm_XX_nazwa/tests/test_parameter_01_num_colors.py

# Uruchomienie wszystkich testów parametrów
python -m unittest discover app/algorithms/algorithm_XX_nazwa/tests/ -p "test_parameter_*.py"
```

### Testy z pytest (zalecane)
```bash
# Uruchomienie konkretnego parametru
pytest app/algorithms/algorithm_XX_nazwa/tests/test_parameter_01_num_colors.py -v

# Uruchomienie wszystkich testów parametrów
pytest app/algorithms/algorithm_XX_nazwa/tests/ -k "test_parameter" -v
```

---

## 7. Złote Zasady Testowania (System Rules)

- **STARTUJ SERWER PRZED TESTAMI:** Przed uruchomieniem testów, zawsze wykonaj `python server_manager_enhanced.py start`, aby upewnić się, że środowisko jest gotowe.
- **DZIEDZICZ Z BAZY:** Każda nowa klasa testowa dla algorytmu musi dziedziczyć po `BaseAlgorithmTestCase`.
- **UŻYWAJ NUMEROWANEJ STRUKTURY:** Testy parametrów muszą używać konwencji `test_parameter_XX_nazwa.py` (gdzie XX to 01-18).
- **ORGANIZUJ W KATALOGACH:** Wszystkie testy algorytmu muszą być w dedykowanym folderze `tests/` wewnątrz modułu algorytmu.
- **DOKUMENTUJ TESTY:** Każdy folder `tests/` musi zawierać `README.md` z opisem struktury i statusem implementacji.
- **GENERUJ, NIE PRZECHOWUJ:** Wszystkie dane testowe, zwłaszcza obrazy, muszą być generowane programistycznie za pomocą `self.create_test_image()` wewnątrz metod testowych.
- **NIE SPRZĄTAJ RĘCZNIE:** Nigdy nie pisz własnej logiki usuwania plików w testach. Mechanizm `tearDown` z klasy bazowej zajmuje się tym automatycznie.
- **TESTUJ JEDNO ZJAWISKO:** Każda metoda testowa (`test_*`) powinna weryfikować jeden, konkretny aspekt działania algorytmu.
- **UŻYWAJ ASERCJI:** Każdy test musi kończyć się przynajmniej jedną asercją (np. `self.assertTrue(...)`, `self.assertEqual(...)`), która jednoznacznie określa, czy test zakończył się sukcesem.
- **AKTUALIZUJ STATUS:** Po implementacji testu parametru, zaktualizuj status w `tests/README.md` (✅ Implemented, ⚠️ Partial, ❌ Missing).
