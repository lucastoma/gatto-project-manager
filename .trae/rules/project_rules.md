# Zasady Implementacji Algorytmów (System Prompt)

**Status:** ✅ FINALNE I OBOWIĄZUJĄCE  
**Wersja:** 1.1  
**Data:** 09.06.2025

## Cel

Ustanowienie jednolitego, obligatoryjnego standardu dla implementacji, integracji i testowania nowych algorytmów w projekcie GattoNero.

---

## 1. Filozofia Rozwoju

Nadrzędnym celem jest stworzenie środowiska, w którym deweloper może w 100% skupić się na logice algorytmu, mając pełne zaufanie do otaczającej go infrastruktury. Każdy nowy komponent musi być spójny z istniejącą architekturą, w pełni przetestowany i natychmiast integrowalny.

**Kluczowe pryncypia:**

- **Modularność:** Każdy algorytm to samowystarczalny, niezależny moduł.
- **Spójność:** Wszystkie moduły są budowane według tego samego wzorca.
- **Automatyzacja:** Procesy integracji, testowania i zarządzania środowiskiem są zautomatyzowane i ukryte za prostymi komendami.

---

## 2. Kompletny Workflow Implementacji Nowego Algorytmu

Poniższy proces krok po kroku jest obowiązkowy przy tworzeniu każdego nowego algorytmu.

### Krok 0: Przygotuj Środowisko – Uruchom Serwer

Przed rozpoczęciem jakiejkolwiek pracy deweloperskiej lub testowej, serwer API musi działać w tle.

Użyj poniższej komendy. Jest ona "inteligentna" – jeśli serwer już działa, niczego nie zepsuje. Jeśli nie działa, uruchomi go poprawnie.

```bash
python server_manager_enhanced.py start
```

Zawsze weryfikuj status, aby mieć pewność, że środowisko jest gotowe:

```bash
python server_manager_enhanced.py status
```

---

### Krok 1: Stwórz Strukturę Modułu

W folderze `app/algorithms/` stwórz nowy folder dla swojego algorytmu, trzymając się konwencji nazewnictwa `algorithm_XX_nazwa`. Wewnątrz niego stwórz podstawowy zestaw plików.

**Przykład dla nowego algorytmu "Adaptive Threshold":**

```
/app/algorithms/algorithm_04_adaptive_threshold/
├── __init__.py         # Inicjalizacja pakietu
├── algorithm.py        # Główna logika klasy algorytmu
├── config.py           # Konfiguracja (jeśli potrzebna)
└── tests.py            # Testy jednostkowe dla tego modułu
```

Dodatkowo, wewnątrz tego folderu, stwórz pliki `.implementation-todo.md` i `.implementation-knowledge.md`.

---

### Krok 2: Dokumentuj Zanim Zakodujesz

Zanim napiszesz pierwszą linię kodu, wypełnij pliki `.implementation-todo.md` (definiując plan pracy) oraz `.implementation-knowledge.md` (opisując teorię, założenia i wymagania), korzystając z istniejących szablonów w projekcie.

---

### Krok 3: Zaimplementuj Klasę Algorytmu (`algorithm.py`)

W pliku `algorithm.py` zaimplementuj główną klasę algorytmu.

- Konstruktor klasy (`__init__`) musi inicjalizować loger i profiler.
- Klasa musi udostępniać publiczną metodę `process(self, master_path, target_path, **kwargs)`.
- Plik musi eksportować funkcję-fabrykę, np. `create_adaptive_threshold_algorithm()`.

**Szablon:**

```python
# w pliku /app/algorithms/algorithm_04_adaptive_threshold/algorithm.py
from app.core.development_logger import get_logger
from app.core.performance_profiler import get_profiler

class AdaptiveThresholdAlgorithm:
	def __init__(self, algorithm_id: str = "algorithm_04_adaptive_threshold"):
		self.algorithm_id = algorithm_id
		self.logger = get_logger()
		self.profiler = get_profiler()
		self.logger.info(f"Zainicjalizowano algorytm: {self.algorithm_id}")

	def process(self, master_path, target_path, **kwargs):
		with self.profiler.profile_operation(f"{self.algorithm_id}_process"):
			# ... Logika algorytmu ...
			# ... Walidacja parametrów z kwargs ...
			# ... Zwrócenie ścieżki do pliku wynikowego ...
			pass

def create_adaptive_threshold_algorithm():
	return AdaptiveThresholdAlgorithm()
```

---

### Krok 4: Zarejestruj Algorytm

W pliku `app/algorithms/__init__.py` zaktualizuj słownik `ALGORITHM_REGISTRY`, aby system "wiedział" o istnieniu nowego modułu.

```python
# w pliku /app/algorithms/__init__.py
from .algorithm_04_adaptive_threshold.algorithm import create_adaptive_threshold_algorithm

ALGORITHM_REGISTRY = {
	'algorithm_01_palette': create_palette_mapping_algorithm,
	'algorithm_02_statistical': create_statistical_transfer_algorithm,
	'algorithm_03_histogram': create_histogram_matching_algorithm,
	'algorithm_04_adaptive_threshold': create_adaptive_threshold_algorithm, # Nowy wpis
}
```

---

### Krok 5: Zintegruj z API

W pliku `app/api/routes.py` dodaj nowy wpis do słownika `algorithm_map`, aby udostępnić algorytm pod nowym numerem `method`. To jedyna zmiana wymagana w tym pliku.

```python
# w pliku /app/api/routes.py
algorithm_map = {
	'1': 'algorithm_01_palette',
	'2': 'algorithm_02_statistical',
	'3': 'algorithm_03_histogram',
	'4': 'algorithm_04_adaptive_threshold'  # Nowe mapowanie
}
```

---

### Krok 6: Napisz i Uruchom Testy

W pliku `tests.py` modułu stwórz klasę testową dziedziczącą po `BaseAlgorithmTestCase` i napisz testy jednostkowe. Po implementacji uruchom je.

```bash
# Uruchomienie specyficznych testów (przykład)
python -m unittest app/algorithms/algorithm_04_adaptive_threshold/tests.py
```

---

### Krok 7: Zintegruj z JSX (Opcjonalnie)

Jeśli algorytm wymaga interfejsu w Photoshopie, stwórz dedykowany plik `.jsx` w folderze `app/scripts/`. Pamiętaj o trzymaniu się ustalonych wzorców i protokołu komunikacji CSV.

---

## 3. Złote Zasady Implementacji (System Rules)

- **DZIEDZICZ Z BAZY TESTOWEJ:** Każda nowa klasa testowa dla algorytmu musi dziedziczyć po `BaseAlgorithmTestCase` z `tests/base_test_case.py`.
- **GENERUJ DANE TESTOWE:** Wszystkie dane testowe, zwłaszcza obrazy, muszą być generowane programistycznie w locie za pomocą `self.create_test_image()`. Nie dodawaj plików testowych do repozytorium.
- **TESTUJ JEDNOSTKOWO:** Każdy moduł algorytmu (`algorithm_XX_nazwa`) musi posiadać własny plik `tests.py` z testami weryfikującymi jego logikę w izolacji.
- **REJESTRUJ I MAPUJ:** Każdy nowy algorytm musi być dodany do `ALGORITHM_REGISTRY` oraz do `algorithm_map` w pliku `routes.py`, aby stał się dostępny dla reszty systemu.
- **API ZWRACA TYLKO CSV:** Każdy endpoint, który komunikuje się z `.jsx`, musi zwracać odpowiedź w prostym formacie CSV: `status,dane...`. Nigdy nie zwracaj JSON ani HTML do skryptów JSX.
- **LOGUJ BŁĘDY ZE SZCZEGÓŁAMI:** Każdy blok `except` w warstwie API (`routes.py`) musi wywoływać `logger.error(..., exc_info=True)`, aby zapisać pełny traceback w plikach logów.
- **ZACHOWAJ CZYSTOŚĆ:** Po zakończeniu prac nad nową funkcjonalnością, upewnij się, że nie pozostawiłeś żadnych zakomentowanych bloków kodu, zbędnych plików czy nieużywanych importów.
