---
version: "1.0"
last_updated: 2025-06-10
author: lucastoma
interface_stable: false
stop_deep_scan: true
tags:
  - api
  - module
  - interface
  - computer_vision
aliases:
  - "[[PaletteMappingAlgorithm]]"
  - "Algorithm 01"
links:
  - "[[README.concepts]]"
  - "[[README.todo]]"
cssclasses:
  - readme-template
---

# [[Algorithm 01: Palette Mapping]]

Moduł do ekstrakcji palety kolorów z obrazu źródłowego i mapowania jej na obraz docelowy. Umożliwia transfer nastroju kolorystycznego między grafikami.

## 1. Overview & Quick Start

### Co to jest

Ten moduł implementuje algorytm dopasowania kolorów oparty na paletach. Jego główna funkcja to analiza obrazu "Master" w celu znalezienia jego dominujących kolorów, a następnie modyfikacja obrazu "Target" tak, by używał wyłącznie kolorów z wygenerowanej palety. Jest to kluczowy komponent systemu `GattoNero AI Manager` do automatyzacji procesów graficznych.

### Szybki start

```python
# Użycie modułu do przetworzenia dwóch obrazów
from app.algorithms.algorithm_01_palette import PaletteMappingAlgorithm

# Inicjalizacja
palette_mapper = PaletteMappingAlgorithm()

# Konfiguracja (opcjonalna, można pominąć)
params = {
    'num_colors': 16,
    'dithering_method': 'floyd_steinberg',
    'preserve_extremes': True
}

# Przetwarzanie
success = palette_mapper.process_images(
    master_path='path/to/master_image.tif',
    target_path='path/to/target_image.tif',
    output_path='path/to/result.tif',
    **params
)

if success:
    print("Obraz został przetworzony pomyślnie!")
```

### Struktura katalogu

```
/app/algorithms/algorithm_01_palette/
├── __init__.py      # Inicjalizuje moduł i eksportuje główne klasy
├── algorithm.py     # Główna implementacja logiki algorytmu
└── config.py        # Struktury danych dla konfiguracji (np. dataclass)
```

### Wymagania

- Python 3.8+
- Biblioteki: `Pillow`, `numpy`, `scikit-learn`, `scipy` (opcjonalnie dla zaawansowanych funkcji)
- Wystarczająca ilość RAM do przetwarzania obrazów

### Najczęstsze problemy

- **Błąd importu `skimage` lub `sklearn`:** Upewnij się, że biblioteki są zainstalowane (`pip install scikit-learn scikit-image`).
- **Niska jakość palety:** Zwiększ parametr `quality` lub `num_colors` przy wywołaniu.
- **Długi czas przetwarzania:** Zmniejsz parametr `quality` lub wyłącz `dithering`. Użyj `use_vectorized=True`.

---

## 2. API Documentation

### Klasy dostępne

#### [[PaletteMappingAlgorithm]]

**Przeznaczenie:** Zarządza całym procesem od ekstrakcji palety po mapowanie kolorów i zapis wyniku.

##### Konstruktor

```python
PaletteMappingAlgorithm(config_path: str = None, algorithm_id: str = "algorithm_01_palette")
```

- **`config_path`** (str, optional): Ścieżka do pliku konfiguracyjnego JSON. Jeśli nie podana, używana jest konfiguracja domyślna.
- **`algorithm_id`** (str, optional): Identyfikator używany w logach.

##### Główne metody

**[[process_images()]]**

```python
result = instance.process_images(master_path: str, target_path: str, output_path: str, **kwargs) -> bool
```

- **Input `master_path`:** Ścieżka do obrazu, z którego zostanie wyekstrahowana paleta.
- **Input `target_path`:** Ścieżka do obrazu, który zostanie zmodyfikowany.
- **Input `output_path`:** Ścieżka, gdzie zostanie zapisany wynik.
- **Input `**kwargs`:** Słownik z parametrami, które nadpisują domyślną konfigurację (np. `num_colors=32`, `dithering_method='floyd_steinberg'`).
- **Output:** `True` jeśli operacja się powiodła, `False` w przeciwnym razie.

**[[extract_palette()]]**

```python
palette = instance.extract_palette(image_path: str, num_colors: int = 16, method: str = 'kmeans') -> list[list[int]]
```

- **Input `image_path`:** Ścieżka do obrazu do analizy.
- **Input `num_colors`:** Liczba dominujących kolorów do znalezienia.
- **Input `method`:** Metoda ekstrakcji ('kmeans' lub 'median_cut').
- **Output:** Lista list, gdzie każda wewnętrzna lista to kolor w formacie `[R, G, B]`. `[[255, 0, 0], [0, 255, 0], ...]`

**[[apply_mapping()]]**

```python
result_image = instance.apply_mapping(target_image_path: str, master_palette: list) -> PIL.Image.Image
```

- **Input `target_image_path`:** Ścieżka do obrazu, który ma zostać przetworzony.
- **Input `master_palette`:** Paleta kolorów uzyskana z `extract_palette()`.
- **Output:** Obiekt `Image` z biblioteki Pillow, gotowy do zapisu lub dalszej modyfikacji.

### Error codes

Moduł nie używa kodów błędów, lecz rzuca wyjątki lub loguje błędy.

- **`ValueError`:** Rzucany, gdy paleta jest pusta lub ma nieprawidłowy format.
- **`FileNotFoundError`:** Rzucany, gdy plik wejściowy nie istnieje.
- **Logi błędów:** Błędy odczytu/zapisu plików lub problemy z bibliotekami są logowane przez `development_logger`.

### Dependencies

**Import:**

```python
from app.algorithms.algorithm_01_palette import PaletteMappingAlgorithm
```

**External dependencies:**

```txt
numpy
Pillow
scikit-learn
scipy
tqdm
```

### File locations

- **Main class:** `./app/algorithms/algorithm_01_palette/algorithm.py` (linie 27-460)
- **Default config:** `./app/algorithms/algorithm_01_palette/algorithm.py` (metoda `default_config`, linia 41)
- **Dataclass config:** `./app/algorithms/algorithm_01_palette/config.py`
