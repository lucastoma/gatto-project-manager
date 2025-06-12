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

### Szybki start (GPU)

```python
from app.algorithms.algorithm_01_palette import PaletteMappingAlgorithmGPU
# lub: from app.algorithms.algorithm_01_palette.algorithm_gpu import PaletteMappingAlgorithmGPU

palette_mapper = PaletteMappingAlgorithmGPU()

params = {
    "num_colors": 24,
    "distance_metric": "weighted_hsv",
    "edge_blur_enabled": True,
    # przykładowe ustawienia zaawansowane
    "gpu_batch_size": 4_000_000,
    "enable_kernel_fusion": True,
}

success = palette_mapper.process_images(
    master_path="master.tif",
    target_path="target.tif",
    output_path="result.tif",
    **params,
)
```

> Algorytm GPU automatycznie wybiera najlepsze dostępne urządzenie OpenCL. Gdy GPU jest niedostępne,
> **rzuca wyjątek** (chyba że ustawisz `edge_blur_device="auto"`, co pozwoli na cichy fallback na CPU).

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

---

## 3. Macierz parametrów (CPU vs GPU)

| Parametr | Typ | Domyślna wartość | Obsługa CPU | Obsługa GPU | Opis |
|----------|-----|------------------|:-----------:|:-----------:|------|
| `num_colors` | `int` | `16` | ✔️ | ✔️ | Rozmiar palety docelowej |
| `palette_method` | `str` | `kmeans` | ✔️ | ✔️ | `kmeans` lub `median_cut` |
| `quality` | `int` | `5` | ✔️ | ✔️ | Im wyżej, tym dokładniej (wolniej) |
| `distance_metric` | `str` | `weighted_hsv` | ✔️ | ✔️ | `weighted_hsv`, `rgb`, `lab` |
| `hue_weight` | `float` | `3.0` | ✔️ | ✔️ | Waga komponentu H (HSV); aktywna dla `weighted_hsv` |
| `saturation_weight` | `float` | `1.0` | ✔️ | ✔️ | Waga S (HSV) |
| `value_weight` | `float` | `1.0` | ✔️ | ✔️ | Waga V (HSV) |
| `dithering_method` | `str` | `none` | ✔️ | ✔️ | `none` lub `floyd_steinberg` |
| `dithering_strength` | `float` | `8.0` | ✔️ | ✔️ | Siła ditheringu (0–16) |
| `inject_extremes` | `bool` | `False` | ✔️ | ✔️ | Dodaje czysty biały/czarny do palety |
| `preserve_extremes` | `bool` | `False` | ✔️ | ✔️ | Zachowuje piksele ekstremalne |
| `extremes_threshold` | `int` | `10` | ✔️ | ✔️ | Próg (0–255) dla ekstremów |
| `edge_blur_enabled` | `bool` | `False` | ⚠️ powolne | ✔️ szybkie | Rozmycie krawędzi |
| `edge_detection_threshold` | `float` | `25.0` | ✔️ | ✔️ | Próg det. krawędzi |
| `edge_blur_radius` | `float` | `1.5` | ✔️ | ✔️ | Promień Gaussa |
| `edge_blur_strength` | `float` | `0.3` | ✔️ | ✔️ | Mieszanie rozmycia |
| `edge_blur_device` | `str` | `auto` | n/a | ✔️ | `auto|gpu|cpu` wymusza urządzenie |
| `use_color_focus` | `bool` | `False` | ✔️ | ✔️ | Wzmocnienie wybranych zakresów |
| `focus_ranges` | `list` | `[]` | ✔️ | ✔️ | List `[ [h1,s1,v1,h2,s2,v2], ... ]` |
| `force_cpu` | `bool` | `False` | ✔️ | 🔧 debug | Wymusza CPU (dla debugowania) |
| `gpu_batch_size` | `int` | `2_000_000` | n/a | ✔️ | Rozmiar partii przesyłanej do GPU |
| `enable_kernel_fusion` | `bool` | `True` | n/a | ✔️ | Łączy kernele OpenCL w jeden |
| `gpu_memory_cleanup` | `bool` | `True` | n/a | ✔️ | Automatyczne czyszczenie buforów |
| `use_64bit_indices` | `bool` | `False` | n/a | ✔️ | Umożliwia obrazy >4 mld px |

**Legenda:** ✔️ = pełne wsparcie, ⚠️ = dostępne, lecz wolniejsze na CPU, n/a = nie dotyczy.

> Jeśli zależy Ci wyłącznie na wydajności GPU, ustaw `force_cpu=False` i `edge_blur_device="gpu"`.
> Fallback na CPU nastąpi tylko w sytuacji krytycznego błędu OpenCL.
