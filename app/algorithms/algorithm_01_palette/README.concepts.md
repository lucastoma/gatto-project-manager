---
version: "1.0"
last_updated: 2025-06-10
author: lucastoma
type: concepts
implementation_status: implemented
tags:
  - concepts
  - planning
  - color_science
aliases:
  - "[[Algorithm 01 - Concepts]]"
---

# Concepts - [[Algorithm 01: Palette Mapping]]

## Problem do rozwiązania

- **Context:** Potrzeba szybkiego i powtarzalnego ujednolicania kolorystyki wielu zdjęć (np. z jednej sesji) tak, aby pasowały do jednego, wzorcowego obrazu.
- **Pain points:** Ręczna korekcja kolorów jest czasochłonna, subiektywna i trudna do zreplikowania w dużej skali. Automatyczne filtry często niszczą oryginalną tonalność obrazu.
- **Success criteria:** Algorytm musi być w stanie przenieść "nastrój" kolorystyczny z obrazu A na obraz B, zachowując przy tym detale obrazu B. Wynik musi być deterministyczny.

## Podejście koncepcyjne

### Algorithm (high-level)

```
1. Wczytaj obraz "Master" i opcjonalnie przeskaluj go dla wydajności (na podstawie parametru 'quality').
2. Użyj algorytmu klasteryzacji (K-Means) lub kwantyzacji (Median Cut), aby znaleźć N dominujących kolorów (paletę).
3. Wczytaj obraz "Target".
4. Dla każdego piksela w obrazie "Target", znajdź percepcyjnie najbliższy kolor w wygenerowanej palecie "Master".
5. Zastąp oryginalny piksel "Target" znalezionym kolorem z palety.
6. Opcjonalnie zastosuj techniki post-processingu, takie jak dithering (dla gładszych przejść) lub edge blending (dla zmiękczenia krawędzi między obszarami kolorów).
7. Zwróć finalny, zmodyfikowany obraz.
```

### Key design decisions

- **K-Means vs Median Cut:** K-Means daje lepsze wyniki percepcyjne, grupując podobne kolory, ale jest wolniejszy. Median Cut jest szybszy i deterministyczny z natury, ale może gorzej oddawać niuanse. Dajemy użytkownikowi wybór.
- **Przestrzeń barw dla metryki:** Porównywanie kolorów w przestrzeni LAB (w `calculate_rgb_distance`) jest bardziej zgodne z ludzką percepcją niż w RGB.
- **Wektoryzacja NumPy:** Użycie `use_vectorized=True` dramatycznie przyspiesza proces mapowania, wykonując obliczenia na całej macierzy pikseli naraz zamiast w pętli.

## Szkic implementacji

### Data structures

```python
# Input (w metodzie process_images)
master_path: str
target_path: str
output_path: str
config: dict = {
    'num_colors': int,
    'quality': int,
    'dithering_method': str, # 'none' | 'floyd_steinberg'
    # ... i inne
}

# Intermediate
palette: list[list[int]] = [[r1, g1, b1], [r2, g2, b2], ...]

# Output
# Zapisany plik obrazu
# LUB
# obiekt PIL.Image.Image (z apply_mapping)
```

### Components to build

- [x] `[[PaletteExtractor]]` - implementacja w `extract_palette()`.
- [x] `[[ColorMapper]]` - implementacja w `apply_mapping_vectorized()` i `apply_mapping_dithered()`.
- [x] `[[ExtremesPreserver]]` - logika do ochrony cieni i świateł w `_apply_extremes_preservation()`.
- [x] `[[EdgeBlender]]` - implementacja w `apply_edge_blending()`.

## Integration points

- **Needs:** `app.core` dla `development_logger` i `performance_profiler`.
- **Provides:** Interfejs `process_images` dla `app.api.routes`, który obsługuje żądania z zewnątrz.

## Next steps

1. **Benchmark** wydajności metod `K-Means` vs `Median Cut` dla różnych `quality`.
2. **Implementacja** większej liczby metod ditheringu.
3. **Optymalizacja** `Edge Blending` z użyciem OpenCV zamiast `scipy`.
