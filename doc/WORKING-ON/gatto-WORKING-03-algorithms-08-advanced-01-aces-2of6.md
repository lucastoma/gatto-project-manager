# ACES Color Space Transfer - Część 2of6: Pseudokod i Architektura 🏗️

> **Seria:** ACES Color Space Transfer  
> **Część:** 2 z 6 - Pseudokod i Architektura  
> **Wymagania:** [1of6 - Teoria i Podstawy](gatto-WORKING-03-algorithms-08-advanced-01-aces-1of6.md)  
> **Następna część:** [3of6 - Implementacja Core](gatto-WORKING-03-algorithms-08-advanced-01-aces-3of6.md)

---

## 1. Architektura Systemu ACES

### 1.1 Ogólna Struktura

```
┌─────────────────────────────────────────────────────────────┐
│                    ACES Color Transfer System               │
├─────────────────────────────────────────────────────────────┤
│  Input Layer                                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Image Loader│ │Color Profile│ │ Parameters  │          │
│  │   Module    │ │   Manager   │ │   Parser    │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│  Core Processing Layer                                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │    Color    │ │    ACES     │ │    Tone     │          │
│  │ Conversion  │ │  Transform  │ │   Mapping   │          │
│  │   Engine    │ │   Engine    │ │   Engine    │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│  Analysis Layer                                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Statistics  │ │   Quality   │ │ Performance │          │
│  │  Analyzer   │ │  Evaluator  │ │   Monitor   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│  Output Layer                                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │   Result    │ │   Metadata  │ │    Debug    │          │
│  │  Generator  │ │   Manager   │ │   Logger    │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Przepływ Danych

```
Input Image → Color Profile Detection → sRGB Validation
     ↓
Linear RGB Conversion → ACES AP0 Transform → Statistics Analysis
     ↓
Target Analysis → Transform Calculation → Quality Prediction
     ↓
ACES Processing → Tone Mapping → Gamut Compression
     ↓
ACES to sRGB → Gamma Correction → Output Validation
     ↓
Result Image → Metadata → Quality Report
```

---

## 2. Główny Pseudokod Algorytmu

### 2.1 Funkcja Główna

```
FUNCTION aces_color_transfer(source_image, target_image, parameters):
    // === ETAP 1: WALIDACJA I PRZYGOTOWANIE ===
    VALIDATE_INPUT(source_image, target_image, parameters)
    
    // Detekcja profili kolorów
    source_profile = DETECT_COLOR_PROFILE(source_image)
    target_profile = DETECT_COLOR_PROFILE(target_image)
    
    // === ETAP 2: KONWERSJA DO ACES ===
    source_aces = CONVERT_TO_ACES(source_image, source_profile)
    target_aces = CONVERT_TO_ACES(target_image, target_profile)
    
    // === ETAP 3: ANALIZA STATYSTYK ===
    source_stats = ANALYZE_ACES_STATISTICS(source_aces)
    target_stats = ANALYZE_ACES_STATISTICS(target_aces)
    
    // === ETAP 4: OBLICZENIE TRANSFORMACJI ===
    transform_data = CALCULATE_ACES_TRANSFORM(
        source_stats, 
        target_stats, 
        parameters.adaptation_method
    )
    
    // === ETAP 5: PREDYKCJA JAKOŚCI ===
    quality_prediction = PREDICT_QUALITY(
        source_stats, target_stats, transform_data
    )
    
    IF quality_prediction.confidence < parameters.min_confidence:
        RETURN ERROR("Low quality prediction")
    END IF
    
    // === ETAP 6: APLIKACJA TRANSFORMACJI ===
    result_aces = APPLY_ACES_TRANSFORMATION(
        source_aces, 
        transform_data, 
        parameters
    )
    
    // === ETAP 7: POST-PROCESSING ===
    IF parameters.use_tone_mapping:
        result_aces = APPLY_TONE_MAPPING(result_aces, parameters.tone_curve)
    END IF
    
    IF parameters.preserve_luminance:
        result_aces = PRESERVE_LUMINANCE(source_aces, result_aces)
    END IF
    
    IF parameters.gamut_compression:
        result_aces = COMPRESS_GAMUT(result_aces, parameters.compression_params)
    END IF
    
    // === ETAP 8: KONWERSJA WYJŚCIOWA ===
    result_image = CONVERT_FROM_ACES(result_aces, parameters.output_profile)
    
    // === ETAP 9: WALIDACJA WYNIKU ===
    quality_metrics = EVALUATE_RESULT_QUALITY(
        source_image, result_image, target_image
    )
    
    // === ETAP 10: GENEROWANIE RAPORTU ===
    report = GENERATE_PROCESSING_REPORT(
        source_stats, target_stats, transform_data, quality_metrics
    )
    
    RETURN {
        image: result_image,
        quality: quality_metrics,
        report: report,
        metadata: EXTRACT_METADATA(transform_data)
    }
END FUNCTION
```

### 2.2 Konwersja do ACES

```
FUNCTION convert_to_aces(image, color_profile):
    // Normalizacja do [0,1]
    IF image.dtype == UINT8:
        normalized_image = image / 255.0
    ELSE IF image.dtype == UINT16:
        normalized_image = image / 65535.0
    ELSE:
        normalized_image = image
    END IF
    
    // Konwersja do linear RGB
    linear_rgb = APPLY_INVERSE_GAMMA(normalized_image, color_profile.gamma)
    
    // Chromatic adaptation (jeśli potrzebna)
    IF color_profile.white_point != D60:
        adapted_rgb = CHROMATIC_ADAPTATION(
            linear_rgb, 
            color_profile.white_point, 
            D60
        )
    ELSE:
        adapted_rgb = linear_rgb
    END IF
    
    // Transformacja do ACES AP0
    transform_matrix = GET_TRANSFORM_MATRIX(
        color_profile.primaries, 
        ACES_AP0_PRIMARIES
    )
    
    aces_image = APPLY_MATRIX_TRANSFORM(adapted_rgb, transform_matrix)
    
    // Ograniczenie do zakresu ACES
    aces_image = CLAMP(aces_image, 0.0, 65504.0)
    
    RETURN aces_image
END FUNCTION
```

### 2.3 Analiza Statystyk ACES

```
FUNCTION analyze_aces_statistics(aces_image):
    // Reshape do 2D array (pixels × channels)
    pixels = RESHAPE(aces_image, [-1, 3])
    
    // Podstawowe statystyki
    mean_rgb = CALCULATE_MEAN(pixels, axis=0)
    std_rgb = CALCULATE_STD(pixels, axis=0)
    min_rgb = CALCULATE_MIN(pixels, axis=0)
    max_rgb = CALCULATE_MAX(pixels, axis=0)
    
    // Percentyle
    percentiles = {}
    FOR p IN [1, 5, 25, 50, 75, 95, 99]:
        percentiles[p] = CALCULATE_PERCENTILE(pixels, p, axis=0)
    END FOR
    
    // Histogramy dla każdego kanału
    histograms = {}
    FOR channel IN [0, 1, 2]:  // R, G, B
        hist, bins = CALCULATE_HISTOGRAM(
            pixels[:, channel], 
            bins=1024, 
            range=[0, 65504]
        )
        histograms[channel] = {histogram: hist, bins: bins}
    END FOR
    
    // Luminancja w ACES
    luminance = 0.2722287168 * pixels[:, 0] + \
                0.6740817658 * pixels[:, 1] + \
                0.0536895174 * pixels[:, 2]
    
    luminance_stats = {
        mean: CALCULATE_MEAN(luminance),
        std: CALCULATE_STD(luminance),
        histogram: CALCULATE_HISTOGRAM(luminance, bins=512)
    }
    
    // Temperatura kolorów
    color_temperature = ESTIMATE_COLOR_TEMPERATURE(mean_rgb)
    
    // Analiza gamut
    gamut_analysis = ANALYZE_GAMUT_COVERAGE(pixels)
    
    // Analiza kontrastu
    contrast_analysis = {
        global_contrast: CALCULATE_GLOBAL_CONTRAST(luminance),
        local_contrast: CALCULATE_LOCAL_CONTRAST(aces_image),
        dynamic_range: max_rgb - min_rgb
    }
    
    RETURN {
        basic_stats: {
            mean: mean_rgb,
            std: std_rgb,
            min: min_rgb,
            max: max_rgb,
            percentiles: percentiles
        },
        histograms: histograms,
        luminance: luminance_stats,
        color_temperature: color_temperature,
        gamut: gamut_analysis,
        contrast: contrast_analysis
    }
END FUNCTION
```

### 2.4 Obliczenie Transformacji ACES

```
FUNCTION calculate_aces_transform(source_stats, target_stats, method):
    SWITCH method:
        CASE "chromatic_adaptation":
            RETURN CALCULATE_CHROMATIC_ADAPTATION(
                source_stats.color_temperature,
                target_stats.color_temperature
            )
        
        CASE "statistical_matching":
            RETURN CALCULATE_STATISTICAL_TRANSFORM(
                source_stats.basic_stats,
                target_stats.basic_stats
            )
        
        CASE "histogram_matching":
            RETURN CALCULATE_HISTOGRAM_TRANSFORM(
                source_stats.histograms,
                target_stats.histograms
            )
        
        CASE "perceptual_matching":
            RETURN CALCULATE_PERCEPTUAL_TRANSFORM(
                source_stats.luminance,
                target_stats.luminance,
                source_stats.contrast,
                target_stats.contrast
            )
        
        CASE "hybrid":
            // Kombinacja metod z wagami
            chromatic = CALCULATE_CHROMATIC_ADAPTATION(...)
            statistical = CALCULATE_STATISTICAL_TRANSFORM(...)
            
            weight_chromatic = 0.6
            weight_statistical = 0.4
            
            RETURN COMBINE_TRANSFORMS(
                chromatic, statistical,
                weight_chromatic, weight_statistical
            )
        
        DEFAULT:
            THROW ERROR("Unknown transformation method: " + method)
    END SWITCH
END FUNCTION
```

---

## 3. Algorytmy Pomocnicze

### 3.1 Adaptacja Chromatyczna

```
FUNCTION calculate_chromatic_adaptation(source_temp, target_temp):
    // Bradford adaptation matrix
    bradford_matrix = [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296]
    ]
    
    bradford_inverse = MATRIX_INVERSE(bradford_matrix)
    
    // Konwersja temperatur na illuminanty XYZ
    source_illuminant = TEMPERATURE_TO_XYZ(source_temp)
    target_illuminant = TEMPERATURE_TO_XYZ(target_temp)
    
    // Transformacja Bradford
    source_bradford = MATRIX_MULTIPLY(bradford_matrix, source_illuminant)
    target_bradford = MATRIX_MULTIPLY(bradford_matrix, target_illuminant)
    
    // Macierz adaptacji
    adaptation_diagonal = DIAGONAL_MATRIX(
        target_bradford / source_bradford
    )
    
    // Finalna macierz transformacji
    adaptation_matrix = MATRIX_MULTIPLY(
        bradford_inverse,
        MATRIX_MULTIPLY(adaptation_diagonal, bradford_matrix)
    )
    
    RETURN {
        matrix: adaptation_matrix,
        source_temp: source_temp,
        target_temp: target_temp,
        method: "bradford"
    }
END FUNCTION
```

### 3.2 Dopasowanie Statystyczne

```
FUNCTION calculate_statistical_transform(source_stats, target_stats):
    // Korekcja średniej
    mean_shift = target_stats.mean - source_stats.mean
    
    // Korekcja odchylenia standardowego
    std_ratio = SAFE_DIVIDE(target_stats.std, source_stats.std)
    
    // Macierz skalowania
    scale_matrix = DIAGONAL_MATRIX(std_ratio)
    
    // Macierz translacji
    translation_vector = mean_shift
    
    // Kombinowana transformacja: T(x) = S * (x - μ_src) + μ_tgt
    // Gdzie S = scale_matrix
    
    RETURN {
        scale_matrix: scale_matrix,
        translation: translation_vector,
        source_mean: source_stats.mean,
        target_mean: target_stats.mean,
        method: "statistical"
    }
END FUNCTION
```

### 3.3 Dopasowanie Histogramów

```
FUNCTION calculate_histogram_transform(source_hists, target_hists):
    transform_luts = {}
    
    FOR channel IN [0, 1, 2]:  // R, G, B
        source_hist = source_hists[channel].histogram
        target_hist = target_hists[channel].histogram
        source_bins = source_hists[channel].bins
        target_bins = target_hists[channel].bins
        
        // Obliczenie CDF (Cumulative Distribution Function)
        source_cdf = CALCULATE_CDF(source_hist)
        target_cdf = CALCULATE_CDF(target_hist)
        
        // Tworzenie LUT (Look-Up Table)
        lut = CREATE_EMPTY_ARRAY(length=len(source_bins))
        
        FOR i IN range(len(source_bins)):
            source_value = source_cdf[i]
            
            // Znajdź najbliższą wartość w target_cdf
            target_index = FIND_CLOSEST_INDEX(target_cdf, source_value)
            lut[i] = target_bins[target_index]
        END FOR
        
        transform_luts[channel] = {
            lut: lut,
            source_bins: source_bins,
            target_bins: target_bins
        }
    END FOR
    
    RETURN {
        luts: transform_luts,
        method: "histogram_matching"
    }
END FUNCTION
```

---

## 4. Aplikacja Transformacji

### 4.1 Główna Funkcja Transformacji

```
FUNCTION apply_aces_transformation(source_aces, transform_data, parameters):
    result_aces = COPY(source_aces)
    original_shape = source_aces.shape
    
    // Reshape do 2D dla efektywnych operacji
    pixels = RESHAPE(result_aces, [-1, 3])
    
    // Aplikacja transformacji według metody
    SWITCH transform_data.method:
        CASE "chromatic_adaptation":
            pixels = APPLY_MATRIX_TRANSFORM(
                pixels, transform_data.matrix
            )
        
        CASE "statistical":
            // T(x) = S * (x - μ_src) + μ_tgt
            centered = pixels - transform_data.source_mean
            scaled = MATRIX_MULTIPLY(centered, transform_data.scale_matrix)
            pixels = scaled + transform_data.target_mean
        
        CASE "histogram_matching":
            FOR channel IN [0, 1, 2]:
                lut = transform_data.luts[channel].lut
                source_bins = transform_data.luts[channel].source_bins
                
                // Interpolacja LUT
                pixels[:, channel] = INTERPOLATE_LUT(
                    pixels[:, channel], source_bins, lut
                )
            END FOR
        
        CASE "hybrid":
            // Aplikacja wielu transformacji z wagami
            FOR transform IN transform_data.transforms:
                weight = transform.weight
                partial_result = APPLY_SINGLE_TRANSFORM(
                    pixels, transform
                )
                pixels = BLEND(pixels, partial_result, weight)
            END FOR
    END SWITCH
    
    // Ograniczenie do zakresu ACES
    pixels = CLAMP(pixels, 0.0, 65504.0)
    
    // Przywrócenie oryginalnego kształtu
    result_aces = RESHAPE(pixels, original_shape)
    
    RETURN result_aces
END FUNCTION
```

### 4.2 Tone Mapping ACES

```
FUNCTION apply_tone_mapping(aces_image, tone_curve_params):
    // ACES RRT (Reference Rendering Transform)
    a = tone_curve_params.get("a", 2.51)
    b = tone_curve_params.get("b", 0.03)
    c = tone_curve_params.get("c", 2.43)
    d = tone_curve_params.get("d", 0.59)
    e = tone_curve_params.get("e", 0.14)
    
    // Aplikacja krzywej tonalnej
    numerator = aces_image * (a * aces_image + b)
    denominator = aces_image * (c * aces_image + d) + e
    
    // Zabezpieczenie przed dzieleniem przez zero
    safe_denominator = MAX(denominator, 1e-10)
    
    tone_mapped = numerator / safe_denominator
    
    // Ograniczenie do [0, 1] po tone mappingu
    tone_mapped = CLAMP(tone_mapped, 0.0, 1.0)
    
    RETURN tone_mapped
END FUNCTION
```

---

## 5. Optymalizacje Algorytmiczne

### 5.1 Przetwarzanie w Blokach

```
FUNCTION process_in_chunks(large_image, transform_function, chunk_size=1024):
    height, width, channels = large_image.shape
    result = CREATE_ZEROS_LIKE(large_image)
    
    FOR y IN range(0, height, chunk_size):
        FOR x IN range(0, width, chunk_size):
            // Wyznaczenie granic bloku
            y_end = MIN(y + chunk_size, height)
            x_end = MIN(x + chunk_size, width)
            
            // Ekstrakcja bloku
            chunk = large_image[y:y_end, x:x_end, :]
            
            // Przetworzenie bloku
            processed_chunk = transform_function(chunk)
            
            // Zapisanie wyniku
            result[y:y_end, x:x_end, :] = processed_chunk
            
            // Opcjonalne: zwolnienie pamięci
            IF MEMORY_USAGE() > MEMORY_THRESHOLD:
                GARBAGE_COLLECT()
            END IF
        END FOR
    END FOR
    
    RETURN result
END FUNCTION
```

### 5.2 Równoległe Przetwarzanie

```
FUNCTION parallel_aces_transform(image, transform_data, num_threads=4):
    height, width, channels = image.shape
    
    // Podział obrazu na paski
    strip_height = height // num_threads
    strips = []
    
    FOR i IN range(num_threads):
        start_y = i * strip_height
        end_y = (i + 1) * strip_height IF i < num_threads - 1 ELSE height
        
        strip = image[start_y:end_y, :, :]
        strips.append(strip)
    END FOR
    
    // Przetwarzanie równoległe
    processed_strips = PARALLEL_MAP(
        lambda strip: apply_aces_transformation(strip, transform_data),
        strips
    )
    
    // Łączenie wyników
    result = CONCATENATE(processed_strips, axis=0)
    
    RETURN result
END FUNCTION
```

---

## 6. Podsumowanie Architektury

### Kluczowe Komponenty
1. **Walidacja wejścia**: Sprawdzenie formatów i profili kolorów
2. **Konwersja ACES**: Transformacja do przestrzeni roboczej
3. **Analiza statystyk**: Ekstraktowanie cech charakterystycznych
4. **Obliczenie transformacji**: Wybór i parametryzacja metody
5. **Aplikacja**: Zastosowanie transformacji do obrazu
6. **Post-processing**: Tone mapping, gamut compression
7. **Walidacja wyjścia**: Kontrola jakości wyniku

### Zalety Architektury
- **Modularność**: Każdy komponent może być rozwijany niezależnie
- **Skalowalność**: Wsparcie dla różnych rozmiarów obrazów
- **Elastyczność**: Możliwość dodawania nowych metod transformacji
- **Wydajność**: Optymalizacje pamięciowe i obliczeniowe

---

**Następna część:** [3of6 - Implementacja Core](gatto-WORKING-03-algorithms-08-advanced-01-aces-3of6.md)  
**Poprzednia część:** [1of6 - Teoria i Podstawy](gatto-WORKING-03-algorithms-08-advanced-01-aces-1of6.md)  
**Powrót do:** [Spis treści](gatto-WORKING-03-algorithms-08-advanced-01-aces-0of6.md)

*Część 2of6 - Pseudokod i Architektura | Wersja 1.0 | 2024-01-20*