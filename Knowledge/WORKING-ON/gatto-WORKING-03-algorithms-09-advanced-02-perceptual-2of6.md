# Perceptual Color Matching Algorithm - Pseudokod i Architektura Systemu

**Część 2 z 6: Pseudokod i Architektura Systemu**

---

## Spis Treści Części

1. [Architektura Ogólna](#architektura-ogólna)
2. [Diagramy Przepływu Danych](#diagramy-przepływu-danych)
3. [Pseudokod Głównego Algorytmu](#pseudokod-głównego-algorytmu)
4. [Pseudokod Funkcji Pomocniczych](#pseudokod-funkcji-pomocniczych)
5. [Strategie Przetwarzania](#strategie-przetwarzania)
6. [Optymalizacje Algorytmiczne](#optymalizacje-algorytmiczne)
7. [Architektura Modułowa](#architektura-modułowa)
8. [Wzorce Projektowe](#wzorce-projektowe)

---

## Architektura Ogólna

### Struktura Systemu

```
┌─────────────────────────────────────────────────────────────────┐
│                    PERCEPTUAL COLOR MATCHING                   │
│                         SYSTEM ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   INPUT LAYER   │    │ PROCESSING CORE │    │  OUTPUT LAYER   │
│                 │    │                 │    │                 │
│ • Source Image  │───▶│ • Color Space   │───▶│ • Matched Image │
│ • Target Image  │    │   Conversion    │    │ • Quality       │
│ • Parameters    │    │ • Perceptual    │    │   Metrics       │
│ • Viewing       │    │   Analysis      │    │ • Debug Info    │
│   Conditions    │    │ • Mapping Calc  │    │ • Statistics    │
└─────────────────┘    │ • Transformation│    └─────────────────┘
                       │ • Optimization  │
                       └─────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      SUPPORT MODULES                           │
│                                                                 │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │   Memory    │ │  Validation │ │ Performance │ │   Debug     │ │
│ │   Colors    │ │   & QA      │ │ Monitoring  │ │   Tools     │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Warstwy Abstrakcji

```python
# Pseudokod architektury warstwowej

CLASS PerceptualColorMatchingSystem:
    LAYERS = {
        'presentation': PresentationLayer,
        'application': ApplicationLayer,
        'domain': DomainLayer,
        'infrastructure': InfrastructureLayer
    }
    
    METHOD initialize():
        FOR each layer IN LAYERS:
            layer.initialize()
            layer.configure_dependencies()
    
    METHOD process_request(request):
        validated_request = presentation.validate(request)
        result = application.execute(validated_request)
        response = presentation.format_response(result)
        RETURN response
```

---

## Diagramy Przepływu Danych

### Główny Przepływ Danych

```
┌─────────────┐
│ Source RGB  │
└──────┬──────┘
       │
       ▼
┌─────────────┐    ┌─────────────┐
│ Validation  │───▶│ Color Space │
│ & Preprocessing │ │ Conversion  │
└─────────────┘    └──────┬──────┘
                          │
                          ▼
┌─────────────┐    ┌─────────────┐
│ Target RGB  │───▶│ Perceptual  │
└─────────────┘    │ Analysis    │
                   └──────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │ Mapping     │
                   │ Calculation │
                   └──────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │ Transform   │
                   │ Application │
                   └──────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │ Post-       │
                   │ Processing  │
                   └──────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │ RGB Output  │
                   └─────────────┘
```

### Przepływ Analizy Percepcyjnej

```
┌─────────────┐
│ LAB Image   │
└──────┬──────┘
       │
       ├─────────────────┬─────────────────┬─────────────────┐
       ▼                 ▼                 ▼                 ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ Statistical │   │ Memory      │   │ Spatial     │   │ Perceptual  │
│ Analysis    │   │ Colors      │   │ Analysis    │   │ Weights     │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │                 │
       └─────────────────┼─────────────────┼─────────────────┘
                         ▼
                  ┌─────────────┐
                  │ Integrated  │
                  │ Analysis    │
                  └─────────────┘
```

---

## Pseudokod Głównego Algorytmu

### Funkcja Główna

```python
FUNCTION perceptual_color_match(source_image, target_image, parameters):
    """
    Główna funkcja dopasowania percepcyjnego kolorów
    
    INPUT:
        source_image: RGB array [H, W, 3]
        target_image: RGB array [H, W, 3]
        parameters: PerceptualParameters object
    
    OUTPUT:
        result_image: RGB array [H, W, 3]
        quality_metrics: dict
        debug_info: dict
    """
    
    # === FAZA 1: WALIDACJA I PREPROCESSING ===
    CALL validate_inputs(source_image, target_image, parameters)
    
    source_preprocessed = CALL preprocess_image(source_image)
    target_preprocessed = CALL preprocess_image(target_image)
    
    # === FAZA 2: KONWERSJA DO PRZESTRZENI PERCEPCYJNEJ ===
    IF parameters.color_space == 'lab':
        source_perceptual = CALL rgb_to_lab(source_preprocessed, parameters.source_illuminant)
        target_perceptual = CALL rgb_to_lab(target_preprocessed, parameters.target_illuminant)
    ELIF parameters.color_space == 'cam16ucs':
        source_perceptual = CALL rgb_to_cam16ucs(source_preprocessed, parameters.viewing_conditions)
        target_perceptual = CALL rgb_to_cam16ucs(target_preprocessed, parameters.viewing_conditions)
    ELSE:
        THROW UnsupportedColorSpaceError
    
    # === FAZA 3: ANALIZA PERCEPCYJNA ===
    source_characteristics = CALL analyze_perceptual_characteristics(
        source_perceptual, 
        parameters.color_space,
        parameters.analysis_options
    )
    
    target_characteristics = CALL analyze_perceptual_characteristics(
        target_perceptual,
        parameters.color_space,
        parameters.analysis_options
    )
    
    # === FAZA 4: OBLICZENIE MAPOWANIA ===
    mapping = CALL calculate_perceptual_mapping(
        source_characteristics,
        target_characteristics,
        parameters.mapping_method,
        parameters.mapping_options
    )
    
    # === FAZA 5: OBLICZENIE WAG PERCEPCYJNYCH ===
    IF parameters.use_perceptual_weights:
        perceptual_weights = CALL calculate_perceptual_weights(
            source_perceptual,
            source_characteristics,
            parameters.weight_options
        )
    ELSE:
        perceptual_weights = NULL
    
    # === FAZA 6: APLIKACJA TRANSFORMACJI ===
    result_perceptual = CALL apply_perceptual_mapping(
        source_perceptual,
        mapping,
        perceptual_weights,
        parameters.application_options
    )
    
    # === FAZA 7: ADAPTACJA CHROMATYCZNA ===
    IF parameters.chromatic_adaptation AND 
       parameters.source_illuminant != parameters.target_illuminant:
        result_perceptual = CALL apply_chromatic_adaptation(
            result_perceptual,
            parameters.source_illuminant,
            parameters.target_illuminant,
            parameters.adaptation_method
        )
    
    # === FAZA 8: POST-PROCESSING ===
    result_perceptual = CALL apply_post_processing(
        result_perceptual,
        source_perceptual,
        target_perceptual,
        parameters.post_processing_options
    )
    
    # === FAZA 9: KONWERSJA Z POWROTEM DO RGB ===
    IF parameters.color_space == 'lab':
        result_image = CALL lab_to_rgb(result_perceptual, parameters.target_illuminant)
    ELIF parameters.color_space == 'cam16ucs':
        result_image = CALL cam16ucs_to_rgb(result_perceptual, parameters.viewing_conditions)
    
    # === FAZA 10: OCENA JAKOŚCI ===
    quality_metrics = CALL evaluate_perceptual_quality(
        source_image,
        result_image,
        target_image,
        parameters.quality_metrics
    )
    
    # === FAZA 11: GENEROWANIE DEBUG INFO ===
    debug_info = CALL generate_debug_info(
        source_characteristics,
        target_characteristics,
        mapping,
        perceptual_weights,
        quality_metrics
    )
    
    RETURN result_image, quality_metrics, debug_info

END FUNCTION
```

### Algorytm Iteracyjny

```python
FUNCTION iterative_perceptual_matching(source_image, target_image, parameters):
    """
    Iteracyjne dopasowanie z optymalizacją jakości
    """
    
    current_result = source_image
    best_result = source_image
    best_quality = INFINITY
    
    FOR iteration = 1 TO parameters.max_iterations:
        # Dopasowanie percepcyjne
        current_result, quality, debug = CALL perceptual_color_match(
            current_result, target_image, parameters
        )
        
        # Ocena jakości
        current_quality = quality['mean_delta_e']
        
        # Sprawdzenie poprawy
        IF current_quality < best_quality:
            best_result = current_result
            best_quality = current_quality
            
            # Adaptacja parametrów dla następnej iteracji
            parameters = CALL adapt_parameters(parameters, quality, debug)
        ELSE:
            # Brak poprawy - zakończenie
            BREAK
        
        # Sprawdzenie kryterium zatrzymania
        IF current_quality < parameters.target_quality:
            BREAK
    
    RETURN best_result, best_quality

END FUNCTION
```

---

## Pseudokod Funkcji Pomocniczych

### Analiza Charakterystyk Percepcyjnych

```python
FUNCTION analyze_perceptual_characteristics(perceptual_image, color_space, options):
    """
    Analiza charakterystyk percepcyjnych obrazu
    """
    
    characteristics = EMPTY_DICT
    
    # === ANALIZA STATYSTYCZNA ===
    characteristics['statistics'] = CALL calculate_perceptual_statistics(
        perceptual_image, color_space
    )
    
    # === ANALIZA MEMORY COLORS ===
    IF options.analyze_memory_colors:
        characteristics['memory_colors'] = CALL analyze_memory_colors(
            perceptual_image, color_space
        )
    
    # === ANALIZA PRZESTRZENNA ===
    IF options.analyze_spatial:
        characteristics['spatial'] = CALL analyze_spatial_distribution(
            perceptual_image
        )
    
    # === ANALIZA DOMINUJĄCYCH KOLORÓW ===
    IF options.analyze_dominant_colors:
        characteristics['dominant_colors'] = CALL find_dominant_colors(
            perceptual_image, options.num_dominant_colors
        )
    
    # === ANALIZA HARMONII KOLORYSTYCZNEJ ===
    IF options.analyze_harmony:
        characteristics['harmony'] = CALL analyze_color_harmony(
            perceptual_image
        )
    
    RETURN characteristics

END FUNCTION

FUNCTION calculate_perceptual_statistics(perceptual_image, color_space):
    """
    Obliczenie statystyk percepcyjnych
    """
    
    stats = EMPTY_DICT
    
    IF color_space == 'lab':
        L, a, b = SPLIT_CHANNELS(perceptual_image)
        
        # Statystyki lightness
        stats['lightness'] = {
            'mean': MEAN(L),
            'std': STD(L),
            'min': MIN(L),
            'max': MAX(L),
            'percentiles': PERCENTILES(L, [5, 25, 50, 75, 95])
        }
        
        # Statystyki chroma
        chroma = SQRT(a^2 + b^2)
        stats['chroma'] = {
            'mean': MEAN(chroma),
            'std': STD(chroma),
            'max': MAX(chroma),
            'percentiles': PERCENTILES(chroma, [5, 25, 50, 75, 95])
        }
        
        # Statystyki hue
        hue = ATAN2(b, a) * 180 / PI
        hue = NORMALIZE_HUE(hue)  # [0, 360)
        
        stats['hue'] = {
            'circular_mean': CIRCULAR_MEAN(hue),
            'circular_std': CIRCULAR_STD(hue),
            'distribution': HISTOGRAM(hue, bins=36)  # 10° bins
        }
        
    ELIF color_space == 'cam16ucs':
        J, a_ucs, b_ucs = SPLIT_CHANNELS(perceptual_image)
        
        # Analogiczne statystyki dla CAM16-UCS
        stats['lightness'] = CALCULATE_CHANNEL_STATS(J)
        stats['chroma'] = CALCULATE_CHANNEL_STATS(SQRT(a_ucs^2 + b_ucs^2))
        stats['hue'] = CALCULATE_HUE_STATS(ATAN2(b_ucs, a_ucs))
    
    RETURN stats

END FUNCTION
```

### Obliczanie Mapowania Percepcyjnego

```python
FUNCTION calculate_perceptual_mapping(source_chars, target_chars, method, options):
    """
    Obliczenie mapowania percepcyjnego między charakterystykami
    """
    
    mapping = EMPTY_DICT
    
    IF method == 'statistical':
        mapping = CALL calculate_statistical_mapping(source_chars, target_chars)
        
    ELIF method == 'memory_color_preservation':
        mapping = CALL calculate_memory_color_mapping(source_chars, target_chars)
        
    ELIF method == 'histogram_matching':
        mapping = CALL calculate_histogram_mapping(source_chars, target_chars)
        
    ELIF method == 'dominant_color_matching':
        mapping = CALL calculate_dominant_color_mapping(source_chars, target_chars)
        
    ELIF method == 'hybrid':
        # Kombinacja różnych metod
        stat_mapping = CALL calculate_statistical_mapping(source_chars, target_chars)
        memory_mapping = CALL calculate_memory_color_mapping(source_chars, target_chars)
        
        # Ważona kombinacja
        mapping = CALL combine_mappings(
            [stat_mapping, memory_mapping],
            weights=[options.statistical_weight, options.memory_weight]
        )
    
    # Walidacja i ograniczenia mapowania
    mapping = CALL validate_mapping(mapping, options.constraints)
    
    RETURN mapping

END FUNCTION

FUNCTION calculate_statistical_mapping(source_chars, target_chars):
    """
    Mapowanie oparte na statystykach
    """
    
    mapping = EMPTY_DICT
    
    # Mapowanie lightness
    source_L_stats = source_chars['statistics']['lightness']
    target_L_stats = target_chars['statistics']['lightness']
    
    mapping['lightness_scale'] = target_L_stats['std'] / source_L_stats['std']
    mapping['lightness_shift'] = target_L_stats['mean'] - 
                                 source_L_stats['mean'] * mapping['lightness_scale']
    
    # Mapowanie chroma
    source_C_stats = source_chars['statistics']['chroma']
    target_C_stats = target_chars['statistics']['chroma']
    
    mapping['chroma_scale'] = target_C_stats['std'] / source_C_stats['std']
    mapping['chroma_shift'] = target_C_stats['mean'] - 
                              source_C_stats['mean'] * mapping['chroma_scale']
    
    # Mapowanie hue (przesunięcie kołowe)
    source_hue_mean = source_chars['statistics']['hue']['circular_mean']
    target_hue_mean = target_chars['statistics']['hue']['circular_mean']
    
    hue_shift = target_hue_mean - source_hue_mean
    
    # Normalizacja do [-180, 180]
    IF hue_shift > 180:
        hue_shift = hue_shift - 360
    ELIF hue_shift < -180:
        hue_shift = hue_shift + 360
    
    mapping['hue_shift'] = hue_shift
    
    RETURN mapping

END FUNCTION
```

### Aplikacja Mapowania

```python
FUNCTION apply_perceptual_mapping(source_perceptual, mapping, weights, options):
    """
    Aplikacja mapowania percepcyjnego z wagami
    """
    
    result_perceptual = COPY(source_perceptual)
    height, width, channels = SHAPE(source_perceptual)
    
    # Rozdzielenie kanałów
    L, a, b = SPLIT_CHANNELS(source_perceptual)
    
    # Obliczenie chroma i hue
    chroma = SQRT(a^2 + b^2)
    hue = ATAN2(b, a)
    
    # === APLIKACJA MAPOWANIA LIGHTNESS ===
    new_L = L * mapping['lightness_scale'] + mapping['lightness_shift']
    new_L = CLIP(new_L, 0, 100)
    
    # === APLIKACJA MAPOWANIA CHROMA ===
    new_chroma = chroma * mapping['chroma_scale'] + mapping['chroma_shift']
    new_chroma = MAX(new_chroma, 0)
    
    # === APLIKACJA MAPOWANIA HUE ===
    new_hue = hue + RADIANS(mapping['hue_shift'])
    
    # Konwersja z powrotem do a*, b*
    new_a = new_chroma * COS(new_hue)
    new_b = new_chroma * SIN(new_hue)
    
    # === APLIKACJA WAG PERCEPCYJNYCH ===
    IF weights IS NOT NULL:
        FOR i = 0 TO height-1:
            FOR j = 0 TO width-1:
                pixel_weights = CALL calculate_pixel_weights(
                    source_perceptual[i, j], weights[i, j]
                )
                
                # Mieszanie oryginalnego i zmapowanego koloru
                blend_factor = pixel_weights['overall']
                blend_factor = CLIP(blend_factor, 0, 1)
                
                result_perceptual[i, j, 0] = L[i, j] * (1 - blend_factor) + 
                                             new_L[i, j] * blend_factor
                result_perceptual[i, j, 1] = a[i, j] * (1 - blend_factor) + 
                                             new_a[i, j] * blend_factor
                result_perceptual[i, j, 2] = b[i, j] * (1 - blend_factor) + 
                                             new_b[i, j] * blend_factor
    ELSE:
        # Bez wag - pełne mapowanie
        result_perceptual[:, :, 0] = new_L
        result_perceptual[:, :, 1] = new_a
        result_perceptual[:, :, 2] = new_b
    
    # === POST-PROCESSING ===
    IF options.local_contrast_correction:
        result_perceptual = CALL apply_local_contrast_correction(
            result_perceptual, source_perceptual
        )
    
    IF options.gamut_mapping:
        result_perceptual = CALL apply_gamut_mapping(
            result_perceptual, options.target_gamut
        )
    
    RETURN result_perceptual

END FUNCTION
```

---

## Strategie Przetwarzania

### Przetwarzanie Sekwencyjne

```python
FUNCTION sequential_processing(image_list, target, parameters):
    """
    Sekwencyjne przetwarzanie listy obrazów
    """
    
    results = EMPTY_LIST
    
    FOR each image IN image_list:
        result, quality, debug = CALL perceptual_color_match(
            image, target, parameters
        )
        
        results.APPEND({
            'image': result,
            'quality': quality,
            'debug': debug
        })
    
    RETURN results

END FUNCTION
```

### Przetwarzanie Równoległe

```python
FUNCTION parallel_processing(image_list, target, parameters, num_workers):
    """
    Równoległe przetwarzanie z podziałem zadań
    """
    
    # Podział zadań
    chunks = SPLIT_LIST(image_list, num_workers)
    
    # Uruchomienie workerów
    workers = EMPTY_LIST
    FOR each chunk IN chunks:
        worker = START_WORKER(
            FUNCTION=sequential_processing,
            ARGS=(chunk, target, parameters)
        )
        workers.APPEND(worker)
    
    # Zbieranie wyników
    all_results = EMPTY_LIST
    FOR each worker IN workers:
        chunk_results = WAIT_FOR_WORKER(worker)
        all_results.EXTEND(chunk_results)
    
    RETURN all_results

END FUNCTION
```

### Przetwarzanie Kafelkowe

```python
FUNCTION tile_based_processing(large_image, target, parameters, tile_size, overlap):
    """
    Przetwarzanie dużych obrazów metodą kafelkową
    """
    
    height, width = SHAPE(large_image)[:2]
    result = ZEROS_LIKE(large_image)
    
    # Obliczenie siatki kafelków
    tiles = CALCULATE_TILE_GRID(height, width, tile_size, overlap)
    
    FOR each tile IN tiles:
        y_start, y_end, x_start, x_end = tile.coordinates
        
        # Wycinanie kafelka z nakładaniem
        source_tile = large_image[y_start:y_end, x_start:x_end]
        
        # Dopasowanie rozmiaru target do tile
        target_tile = RESIZE(target, (y_end-y_start, x_end-x_start))
        
        # Przetwarzanie kafelka
        processed_tile, _, _ = CALL perceptual_color_match(
            source_tile, target_tile, parameters
        )
        
        # Mieszanie z nakładaniem
        IF tile.has_overlap:
            result[y_start:y_end, x_start:x_end] = CALL blend_with_overlap(
                result[y_start:y_end, x_start:x_end],
                processed_tile,
                overlap
            )
        ELSE:
            result[y_start:y_end, x_start:x_end] = processed_tile
    
    RETURN result

END FUNCTION

FUNCTION blend_with_overlap(existing, new_tile, overlap):
    """
    Mieszanie kafelków z nakładaniem
    """
    
    height, width = SHAPE(new_tile)[:2]
    
    # Tworzenie maski mieszania
    blend_mask = ONES((height, width))
    
    # Gradient w obszarze nakładania
    FOR i = 0 TO overlap-1:
        weight = i / overlap
        
        # Górna krawędź
        blend_mask[i, :] = weight
        
        # Dolna krawędź
        blend_mask[height-1-i, :] = weight
        
        # Lewa krawędź
        blend_mask[:, i] = MIN(blend_mask[:, i], weight)
        
        # Prawa krawędź
        blend_mask[:, width-1-i] = MIN(blend_mask[:, width-1-i], weight)
    
    # Mieszanie
    blended = existing * (1 - blend_mask[:, :, NEWAXIS]) + 
              new_tile * blend_mask[:, :, NEWAXIS]
    
    RETURN blended

END FUNCTION
```

---

## Optymalizacje Algorytmiczne

### Optymalizacja Pamięci

```python
FUNCTION memory_optimized_processing(large_image, target, parameters):
    """
    Przetwarzanie z optymalizacją pamięci
    """
    
    # Obliczenie optymalnego rozmiaru chunka
    available_memory = GET_AVAILABLE_MEMORY()
    optimal_chunk_size = CALCULATE_OPTIMAL_CHUNK_SIZE(
        large_image, available_memory
    )
    
    # Przetwarzanie w chunkach
    original_shape = SHAPE(large_image)
    flat_image = RESHAPE(large_image, (-1, 3))
    flat_result = ZEROS_LIKE(flat_image)
    
    FOR start = 0 TO LENGTH(flat_image) STEP optimal_chunk_size:
        end = MIN(start + optimal_chunk_size, LENGTH(flat_image))
        
        # Przetwarzanie chunka
        chunk = flat_image[start:end]
        chunk_2d = RESHAPE(chunk, (-1, 1, 3))  # Symulacja obrazu 2D
        
        processed_chunk, _, _ = CALL perceptual_color_match(
            chunk_2d, target, parameters
        )
        
        flat_result[start:end] = RESHAPE(processed_chunk, (-1, 3))
        
        # Zwolnienie pamięci
        DELETE chunk, chunk_2d, processed_chunk
        GARBAGE_COLLECT()
    
    result = RESHAPE(flat_result, original_shape)
    RETURN result

END FUNCTION
```

### Optymalizacja Obliczeniowa

```python
FUNCTION computational_optimizations(source, target, parameters):
    """
    Różne optymalizacje obliczeniowe
    """
    
    # === EARLY TERMINATION ===
    IF IMAGES_IDENTICAL(source, target, threshold=0.01):
        RETURN source, PERFECT_QUALITY, EMPTY_DEBUG
    
    # === DOWNSAMPLING DLA ANALIZY ===
    IF parameters.use_downsampling_for_analysis:
        analysis_scale = 0.25
        source_small = RESIZE(source, scale=analysis_scale)
        target_small = RESIZE(target, scale=analysis_scale)
        
        # Analiza na małych obrazach
        source_chars = CALL analyze_perceptual_characteristics(
            source_small, parameters.color_space, parameters.analysis_options
        )
        target_chars = CALL analyze_perceptual_characteristics(
            target_small, parameters.color_space, parameters.analysis_options
        )
        
        # Mapowanie na pełnych obrazach
        mapping = CALL calculate_perceptual_mapping(
            source_chars, target_chars, parameters.mapping_method
        )
    ELSE:
        # Standardowa analiza
        source_chars = CALL analyze_perceptual_characteristics(source, ...)
        target_chars = CALL analyze_perceptual_characteristics(target, ...)
        mapping = CALL calculate_perceptual_mapping(source_chars, target_chars, ...)
    
    # === LOOKUP TABLES ===
    IF parameters.use_lookup_tables:
        lut = CALL create_color_lookup_table(mapping, parameters.lut_size)
        result = CALL apply_lookup_table(source, lut)
    ELSE:
        result = CALL apply_perceptual_mapping(source, mapping, ...)
    
    RETURN result

END FUNCTION

FUNCTION create_color_lookup_table(mapping, lut_size):
    """
    Tworzenie tabeli lookup dla szybkiego mapowania
    """
    
    lut = ZEROS((lut_size, lut_size, lut_size, 3))
    
    FOR L = 0 TO lut_size-1:
        FOR a = 0 TO lut_size-1:
            FOR b = 0 TO lut_size-1:
                # Normalizacja do rzeczywistych wartości LAB
                L_real = L * 100.0 / (lut_size - 1)
                a_real = (a - lut_size/2) * 256.0 / lut_size
                b_real = (b - lut_size/2) * 256.0 / lut_size
                
                # Aplikacja mapowania
                mapped_L = L_real * mapping['lightness_scale'] + mapping['lightness_shift']
                
                chroma = SQRT(a_real^2 + b_real^2)
                hue = ATAN2(b_real, a_real)
                
                mapped_chroma = chroma * mapping['chroma_scale'] + mapping['chroma_shift']
                mapped_hue = hue + RADIANS(mapping['hue_shift'])
                
                mapped_a = mapped_chroma * COS(mapped_hue)
                mapped_b = mapped_chroma * SIN(mapped_hue)
                
                lut[L, a, b] = [mapped_L, mapped_a, mapped_b]
    
    RETURN lut

END FUNCTION
```

---

## Architektura Modułowa

### Struktura Modułów

```python
# Pseudokod architektury modułowej

MODULE ColorSpaceModule:
    EXPORTS:
        - rgb_to_lab()
        - lab_to_rgb()
        - rgb_to_cam16ucs()
        - cam16ucs_to_rgb()
    
    DEPENDENCIES:
        - numpy
        - colour-science

MODULE PerceptualAnalysisModule:
    EXPORTS:
        - analyze_perceptual_characteristics()
        - calculate_perceptual_statistics()
        - analyze_memory_colors()
        - calculate_perceptual_weights()
    
    DEPENDENCIES:
        - ColorSpaceModule
        - MemoryColorsModule

MODULE MappingModule:
    EXPORTS:
        - calculate_perceptual_mapping()
        - apply_perceptual_mapping()
        - validate_mapping()
    
    DEPENDENCIES:
        - PerceptualAnalysisModule

MODULE OptimizationModule:
    EXPORTS:
        - memory_optimized_processing()
        - parallel_processing()
        - tile_based_processing()
    
    DEPENDENCIES:
        - MappingModule
        - multiprocessing

MODULE QualityModule:
    EXPORTS:
        - evaluate_perceptual_quality()
        - calculate_delta_e_2000()
        - calculate_delta_e_cam16()
    
    DEPENDENCIES:
        - ColorSpaceModule

MODULE MainModule:
    EXPORTS:
        - PerceptualColorMatcher class
        - perceptual_color_match()
    
    DEPENDENCIES:
        - ALL above modules
```

### Interfejsy Modułów

```python
INTERFACE IColorSpaceConverter:
    METHOD convert_to_perceptual(rgb_image, color_space, illuminant)
    METHOD convert_from_perceptual(perceptual_image, color_space, illuminant)
    METHOD get_supported_spaces()

INTERFACE IPerceptualAnalyzer:
    METHOD analyze_characteristics(perceptual_image, options)
    METHOD calculate_statistics(perceptual_image)
    METHOD identify_memory_colors(perceptual_image)

INTERFACE IMappingCalculator:
    METHOD calculate_mapping(source_chars, target_chars, method)
    METHOD apply_mapping(image, mapping, weights)
    METHOD validate_mapping(mapping, constraints)

INTERFACE IQualityEvaluator:
    METHOD evaluate_quality(original, result, target, metrics)
    METHOD calculate_delta_e(image1, image2, method)
    METHOD generate_quality_report(quality_data)
```

---

## Wzorce Projektowe

### Strategy Pattern dla Metod Mapowania

```python
CLASS MappingStrategy:
    ABSTRACT METHOD calculate(source_chars, target_chars, options)

CLASS StatisticalMappingStrategy EXTENDS MappingStrategy:
    METHOD calculate(source_chars, target_chars, options):
        RETURN CALL calculate_statistical_mapping(source_chars, target_chars)

CLASS MemoryColorMappingStrategy EXTENDS MappingStrategy:
    METHOD calculate(source_chars, target_chars, options):
        RETURN CALL calculate_memory_color_mapping(source_chars, target_chars)

CLASS MappingContext:
    FIELD strategy: MappingStrategy
    
    METHOD set_strategy(strategy: MappingStrategy):
        self.strategy = strategy
    
    METHOD execute_mapping(source_chars, target_chars, options):
        RETURN self.strategy.calculate(source_chars, target_chars, options)
```

### Factory Pattern dla Przestrzeni Kolorów

```python
CLASS ColorSpaceFactory:
    STATIC METHOD create_converter(color_space_name):
        IF color_space_name == 'lab':
            RETURN LABConverter()
        ELIF color_space_name == 'cam16ucs':
            RETURN CAM16UCSConverter()
        ELIF color_space_name == 'luv':
            RETURN LUVConverter()
        ELSE:
            THROW UnsupportedColorSpaceError(color_space_name)
```

### Observer Pattern dla Monitorowania Postępu

```python
CLASS ProgressObserver:
    ABSTRACT METHOD on_progress_update(stage, progress, message)

CLASS PerceptualColorMatcher:
    FIELD observers: LIST of ProgressObserver
    
    METHOD add_observer(observer: ProgressObserver):
        self.observers.APPEND(observer)
    
    METHOD notify_progress(stage, progress, message):
        FOR each observer IN self.observers:
            observer.on_progress_update(stage, progress, message)
    
    METHOD process_with_progress(source, target, parameters):
        self.notify_progress('validation', 0.1, 'Validating inputs')
        # ... walidacja ...
        
        self.notify_progress('conversion', 0.2, 'Converting color spaces')
        # ... konwersja ...
        
        self.notify_progress('analysis', 0.4, 'Analyzing characteristics')
        # ... analiza ...
        
        self.notify_progress('mapping', 0.6, 'Calculating mapping')
        # ... mapowanie ...
        
        self.notify_progress('application', 0.8, 'Applying transformation')
        # ... aplikacja ...
        
        self.notify_progress('completion', 1.0, 'Processing complete')
```

---

## Nawigacja

**◀️ Poprzednia część**: [Teoria i Podstawy Percepcyjne](gatto-WORKING-03-algorithms-09-advanced-02-perceptual-1of6.md)  
**▶️ Następna część**: [Implementacja Podstawowa](gatto-WORKING-03-algorithms-09-advanced-02-perceptual-3of6.md)  
**🏠 Powrót do**: [Spis Treści](gatto-WORKING-03-algorithms-09-advanced-02-perceptual-0of6.md#spis-treści---kompletna-dokumentacja)

---

*Ostatnia aktualizacja: 2024-01-20*  
*Autor: GattoNero AI Assistant*  
*Wersja: 1.0*  
*Status: Dokumentacja kompletna* ✅