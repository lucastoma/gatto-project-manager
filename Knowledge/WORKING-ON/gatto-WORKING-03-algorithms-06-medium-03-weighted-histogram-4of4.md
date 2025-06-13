# Weighted Histogram Matching - Część 4: Praktyczne Zastosowania i Integracja

## 🟡 Poziom: Medium
**Trudność**: Średnia | **Czas implementacji**: 4-6 godzin | **Złożoność**: O(n log n)

---

## Praktyczne Przykłady Zastosowań

### 1. Korekcja Ekspozycji Portretów

```python
def correct_portrait_exposure(source_path, target_path, output_path):
    """Koryguje ekspozycję portretu na podstawie dobrze naświetlonego wzorca"""
    
    # Wczytaj obrazy
    source = cv2.imread(source_path)
    target = cv2.imread(target_path)
    source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    
    # Konfiguracja dla portretów
    portrait_config = {
        'weight_type': 'segmented',
        'shadow_weight': 0.9,    # Mocna korekcja cieni (twarz)
        'midtone_weight': 1.0,   # Pełna korekcja średnich tonów
        'highlight_weight': 0.7  # Delikatna korekcja świateł
    }
    
    # Utwórz matcher
    matcher = WeightedHistogramMatching()
    
    # Przetwórz obraz
    result = matcher.process_rgb_image(
        source_rgb, target_rgb,
        weight_config=portrait_config
    )
    
    # Zapisz wynik
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    
    # Pokaż porównanie
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(source_rgb)
    axes[0].set_title('Oryginał (niedoświetlony)')
    axes[0].axis('off')
    
    axes[1].imshow(target_rgb)
    axes[1].set_title('Wzorzec (dobrze naświetlony)')
    axes[1].axis('off')
    
    axes[2].imshow(result)
    axes[2].set_title('Wynik korekcji')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return result

# Przykład użycia
result = correct_portrait_exposure(
    'underexposed_portrait.jpg',
    'well_exposed_portrait.jpg',
    'corrected_portrait.jpg'
)
```

### 2. Stylizacja Krajobrazów

```python
def stylize_landscape(source_path, style_path, output_path, style_strength=0.8):
    """Stylizuje krajobraz na podstawie obrazu wzorcowego"""
    
    # Wczytaj obrazy
    source = cv2.imread(source_path)
    style = cv2.imread(style_path)
    source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    style_rgb = cv2.cvtColor(style, cv2.COLOR_BGR2RGB)
    
    # Konfiguracja adaptacyjna dla krajobrazów
    matcher = AdaptiveWeightedHistogramMatching()
    
    # Użyj adaptacji opartej na zawartości
    result, weight_map = matcher.adaptive_weight_matching(
        source_rgb, style_rgb,
        adaptation_method='content_aware',
        global_strength=style_strength
    )
    
    # Zapisz wynik
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    
    # Wizualizuj mapę wag
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(source_rgb)
    axes[0, 0].set_title('Oryginał')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(style_rgb)
    axes[0, 1].set_title('Styl wzorcowy')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(result)
    axes[1, 0].set_title('Wynik stylizacji')
    axes[1, 0].axis('off')
    
    # Pokaż mapę wag (średnia z kanałów)
    weight_avg = np.mean(weight_map, axis=2)
    im = axes[1, 1].imshow(weight_avg, cmap='viridis')
    axes[1, 1].set_title('Mapa intensywności stylizacji')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()
    
    return result, weight_map

# Przykład użycia
result, weights = stylize_landscape(
    'landscape_original.jpg',
    'landscape_golden_hour.jpg',
    'landscape_stylized.jpg',
    style_strength=0.7
)
```

### 3. Selektywna Korekcja Kolorów

```python
def selective_color_correction(image_path, output_path, corrections):
    """Selektywna korekcja kolorów w określonych regionach
    
    Args:
        image_path: Ścieżka do obrazu
        output_path: Ścieżka zapisu
        corrections: Lista słowników z konfiguracją korekcji
                    [{'roi': (x, y, w, h), 'target_color': (r, g, b), 'strength': 0.8}]
    """
    
    # Wczytaj obraz
    source = cv2.imread(image_path)
    source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    
    # Utwórz matcher z maskami
    matcher = MaskedWeightedHistogramMatching()
    
    result = source_rgb.copy()
    
    for correction in corrections:
        roi = correction['roi']
        target_color = correction['target_color']
        strength = correction.get('strength', 0.8)
        
        # Utwórz obraz docelowy z pożądanym kolorem
        target_image = np.full_like(source_rgb, target_color, dtype=np.uint8)
        
        # Konfiguracja wag dla selektywnej korekcji
        weight_config = {
            'weight_type': 'gaussian',
            'center': 128,
            'sigma': 64,
            'amplitude': strength
        }
        
        # Zastosuj korekcję tylko w ROI
        corrected_roi = matcher.masked_weighted_matching(
            result, target_image,
            roi=roi,
            weight_config=weight_config
        )
        
        # Blend z oryginalnym obrazem
        x, y, w, h = roi
        alpha = strength
        result[y:y+h, x:x+w] = (
            alpha * corrected_roi[y:y+h, x:x+w] + 
            (1 - alpha) * result[y:y+h, x:x+w]
        ).astype(np.uint8)
    
    # Zapisz wynik
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    
    # Wizualizuj
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(source_rgb)
    axes[0].set_title('Oryginał')
    axes[0].axis('off')
    
    # Zaznacz ROI na oryginalnym obrazie
    for correction in corrections:
        x, y, w, h = correction['roi']
        rect = plt.Rectangle((x, y), w, h, linewidth=2, 
                           edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
    
    axes[1].imshow(result)
    axes[1].set_title('Po korekcji')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return result

# Przykład użycia - korekcja nieba i trawy
corrections = [
    {
        'roi': (0, 0, 800, 300),  # Niebo
        'target_color': (135, 206, 235),  # Niebieski
        'strength': 0.6
    },
    {
        'roi': (0, 300, 800, 200),  # Trawa
        'target_color': (34, 139, 34),  # Zielony
        'strength': 0.7
    }
]

result = selective_color_correction(
    'landscape.jpg',
    'landscape_corrected.jpg',
    corrections
)
```

### 4. Batch Processing

```python
class BatchWeightedHistogramProcessor:
    """Klasa do przetwarzania wsadowego obrazów"""
    
    def __init__(self, config_path=None):
        self.matcher = OptimizedWeightedHistogramMatching(use_numba=True)
        self.config = WeightedHistogramConfig()
        
        if config_path:
            self.config.load_from_file(config_path)
    
    def process_directory(self, input_dir, output_dir, target_image_path, 
                         file_pattern="*.jpg", preset='balanced'):
        """Przetwarza wszystkie obrazy w katalogu"""
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Wczytaj obraz docelowy
        target_image = cv2.imread(target_image_path)
        target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        
        # Pobierz konfigurację presetu
        preset_config = self.config.get_preset(preset)
        
        # Znajdź wszystkie pliki
        image_files = list(input_path.glob(file_pattern))
        
        print(f"Przetwarzanie {len(image_files)} obrazów...")
        
        results = []
        
        for i, image_file in enumerate(image_files):
            try:
                print(f"Przetwarzanie {i+1}/{len(image_files)}: {image_file.name}")
                
                # Wczytaj obraz źródłowy
                source_image = cv2.imread(str(image_file))
                source_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
                
                # Przetwórz obraz
                start_time = time.time()
                
                if source_rgb.shape[0] * source_rgb.shape[1] > 1024 * 1024:
                    # Użyj tiled processing dla dużych obrazów
                    result = self.matcher.process_large_image_tiled(
                        source_rgb, target_rgb,
                        tile_size=512, overlap=64,
                        weight_config=preset_config.weight_config.__dict__
                    )
                else:
                    result = self.matcher.process_rgb_image(
                        source_rgb, target_rgb,
                        weight_config=preset_config.weight_config.__dict__
                    )
                
                processing_time = time.time() - start_time
                
                # Zapisz wynik
                output_file = output_path / f"processed_{image_file.name}"
                result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_file), result_bgr)
                
                results.append({
                    'file': image_file.name,
                    'processing_time': processing_time,
                    'input_size': source_rgb.shape,
                    'output_file': output_file.name,
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"Błąd przetwarzania {image_file.name}: {e}")
                results.append({
                    'file': image_file.name,
                    'processing_time': 0,
                    'input_size': None,
                    'output_file': None,
                    'status': f'error: {e}'
                })
        
        # Zapisz raport
        self._save_processing_report(results, output_path / 'processing_report.json')
        
        return results
    
    def _save_processing_report(self, results, report_path):
        """Zapisuje raport przetwarzania"""
        import json
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_files': len(results),
            'successful': len([r for r in results if r['status'] == 'success']),
            'failed': len([r for r in results if r['status'] != 'success']),
            'total_time': sum(r['processing_time'] for r in results),
            'average_time': np.mean([r['processing_time'] for r in results if r['processing_time'] > 0]),
            'results': results
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nRaport zapisany: {report_path}")
        print(f"Przetworzone pomyślnie: {report['successful']}/{report['total_files']}")
        print(f"Całkowity czas: {report['total_time']:.1f}s")
        print(f"Średni czas na obraz: {report['average_time']:.2f}s")

# Przykład użycia batch processing
processor = BatchWeightedHistogramProcessor()

results = processor.process_directory(
    input_dir='input_photos',
    output_dir='output_photos',
    target_image_path='reference_style.jpg',
    file_pattern='*.jpg',
    preset='portrait'
)
```

---

## Integracja z Głównym Systemem Flask

### API Endpoints

```python
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import tempfile
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Globalne instancje matcherów
matchers = {
    'basic': WeightedHistogramMatching(),
    'optimized': OptimizedWeightedHistogramMatching(use_numba=True),
    'adaptive': AdaptiveWeightedHistogramMatching(),
    'local': LocalWeightedHistogramMatching(),
    'masked': MaskedWeightedHistogramMatching()
}

config_manager = WeightedHistogramConfig()

@app.route('/api/weighted-histogram/process', methods=['POST'])
def process_weighted_histogram():
    """Główny endpoint do przetwarzania obrazów"""
    try:
        # Sprawdź czy pliki zostały przesłane
        if 'source_image' not in request.files or 'target_image' not in request.files:
            return jsonify({'error': 'Brak wymaganych plików'}), 400
        
        source_file = request.files['source_image']
        target_file = request.files['target_image']
        
        # Pobierz parametry
        matcher_type = request.form.get('matcher_type', 'basic')
        preset = request.form.get('preset', 'balanced')
        custom_config = request.form.get('custom_config')  # JSON string
        
        # Sprawdź typ matchera
        if matcher_type not in matchers:
            return jsonify({'error': f'Nieznany typ matchera: {matcher_type}'}), 400
        
        # Utwórz tymczasowe pliki
        with tempfile.TemporaryDirectory() as temp_dir:
            # Zapisz przesłane pliki
            source_path = os.path.join(temp_dir, 'source.jpg')
            target_path = os.path.join(temp_dir, 'target.jpg')
            output_path = os.path.join(temp_dir, 'result.jpg')
            
            source_file.save(source_path)
            target_file.save(target_path)
            
            # Wczytaj obrazy
            source_image = cv2.imread(source_path)
            target_image = cv2.imread(target_path)
            
            if source_image is None or target_image is None:
                return jsonify({'error': 'Nie można wczytać obrazów'}), 400
            
            source_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
            target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            
            # Przygotuj konfigurację
            if custom_config:
                try:
                    import json
                    weight_config = json.loads(custom_config)
                except json.JSONDecodeError:
                    return jsonify({'error': 'Nieprawidłowa konfiguracja JSON'}), 400
            else:
                preset_config = config_manager.get_preset(preset)
                weight_config = preset_config.weight_config.__dict__
            
            # Przetwórz obraz
            matcher = matchers[matcher_type]
            
            start_time = time.time()
            
            if matcher_type == 'adaptive':
                result, weight_map = matcher.adaptive_weight_matching(
                    source_rgb, target_rgb,
                    adaptation_method='content_aware'
                )
            elif matcher_type == 'optimized' and source_rgb.shape[0] * source_rgb.shape[1] > 1024 * 1024:
                result = matcher.process_large_image_tiled(
                    source_rgb, target_rgb,
                    weight_config=weight_config
                )
            else:
                result = matcher.process_rgb_image(
                    source_rgb, target_rgb,
                    weight_config=weight_config
                )
            
            processing_time = time.time() - start_time
            
            # Zapisz wynik
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr)
            
            # Przygotuj odpowiedź
            response_data = {
                'processing_time': processing_time,
                'matcher_type': matcher_type,
                'preset': preset,
                'source_size': source_rgb.shape,
                'target_size': target_rgb.shape,
                'result_size': result.shape
            }
            
            # Zwróć plik wynikowy
            return send_file(
                output_path,
                as_attachment=True,
                download_name='weighted_histogram_result.jpg',
                mimetype='image/jpeg'
            )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/weighted-histogram/presets', methods=['GET'])
def get_presets():
    """Zwraca dostępne presety konfiguracji"""
    try:
        presets = {
            'portrait': 'Optymalizacja dla portretów',
            'landscape': 'Optymalizacja dla krajobrazów',
            'low_light': 'Korekcja słabego oświetlenia',
            'high_contrast': 'Obrazy wysokiego kontrastu',
            'subtle': 'Delikatne dopasowanie',
            'balanced': 'Zrównoważone ustawienia'
        }
        
        return jsonify({
            'presets': presets,
            'default': 'balanced'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/weighted-histogram/config/<preset>', methods=['GET'])
def get_preset_config(preset):
    """Zwraca konfigurację dla określonego presetu"""
    try:
        preset_config = config_manager.get_preset(preset)
        
        return jsonify({
            'preset': preset,
            'weight_config': preset_config.weight_config.__dict__,
            'processing_config': preset_config.processing_config.__dict__,
            'adaptive_config': preset_config.adaptive_config.__dict__,
            'local_config': preset_config.local_config.__dict__
        })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/weighted-histogram/analyze', methods=['POST'])
def analyze_image():
    """Analizuje obraz i zwraca statystyki histogramu"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Brak pliku obrazu'}), 400
        
        image_file = request.files['image']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, 'image.jpg')
            image_file.save(image_path)
            
            # Wczytaj obraz
            image = cv2.imread(image_path)
            if image is None:
                return jsonify({'error': 'Nie można wczytać obrazu'}), 400
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Analizuj każdy kanał
            matcher = matchers['basic']
            analysis = {}
            
            for i, channel_name in enumerate(['red', 'green', 'blue']):
                channel_data = image_rgb[:, :, i]
                stats = matcher.calculate_histogram_stats(channel_data)
                
                analysis[channel_name] = {
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'median': float(stats['median']),
                    'p5': float(stats['p5']),
                    'p95': float(stats['p95']),
                    'histogram': stats['histogram'].tolist()
                }
            
            # Analiza ogólna
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            overall_stats = matcher.calculate_histogram_stats(gray)
            
            analysis['overall'] = {
                'mean': float(overall_stats['mean']),
                'std': float(overall_stats['std']),
                'median': float(overall_stats['median']),
                'p5': float(overall_stats['p5']),
                'p95': float(overall_stats['p95']),
                'histogram': overall_stats['histogram'].tolist()
            }
            
            # Rekomendacje
            recommendations = _generate_recommendations(analysis)
            
            return jsonify({
                'analysis': analysis,
                'recommendations': recommendations,
                'image_size': image_rgb.shape
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _generate_recommendations(analysis):
    """Generuje rekomendacje na podstawie analizy obrazu"""
    recommendations = []
    
    overall = analysis['overall']
    
    # Sprawdź ekspozycję
    if overall['mean'] < 80:
        recommendations.append({
            'type': 'exposure',
            'message': 'Obraz wydaje się niedoświetlony',
            'suggested_preset': 'low_light',
            'suggested_config': {
                'weight_type': 'segmented',
                'shadow_weight': 0.9,
                'midtone_weight': 1.0,
                'highlight_weight': 0.7
            }
        })
    elif overall['mean'] > 180:
        recommendations.append({
            'type': 'exposure',
            'message': 'Obraz wydaje się prześwietlony',
            'suggested_preset': 'high_contrast',
            'suggested_config': {
                'weight_type': 'segmented',
                'shadow_weight': 0.6,
                'midtone_weight': 0.8,
                'highlight_weight': 0.9
            }
        })
    
    # Sprawdź kontrast
    if overall['std'] < 30:
        recommendations.append({
            'type': 'contrast',
            'message': 'Niski kontrast - rozważ użycie local matching',
            'suggested_preset': 'high_contrast',
            'suggested_matcher': 'local'
        })
    
    # Sprawdź balans kolorów
    red_mean = analysis['red']['mean']
    green_mean = analysis['green']['mean']
    blue_mean = analysis['blue']['mean']
    
    color_diff = max(red_mean, green_mean, blue_mean) - min(red_mean, green_mean, blue_mean)
    
    if color_diff > 30:
        recommendations.append({
            'type': 'color_balance',
            'message': 'Niezbalansowane kolory - rozważ użycie adaptive matching',
            'suggested_matcher': 'adaptive',
            'suggested_method': 'content_aware'
        })
    
    return recommendations

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

---

## Rozwiązywanie Problemów i Debugowanie

### Diagnostyka Problemów

```python
class WeightedHistogramDiagnostics:
    """Klasa do diagnostyki problemów z weighted histogram matching"""
    
    def __init__(self):
        self.matcher = WeightedHistogramMatching()
    
    def diagnose_image_pair(self, source_image, target_image):
        """Diagnozuje potencjalne problemy z parą obrazów"""
        issues = []
        warnings = []
        
        # Sprawdź wymiary
        if source_image.shape != target_image.shape:
            issues.append({
                'type': 'dimension_mismatch',
                'message': f'Różne wymiary: source {source_image.shape} vs target {target_image.shape}',
                'severity': 'warning',
                'fix': 'Przeskaluj obrazy do tego samego rozmiaru'
            })
        
        # Sprawdź zakres wartości
        source_min, source_max = source_image.min(), source_image.max()
        target_min, target_max = target_image.min(), target_image.max()
        
        if source_min == source_max:
            issues.append({
                'type': 'uniform_source',
                'message': 'Obraz źródłowy jest jednolity (brak wariancji)',
                'severity': 'error',
                'fix': 'Użyj obrazu z większą wariancją kolorów'
            })
        
        if target_min == target_max:
            issues.append({
                'type': 'uniform_target',
                'message': 'Obraz docelowy jest jednolity (brak wariancji)',
                'severity': 'error',
                'fix': 'Użyj obrazu wzorcowego z większą wariancją kolorów'
            })
        
        # Sprawdź histogramy
        for channel in range(min(source_image.shape[2], target_image.shape[2])):
            source_hist = np.histogram(source_image[:, :, channel], bins=256, range=(0, 255))[0]
            target_hist = np.histogram(target_image[:, :, channel], bins=256, range=(0, 255))[0]
            
            # Sprawdź czy histogramy mają wystarczającą wariancję
            source_entropy = self._calculate_entropy(source_hist)
            target_entropy = self._calculate_entropy(target_hist)
            
            if source_entropy < 3.0:
                warnings.append({
                    'type': 'low_entropy_source',
                    'message': f'Niska entropia w kanale {channel} obrazu źródłowego',
                    'severity': 'warning',
                    'fix': 'Rozważ użycie adaptive matching'
                })
            
            if target_entropy < 3.0:
                warnings.append({
                    'type': 'low_entropy_target',
                    'message': f'Niska entropia w kanale {channel} obrazu docelowego',
                    'severity': 'warning',
                    'fix': 'Wybierz obraz wzorcowy z większą różnorodnością kolorów'
                })
        
        # Sprawdź podobieństwo histogramów
        similarity = self._calculate_histogram_similarity(source_image, target_image)
        
        if similarity > 0.95:
            warnings.append({
                'type': 'high_similarity',
                'message': f'Obrazy są bardzo podobne (podobieństwo: {similarity:.3f})',
                'severity': 'info',
                'fix': 'Efekt może być subtelny'
            })
        
        return {
            'issues': issues,
            'warnings': warnings,
            'similarity': similarity,
            'recommendation': self._generate_processing_recommendation(issues, warnings)
        }
    
    def _calculate_entropy(self, histogram):
        """Oblicza entropię histogramu"""
        # Normalizuj histogram
        hist_norm = histogram / np.sum(histogram)
        
        # Usuń zera aby uniknąć log(0)
        hist_norm = hist_norm[hist_norm > 0]
        
        # Oblicz entropię
        entropy = -np.sum(hist_norm * np.log2(hist_norm))
        
        return entropy
    
    def _calculate_histogram_similarity(self, image1, image2):
        """Oblicza podobieństwo histogramów między obrazami"""
        similarities = []
        
        for channel in range(min(image1.shape[2], image2.shape[2])):
            hist1 = np.histogram(image1[:, :, channel], bins=256, range=(0, 255))[0]
            hist2 = np.histogram(image2[:, :, channel], bins=256, range=(0, 255))[0]
            
            # Normalizuj histogramy
            hist1_norm = hist1 / np.sum(hist1)
            hist2_norm = hist2 / np.sum(hist2)
            
            # Oblicz korelację
            correlation = np.corrcoef(hist1_norm, hist2_norm)[0, 1]
            
            if not np.isnan(correlation):
                similarities.append(correlation)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _generate_processing_recommendation(self, issues, warnings):
        """Generuje rekomendacje przetwarzania na podstawie problemów"""
        has_errors = any(issue['severity'] == 'error' for issue in issues)
        
        if has_errors:
            return {
                'recommended_action': 'fix_errors',
                'message': 'Napraw błędy przed przetwarzaniem',
                'suggested_matcher': None
            }
        
        # Sprawdź czy są ostrzeżenia o niskiej entropii
        low_entropy_warnings = [w for w in warnings if 'entropy' in w['type']]
        
        if low_entropy_warnings:
            return {
                'recommended_action': 'use_adaptive',
                'message': 'Użyj adaptive matching dla obrazów o niskiej entropii',
                'suggested_matcher': 'adaptive',
                'suggested_method': 'content_aware'
            }
        
        # Sprawdź podobieństwo
        high_similarity_warnings = [w for w in warnings if w['type'] == 'high_similarity']
        
        if high_similarity_warnings:
            return {
                'recommended_action': 'use_subtle',
                'message': 'Użyj delikatnych ustawień dla podobnych obrazów',
                'suggested_matcher': 'basic',
                'suggested_preset': 'subtle'
            }
        
        return {
            'recommended_action': 'proceed',
            'message': 'Obrazy nadają się do przetwarzania',
            'suggested_matcher': 'basic',
            'suggested_preset': 'balanced'
        }
    
    def test_weight_function(self, weight_config):
        """Testuje funkcję wag"""
        try:
            weights = self.matcher.create_weight_function(**weight_config)
            
            issues = []
            
            # Sprawdź zakres
            if np.any(weights < 0) or np.any(weights > 1):
                issues.append({
                    'type': 'weight_range',
                    'message': 'Wagi poza zakresem [0, 1]',
                    'severity': 'error'
                })
            
            # Sprawdź czy funkcja nie jest płaska
            if np.std(weights) < 0.01:
                issues.append({
                    'type': 'flat_weights',
                    'message': 'Funkcja wag jest prawie płaska',
                    'severity': 'warning'
                })
            
            # Sprawdź ekstremalne wartości
            zero_count = np.sum(weights == 0)
            one_count = np.sum(weights == 1)
            
            if zero_count > 200:
                issues.append({
                    'type': 'too_many_zeros',
                    'message': f'Zbyt wiele zer w funkcji wag ({zero_count}/256)',
                    'severity': 'warning'
                })
            
            if one_count > 200:
                issues.append({
                    'type': 'too_many_ones',
                    'message': f'Zbyt wiele jedynek w funkcji wag ({one_count}/256)',
                    'severity': 'warning'
                })
            
            return {
                'valid': len([i for i in issues if i['severity'] == 'error']) == 0,
                'issues': issues,
                'weights': weights.tolist(),
                'statistics': {
                    'min': float(np.min(weights)),
                    'max': float(np.max(weights)),
                    'mean': float(np.mean(weights)),
                    'std': float(np.std(weights))
                }
            }
        
        except Exception as e:
            return {
                'valid': False,
                'issues': [{
                    'type': 'creation_error',
                    'message': f'Błąd tworzenia funkcji wag: {str(e)}',
                    'severity': 'error'
                }],
                'weights': None,
                'statistics': None
            }

# Funkcja diagnostyczna dla API
@app.route('/api/weighted-histogram/diagnose', methods=['POST'])
def diagnose_images():
    """Endpoint do diagnostyki obrazów"""
    try:
        if 'source_image' not in request.files or 'target_image' not in request.files:
            return jsonify({'error': 'Brak wymaganych plików'}), 400
        
        source_file = request.files['source_image']
        target_file = request.files['target_image']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Zapisz pliki
            source_path = os.path.join(temp_dir, 'source.jpg')
            target_path = os.path.join(temp_dir, 'target.jpg')
            
            source_file.save(source_path)
            target_file.save(target_path)
            
            # Wczytaj obrazy
            source_image = cv2.imread(source_path)
            target_image = cv2.imread(target_path)
            
            if source_image is None or target_image is None:
                return jsonify({'error': 'Nie można wczytać obrazów'}), 400
            
            source_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
            target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            
            # Przeprowadź diagnostykę
            diagnostics = WeightedHistogramDiagnostics()
            result = diagnostics.diagnose_image_pair(source_rgb, target_rgb)
            
            return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

---

## Dokumentacja API i Instrukcje Użytkowania

### Szybki Start

```python
# 1. Podstawowe użycie
from weighted_histogram_matching import WeightedHistogramMatching
import cv2

# Utwórz matcher
matcher = WeightedHistogramMatching()

# Wczytaj obrazy
source = cv2.imread('source.jpg')
target = cv2.imread('target.jpg')
source_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

# Przetwórz obraz
result = matcher.process_rgb_image(source_rgb, target_rgb)

# Zapisz wynik
result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imwrite('result.jpg', result_bgr)
```

### Zaawansowane Użycie

```python
# 2. Użycie z konfiguracją
config = {
    'weight_type': 'segmented',
    'shadow_weight': 0.8,
    'midtone_weight': 1.0,
    'highlight_weight': 0.6
}

result = matcher.process_rgb_image(
    source_rgb, target_rgb,
    weight_config=config
)

# 3. Adaptacyjne dopasowanie
adaptive_matcher = AdaptiveWeightedHistogramMatching()

result, weight_map = adaptive_matcher.adaptive_weight_matching(
    source_rgb, target_rgb,
    adaptation_method='content_aware'
)

# 4. Przetwarzanie z maskami
masked_matcher = MaskedWeightedHistogramMatching()

# Utwórz maskę
mask = np.zeros((height, width), dtype=bool)
mask[100:400, 100:400] = True  # Kwadrat w środku

result = masked_matcher.masked_weighted_matching(
    source_rgb, target_rgb,
    mask=mask
)
```

### Najlepsze Praktyki

1. **Wybór Algorytmu**:
   - `basic`: Szybkie, podstawowe dopasowanie
   - `optimized`: Dla dużych obrazów (>1MP)
   - `adaptive`: Dla złożonych obrazów z różnymi regionami
   - `local`: Dla obrazów o niskim kontraście
   - `masked`: Dla selektywnego przetwarzania

2. **Konfiguracja Wag**:
   - `segmented`: Najszybsza, dobra dla większości przypadków
   - `gaussian`: Gładkie przejścia
   - `linear`: Proste, przewidywalne efekty
   - `custom`: Pełna kontrola

3. **Optymalizacja Wydajności**:
   - Użyj `optimized` dla obrazów >1MP
   - Włącz `use_numba=True` dla powtarzalnych operacji
   - Rozważ `tiled processing` dla bardzo dużych obrazów

4. **Kontrola Jakości**:
   - Zawsze sprawdź diagnostykę przed przetwarzaniem
   - Użyj metryk jakości do oceny wyników
   - Testuj różne konfiguracje dla optymalnych rezultatów

---

## Podsumowanie Algorytmu

### Kluczowe Cechy

✅ **Elastyczność**: Różne funkcje wag i metody adaptacji  
✅ **Wydajność**: Optymalizacje Numba i przetwarzanie kafelkowe  
✅ **Jakość**: Zaawansowane metryki i kontrola jakości  
✅ **Użyteczność**: Gotowe presety i łatwa integracja  
✅ **Skalowalność**: Od małych obrazów do batch processing  

### Przypadki Użycia

- **Korekcja ekspozycji** portretów i zdjęć
- **Stylizacja** krajobrazów i scen
- **Selektywna korekcja** kolorów w regionach
- **Batch processing** dużych kolekcji zdjęć
- **Automatyczna poprawa** jakości obrazów

### Ograniczenia

- Nie nadaje się dla obrazów o bardzo niskim kontraście
- Wymaga dobrego obrazu wzorcowego
- Może wprowadzać artefakty przy ekstremalnych ustawieniach
- Czasochłonne dla bardzo dużych obrazów bez optymalizacji

---

**Autor**: GattoNero AI Assistant  
**Data utworzenia**: 2024-01-20  
**Ostatnia aktualizacja**: 2024-01-20  
**Wersja**: 1.0  
**Status**: ✅ Kompletna dokumentacja - wszystkie 4 części