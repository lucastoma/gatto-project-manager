# LAB Color Space Transfer - Część 3b: Przypadki Użycia i Diagnostyka

**Część 3b z 3: Praktyczne Zastosowania i Rozwiązywanie Problemów**

## 🟡 Poziom: Medium
**Trudność**: Średnia | **Czas implementacji**: 2-3 godziny | **Złożoność**: O(n)

---

## Przegląd Części 3b

Ta część koncentruje się na praktycznych zastosowaniach algorytmu LAB Color Transfer oraz diagnostyce i rozwiązywaniu problemów.

### Zawartość
- Przypadki użycia w różnych dziedzinach
- Diagnostyka problemów
- Rozwiązywanie typowych błędów
- Optymalizacja wydajności
- Wskazówki praktyczne

---

## Przypadki Użycia

### 1. Korekcja Oświetlenia Portretów

```python
def portrait_lighting_correction():
    """Przykład korekcji oświetlenia w portretach"""
    
    # Konfiguracja dla portretów
    portrait_config = LABTransferConfig()
    portrait_config.method = 'weighted'
    portrait_config.channel_weights = {
        'L': 0.7,  # Delikatna korekta jasności
        'a': 0.6,  # Umiarkowana korekta chromatyczności
        'b': 0.6
    }
    portrait_config.quality_check = True
    
    transfer = LABColorTransferAdvanced(portrait_config)
    
    # Przetwórz portret
    success = transfer.process_with_config(
        "portrait_bad_lighting.jpg",
        "reference_good_lighting.jpg",
        "portrait_corrected.jpg"
    )
    
    if success:
        print("✅ Korekcja oświetlenia portretu zakończona")
        
        # Analiza jakości
        quality_metrics = transfer.analyze_result_quality(
            "portrait_bad_lighting.jpg",
            "portrait_corrected.jpg"
        )
        
        print(f"📊 Metryki jakości:")
        print(f"   Delta E średnie: {quality_metrics['delta_e_mean']:.1f}")
        print(f"   Korelacja struktury: {quality_metrics['structure_correlation']:.3f}")
        
    else:
        print("❌ Błąd podczas korekcji")
    
    return success

# Przykład użycia dla różnych typów portretów
def batch_portrait_correction():
    """Wsadowa korekcja portretów"""
    
    portrait_types = {
        'studio': {
            'config': {'L': 0.8, 'a': 0.5, 'b': 0.5},
            'reference': 'studio_reference.jpg'
        },
        'outdoor': {
            'config': {'L': 0.6, 'a': 0.7, 'b': 0.7},
            'reference': 'outdoor_reference.jpg'
        },
        'indoor': {
            'config': {'L': 0.7, 'a': 0.6, 'b': 0.6},
            'reference': 'indoor_reference.jpg'
        }
    }
    
    input_portraits = [
        'portrait_001.jpg', 'portrait_002.jpg', 'portrait_003.jpg'
    ]
    
    results = {}
    
    for portrait_type, settings in portrait_types.items():
        print(f"\n🎯 Przetwarzanie portretów typu: {portrait_type}")
        
        config = LABTransferConfig()
        config.method = 'weighted'
        config.channel_weights = settings['config']
        
        transfer = LABColorTransferAdvanced(config)
        
        type_results = []
        
        for portrait_path in input_portraits:
            output_path = f"{portrait_type}_{portrait_path}"
            
            success = transfer.process_with_config(
                portrait_path,
                settings['reference'],
                output_path
            )
            
            type_results.append({
                'input': portrait_path,
                'output': output_path,
                'success': success
            })
            
            if success:
                print(f"   ✅ {portrait_path} → {output_path}")
            else:
                print(f"   ❌ Błąd: {portrait_path}")
        
        results[portrait_type] = type_results
    
    return results
```

### 2. Stylizacja Krajobrazów

```python
def landscape_stylization():
    """Przykład stylizacji krajobrazów"""
    
    # Konfiguracja dla krajobrazów
    landscape_config = LABTransferConfig()
    landscape_config.method = 'selective'
    landscape_config.transfer_channels = ['a', 'b']  # Tylko kolory, zachowaj jasność
    landscape_config.adaptation_method = 'luminance'
    
    transfer = LABColorTransferAdvanced(landscape_config)
    
    # Style do zastosowania
    styles = {
        'sunset': {
            'reference': 'reference_sunset.jpg',
            'description': 'Ciepłe, pomarańczowe tony zachodu słońca'
        },
        'autumn': {
            'reference': 'reference_autumn.jpg',
            'description': 'Złote i czerwone barwy jesieni'
        },
        'winter': {
            'reference': 'reference_winter.jpg',
            'description': 'Chłodne, niebieskie tony zimy'
        },
        'spring': {
            'reference': 'reference_spring.jpg',
            'description': 'Świeże, zielone kolory wiosny'
        }
    }
    
    source_image = "landscape_original.jpg"
    
    print("🌄 Rozpoczynam stylizację krajobrazów...")
    
    for style_name, style_info in styles.items():
        print(f"\n🎨 Aplikuję styl: {style_name}")
        print(f"   {style_info['description']}")
        
        output_path = f"landscape_{style_name}_style.jpg"
        
        success = transfer.process_with_config(
            source_image, 
            style_info['reference'], 
            output_path
        )
        
        if success:
            print(f"   ✅ Styl {style_name} zastosowany → {output_path}")
            
            # Analiza transferu
            analysis = transfer.analyze_color_transfer(
                source_image, 
                style_info['reference'], 
                output_path
            )
            
            print(f"   📊 Zmiana kolorów:")
            print(f"      Kanał a: {analysis['a_channel_change']:.1f}")
            print(f"      Kanał b: {analysis['b_channel_change']:.1f}")
            
        else:
            print(f"   ❌ Błąd przy stylu {style_name}")
    
    return True

# Zaawansowana stylizacja z maskami
def advanced_landscape_stylization():
    """Stylizacja z maskami dla różnych obszarów"""
    
    # Konfiguracja dla różnych obszarów krajobrazów
    area_configs = {
        'sky': {
            'mask_color': [135, 206, 235],  # Sky blue
            'tolerance': 30,
            'weights': {'L': 0.5, 'a': 0.8, 'b': 0.8}
        },
        'vegetation': {
            'mask_color': [34, 139, 34],  # Forest green
            'tolerance': 40,
            'weights': {'L': 0.6, 'a': 0.7, 'b': 0.7}
        },
        'water': {
            'mask_color': [0, 100, 200],  # Water blue
            'tolerance': 35,
            'weights': {'L': 0.4, 'a': 0.9, 'b': 0.9}
        }
    }
    
    source_path = "landscape_complex.jpg"
    reference_path = "reference_dramatic.jpg"
    
    # Wczytaj obrazy
    source_image = Image.open(source_path).convert('RGB')
    reference_image = Image.open(reference_path).convert('RGB')
    
    source_rgb = np.array(source_image)
    reference_rgb = np.array(reference_image)
    
    # Konwertuj do LAB
    transfer = LABColorTransferAdvanced()
    source_lab = transfer.rgb_to_lab_optimized(source_rgb)
    reference_lab = transfer.rgb_to_lab_optimized(reference_rgb)
    
    result_lab = source_lab.copy()
    
    print("🎭 Rozpoczynam zaawansowaną stylizację z maskami...")
    
    for area_name, config in area_configs.items():
        print(f"\n🎯 Przetwarzam obszar: {area_name}")
        
        # Utwórz maskę dla obszaru
        mask = create_color_mask(
            source_rgb, 
            config['mask_color'], 
            config['tolerance']
        )
        
        if np.sum(mask) > 0:  # Jeśli znaleziono piksele
            # Zastosuj transfer tylko do zamaskowanego obszaru
            area_result = transfer.weighted_lab_transfer(
                source_lab, 
                reference_lab, 
                config['weights']
            )
            
            # Aplikuj tylko do zamaskowanego obszaru
            result_lab[mask] = area_result[mask]
            
            print(f"   ✅ Przetworzono {np.sum(mask)} pikseli")
        else:
            print(f"   ⚠️ Nie znaleziono pikseli dla obszaru {area_name}")
    
    # Konwertuj z powrotem do RGB
    result_rgb = transfer.lab_to_rgb_optimized(result_lab)
    
    # Zapisz wynik
    output_path = "landscape_advanced_stylized.jpg"
    Image.fromarray(result_rgb).save(output_path)
    
    print(f"\n✅ Zaawansowana stylizacja zakończona → {output_path}")
    
    return output_path

def create_color_mask(image_rgb, target_color, tolerance):
    """Tworzy maskę dla pikseli podobnych do target_color"""
    diff = np.abs(image_rgb - np.array(target_color))
    distance = np.sqrt(np.sum(diff**2, axis=2))
    mask = distance < tolerance
    return mask
```

### 3. Batch Processing dla Fotografii

```python
def batch_photo_processing():
    """Przykład przetwarzania wsadowego"""
    
    # Konfiguracja dla zdjęć
    photo_config = LABTransferConfig()
    photo_config.method = 'adaptive'
    photo_config.adaptation_method = 'luminance'
    photo_config.quality_check = True
    photo_config.batch_size = 6
    
    transfer = LABColorTransferAdvanced(photo_config)
    
    # Lista zdjęć do przetworzenia
    photo_paths = [
        "photo_001.jpg", "photo_002.jpg", "photo_003.jpg",
        "photo_004.jpg", "photo_005.jpg", "photo_006.jpg"
    ]
    
    reference_path = "reference_professional.jpg"
    output_dir = "processed_photos"
    
    # Utwórz katalog wyjściowy
    os.makedirs(output_dir, exist_ok=True)
    
    print("📸 Rozpoczynam przetwarzanie wsadowe zdjęć...")
    
    # Przetwórz wsadowo
    results = transfer.process_image_batch(
        photo_paths, reference_path, output_dir, method='adaptive'
    )
    
    # Podsumowanie
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"\n📊 Podsumowanie przetwarzania wsadowego:")
    print(f"   Przetworzono: {successful}/{total} zdjęć")
    print(f"   Sukces: {successful/total*100:.1f}%")
    
    # Wyświetl szczegóły
    for result in results:
        if result['success']:
            print(f"   ✅ {result['input']} → {result['output']}")
            print(f"      Czas: {result['processing_time']:.2f}s")
            print(f"      Jakość: {result['quality_score']:.1f}")
        else:
            print(f"   ❌ {result['input']}: {result['error']}")
    
    # Generuj raport
    generate_batch_report(results, output_dir)
    
    return results

def generate_batch_report(results, output_dir):
    """Generuje raport z przetwarzania wsadowego"""
    
    report_path = os.path.join(output_dir, "batch_report.html")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LAB Transfer Batch Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .success {{ color: green; }}
            .error {{ color: red; }}
            .stats {{ background: #f0f0f0; padding: 10px; margin: 10px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>LAB Color Transfer - Raport Wsadowy</h1>
        
        <div class="stats">
            <h2>Statystyki</h2>
            <p>Całkowita liczba plików: {len(results)}</p>
            <p>Przetworzono pomyślnie: {sum(1 for r in results if r['success'])}</p>
            <p>Błędy: {sum(1 for r in results if not r['success'])}</p>
            <p>Średni czas przetwarzania: {np.mean([r.get('processing_time', 0) for r in results if r['success']]):.2f}s</p>
        </div>
        
        <h2>Szczegóły</h2>
        <table>
            <tr>
                <th>Plik wejściowy</th>
                <th>Status</th>
                <th>Plik wyjściowy</th>
                <th>Czas [s]</th>
                <th>Jakość</th>
                <th>Uwagi</th>
            </tr>
    """
    
    for result in results:
        status_class = "success" if result['success'] else "error"
        status_text = "✅ Sukces" if result['success'] else "❌ Błąd"
        
        html_content += f"""
            <tr>
                <td>{result['input']}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{result.get('output', '-')}</td>
                <td>{result.get('processing_time', '-')}</td>
                <td>{result.get('quality_score', '-')}</td>
                <td>{result.get('error', '-')}</td>
            </tr>
        """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📄 Raport zapisany: {report_path}")
```

### 4. Normalizacja Kolorów w E-commerce

```python
def ecommerce_color_normalization():
    """Normalizacja kolorów produktów dla e-commerce"""
    
    # Konfiguracja dla produktów e-commerce
    ecommerce_config = LABTransferConfig()
    ecommerce_config.method = 'weighted'
    ecommerce_config.channel_weights = {
        'L': 0.8,  # Ważna jasność dla widoczności produktu
        'a': 0.9,  # Ważne kolory dla wierności produktu
        'b': 0.9
    }
    ecommerce_config.quality_check = True
    
    transfer = LABColorTransferAdvanced(ecommerce_config)
    
    # Kategorie produktów z różnymi referencjami
    product_categories = {
        'clothing': {
            'reference': 'clothing_reference_white_background.jpg',
            'description': 'Ubrania na białym tle'
        },
        'electronics': {
            'reference': 'electronics_reference_neutral.jpg',
            'description': 'Elektronika w neutralnym oświetleniu'
        },
        'jewelry': {
            'reference': 'jewelry_reference_studio.jpg',
            'description': 'Biżuteria w oświetleniu studyjnym'
        }
    }
    
    # Produkty do przetworzenia
    products = [
        {'path': 'shirt_red_poor_lighting.jpg', 'category': 'clothing'},
        {'path': 'laptop_yellow_tint.jpg', 'category': 'electronics'},
        {'path': 'ring_blue_cast.jpg', 'category': 'jewelry'},
        {'path': 'dress_green_cast.jpg', 'category': 'clothing'}
    ]
    
    print("🛍️ Rozpoczynam normalizację kolorów dla e-commerce...")
    
    results = []
    
    for product in products:
        category = product['category']
        reference_info = product_categories[category]
        
        print(f"\n📦 Przetwarzam: {product['path']}")
        print(f"   Kategoria: {category}")
        print(f"   Referencja: {reference_info['description']}")
        
        output_path = f"normalized_{product['path']}"
        
        success = transfer.process_with_config(
            product['path'],
            reference_info['reference'],
            output_path
        )
        
        if success:
            # Sprawdź jakość normalizacji
            quality_metrics = transfer.analyze_result_quality(
                product['path'],
                output_path
            )
            
            # Sprawdź czy kolory są w akceptowalnym zakresie
            color_validation = validate_ecommerce_colors(output_path)
            
            result = {
                'input': product['path'],
                'output': output_path,
                'category': category,
                'success': True,
                'quality_score': quality_metrics['overall_score'],
                'color_validation': color_validation
            }
            
            print(f"   ✅ Znormalizowano → {output_path}")
            print(f"   📊 Jakość: {quality_metrics['overall_score']:.1f}")
            print(f"   🎨 Walidacja kolorów: {'✅' if color_validation['passed'] else '❌'}")
            
        else:
            result = {
                'input': product['path'],
                'output': None,
                'category': category,
                'success': False,
                'error': 'Processing failed'
            }
            print(f"   ❌ Błąd podczas normalizacji")
        
        results.append(result)
    
    # Generuj raport dla e-commerce
    generate_ecommerce_report(results)
    
    return results

def validate_ecommerce_colors(image_path):
    """Waliduje kolory dla standardów e-commerce"""
    
    image = Image.open(image_path).convert('RGB')
    rgb_array = np.array(image)
    
    # Sprawdź czy tło jest wystarczająco białe
    # (zakładamy, że tło to brzegi obrazu)
    edges = np.concatenate([
        rgb_array[0, :].flatten(),  # górna krawędź
        rgb_array[-1, :].flatten(),  # dolna krawędź
        rgb_array[:, 0].flatten(),  # lewa krawędź
        rgb_array[:, -1].flatten()  # prawa krawędź
    ])
    
    edge_mean = np.mean(edges)
    background_white = edge_mean > 240  # Tło powinno być bardzo jasne
    
    # Sprawdź kontrast
    gray = np.mean(rgb_array, axis=2)
    contrast = np.std(gray)
    good_contrast = contrast > 30  # Wystarczający kontrast
    
    # Sprawdź nasycenie kolorów
    hsv = rgb_to_hsv(rgb_array)
    saturation = hsv[:, :, 1]
    avg_saturation = np.mean(saturation)
    good_saturation = 0.1 < avg_saturation < 0.8  # Nie za szare, nie za nasycone
    
    validation_result = {
        'passed': background_white and good_contrast and good_saturation,
        'background_white': background_white,
        'good_contrast': good_contrast,
        'good_saturation': good_saturation,
        'metrics': {
            'background_brightness': edge_mean,
            'contrast': contrast,
            'avg_saturation': avg_saturation
        }
    }
    
    return validation_result

def rgb_to_hsv(rgb):
    """Konwersja RGB do HSV"""
    rgb = rgb.astype(float) / 255.0
    
    max_val = np.max(rgb, axis=2)
    min_val = np.min(rgb, axis=2)
    diff = max_val - min_val
    
    # Value
    v = max_val
    
    # Saturation
    s = np.where(max_val != 0, diff / max_val, 0)
    
    # Hue
    h = np.zeros_like(max_val)
    
    # Red is max
    idx = (max_val == rgb[:, :, 0]) & (diff != 0)
    h[idx] = (60 * ((rgb[:, :, 1][idx] - rgb[:, :, 2][idx]) / diff[idx]) + 360) % 360
    
    # Green is max
    idx = (max_val == rgb[:, :, 1]) & (diff != 0)
    h[idx] = (60 * ((rgb[:, :, 2][idx] - rgb[:, :, 0][idx]) / diff[idx]) + 120) % 360
    
    # Blue is max
    idx = (max_val == rgb[:, :, 2]) & (diff != 0)
    h[idx] = (60 * ((rgb[:, :, 0][idx] - rgb[:, :, 1][idx]) / diff[idx]) + 240) % 360
    
    h = h / 360.0  # Normalize to [0, 1]
    
    return np.stack([h, s, v], axis=2)

def generate_ecommerce_report(results):
    """Generuje raport dla e-commerce"""
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n📊 Raport E-commerce:")
    print(f"   Przetworzono: {len(successful)}/{len(results)} produktów")
    print(f"   Sukces: {len(successful)/len(results)*100:.1f}%")
    
    if successful:
        avg_quality = np.mean([r['quality_score'] for r in successful])
        passed_validation = sum(1 for r in successful if r.get('color_validation', {}).get('passed', False))
        
        print(f"   Średnia jakość: {avg_quality:.1f}")
        print(f"   Walidacja kolorów: {passed_validation}/{len(successful)} przeszło")
    
    if failed:
        print(f"\n❌ Błędy:")
        for result in failed:
            print(f"   {result['input']}: {result.get('error', 'Unknown error')}")
```

---

## Rozwiązywanie Problemów

### Diagnostyka Problemów

```python
class LABTransferDiagnostics:
    def __init__(self, transfer_instance):
        self.transfer = transfer_instance
    
    def diagnose_conversion_issues(self, rgb_image):
        """Diagnozuje problemy z konwersją kolorów"""
        issues = []
        
        try:
            # Test konwersji RGB → LAB
            lab = self.transfer.rgb_to_lab_optimized(rgb_image)
            
            # Sprawdź zakresy LAB
            L_min, L_max = np.min(lab[:, :, 0]), np.max(lab[:, :, 0])
            a_min, a_max = np.min(lab[:, :, 1]), np.max(lab[:, :, 1])
            b_min, b_max = np.min(lab[:, :, 2]), np.max(lab[:, :, 2])
            
            if L_min < 0 or L_max > 100:
                issues.append(f"L channel out of range: [{L_min:.1f}, {L_max:.1f}]")
            
            if a_min < -128 or a_max > 127:
                issues.append(f"a channel out of range: [{a_min:.1f}, {a_max:.1f}]")
            
            if b_min < -128 or b_max > 127:
                issues.append(f"b channel out of range: [{b_min:.1f}, {b_max:.1f}]")
            
            # Test round-trip conversion
            rgb_recovered = self.transfer.lab_to_rgb_optimized(lab)
            
            # Sprawdź błąd round-trip
            diff = np.abs(rgb_image.astype(float) - rgb_recovered.astype(float))
            mean_error = np.mean(diff)
            max_error = np.max(diff)
            
            if mean_error > 5.0:
                issues.append(f"High round-trip error: mean={mean_error:.1f}")
            
            if max_error > 20.0:
                issues.append(f"Very high max round-trip error: {max_error:.1f}")
            
        except Exception as e:
            issues.append(f"Conversion failed: {str(e)}")
        
        return issues
    
    def diagnose_transfer_quality(self, source_lab, target_lab, result_lab):
        """Diagnozuje jakość transferu"""
        issues = []
        warnings = []
        
        try:
            # Sprawdź czy transfer rzeczywiście zmienił obraz
            if np.allclose(source_lab, result_lab, atol=1.0):
                issues.append("Transfer had no effect - result identical to source")
            
            # Sprawdź czy wynik nie jest zbyt podobny do targetu (over-transfer)
            if np.allclose(result_lab, target_lab, atol=5.0):
                warnings.append("Result very similar to target - possible over-transfer")
            
            # Sprawdź Delta E
            delta_e = self.transfer.calculate_delta_e_lab(source_lab, result_lab)
            mean_delta_e = np.mean(delta_e)
            max_delta_e = np.max(delta_e)
            
            if mean_delta_e < 2.0:
                warnings.append(f"Low color change: mean Delta E = {mean_delta_e:.1f}")
            elif mean_delta_e > 50.0:
                issues.append(f"Excessive color change: mean Delta E = {mean_delta_e:.1f}")
            
            if max_delta_e > 100.0:
                issues.append(f"Extreme local color change: max Delta E = {max_delta_e:.1f}")
            
            # Sprawdź zachowanie struktury
            source_structure = self.calculate_structure_metric(source_lab)
            result_structure = self.calculate_structure_metric(result_lab)
            
            structure_correlation = np.corrcoef(
                source_structure.flatten(), 
                result_structure.flatten()
            )[0, 1]
            
            if structure_correlation < 0.7:
                issues.append(f"Poor structure preservation: correlation = {structure_correlation:.3f}")
            elif structure_correlation < 0.85:
                warnings.append(f"Moderate structure change: correlation = {structure_correlation:.3f}")
            
        except Exception as e:
            issues.append(f"Quality analysis failed: {str(e)}")
        
        return issues, warnings
    
    def calculate_structure_metric(self, lab_image):
        """Oblicza metrykę struktury obrazu"""
        # Użyj kanału L (jasność) jako podstawy struktury
        L_channel = lab_image[:, :, 0]
        
        # Oblicz gradient
        grad_x = np.gradient(L_channel, axis=1)
        grad_y = np.gradient(L_channel, axis=0)
        
        # Magnitude gradientu jako miara struktury
        structure = np.sqrt(grad_x**2 + grad_y**2)
        
        return structure
    
    def suggest_fixes(self, issues, warnings):
        """Sugeruje poprawki na podstawie zdiagnozowanych problemów"""
        suggestions = []
        
        for issue in issues:
            if "out of range" in issue:
                suggestions.append("Check input image format and color space")
                suggestions.append("Ensure RGB values are in [0, 255] range")
            
            elif "round-trip error" in issue:
                suggestions.append("Use higher precision in color conversion")
                suggestions.append("Check for numerical instabilities")
            
            elif "no effect" in issue:
                suggestions.append("Increase transfer weights")
                suggestions.append("Check if source and target are too similar")
                suggestions.append("Try different transfer method")
            
            elif "Excessive color change" in issue:
                suggestions.append("Reduce transfer weights")
                suggestions.append("Use selective transfer instead of global")
                suggestions.append("Check target image quality")
            
            elif "Poor structure preservation" in issue:
                suggestions.append("Use weighted transfer with lower L channel weight")
                suggestions.append("Try selective transfer (a, b channels only)")
                suggestions.append("Check if images are too different")
        
        for warning in warnings:
            if "over-transfer" in warning:
                suggestions.append("Consider reducing transfer strength")
            
            elif "Low color change" in warning:
                suggestions.append("Increase transfer weights if more change desired")
                suggestions.append("Check if source and target are already similar")
        
        # Usuń duplikaty
        suggestions = list(set(suggestions))
        
        return suggestions
    
    def run_full_diagnosis(self, source_path, target_path, result_path=None):
        """Uruchamia pełną diagnozę"""
        print("🔍 Rozpoczynam pełną diagnozę LAB Transfer...")
        
        # Wczytaj obrazy
        source_image = Image.open(source_path).convert('RGB')
        target_image = Image.open(target_path).convert('RGB')
        
        source_rgb = np.array(source_image)
        target_rgb = np.array(target_image)
        
        print(f"\n📊 Informacje o obrazach:")
        print(f"   Source: {source_rgb.shape} - {source_path}")
        print(f"   Target: {target_rgb.shape} - {target_path}")
        
        # Diagnoza konwersji
        print(f"\n🔬 Diagnoza konwersji kolorów:")
        source_issues = self.diagnose_conversion_issues(source_rgb)
        target_issues = self.diagnose_conversion_issues(target_rgb)
        
        if source_issues:
            print(f"   ❌ Problemy z obrazem źródłowym:")
            for issue in source_issues:
                print(f"      - {issue}")
        else:
            print(f"   ✅ Obraz źródłowy: OK")
        
        if target_issues:
            print(f"   ❌ Problemy z obrazem docelowym:")
            for issue in target_issues:
                print(f"      - {issue}")
        else:
            print(f"   ✅ Obraz docelowy: OK")
        
        # Jeśli podano wynik, diagnozuj transfer
        if result_path and os.path.exists(result_path):
            print(f"\n🎯 Diagnoza jakości transferu:")
            
            result_image = Image.open(result_path).convert('RGB')
            result_rgb = np.array(result_image)
            
            # Konwertuj do LAB
            source_lab = self.transfer.rgb_to_lab_optimized(source_rgb)
            target_lab = self.transfer.rgb_to_lab_optimized(target_rgb)
            result_lab = self.transfer.rgb_to_lab_optimized(result_rgb)
            
            transfer_issues, transfer_warnings = self.diagnose_transfer_quality(
                source_lab, target_lab, result_lab
            )
            
            if transfer_issues:
                print(f"   ❌ Problemy z transferem:")
                for issue in transfer_issues:
                    print(f"      - {issue}")
            else:
                print(f"   ✅ Transfer: OK")
            
            if transfer_warnings:
                print(f"   ⚠️ Ostrzeżenia:")
                for warning in transfer_warnings:
                    print(f"      - {warning}")
        
        # Sugestie
        all_issues = source_issues + target_issues
        if result_path and os.path.exists(result_path):
            all_issues += transfer_issues
            all_warnings = transfer_warnings
        else:
            all_warnings = []
        
        if all_issues or all_warnings:
            suggestions = self.suggest_fixes(all_issues, all_warnings)
            
            print(f"\n💡 Sugestie poprawek:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"   {i}. {suggestion}")
        else:
            print(f"\n✅ Wszystko wygląda dobrze!")
        
        return {
            'source_issues': source_issues,
            'target_issues': target_issues,
            'transfer_issues': transfer_issues if result_path else [],
            'transfer_warnings': transfer_warnings if result_path else [],
            'suggestions': suggestions if (all_issues or all_warnings) else []
        }
```

### Typowe Problemy i Rozwiązania

```python
class LABTransferTroubleshooting:
    """Klasa do rozwiązywania typowych problemów"""
    
    @staticmethod
    def fix_poor_color_conversion():
        """Rozwiązuje problemy z konwersją kolorów"""
        print("🔧 Rozwiązywanie problemów z konwersją kolorów:")
        
        fixes = {
            "Sprawdź format obrazu": [
                "Upewnij się, że obraz jest w formacie RGB",
                "Konwertuj CMYK/Grayscale do RGB przed przetwarzaniem",
                "Sprawdź czy obraz nie ma kanału alpha"
            ],
            "Sprawdź zakresy wartości": [
                "RGB powinno być w zakresie [0, 255]",
                "Sprawdź czy nie ma wartości ujemnych",
                "Sprawdź czy nie ma wartości > 255"
            ],
            "Optymalizuj precyzję": [
                "Użyj float64 dla obliczeń pośrednich",
                "Zaokrąglaj dopiero na końcu",
                "Sprawdź numeryczną stabilność"
            ]
        }
        
        for category, solutions in fixes.items():
            print(f"\n📋 {category}:")
            for solution in solutions:
                print(f"   • {solution}")
    
    @staticmethod
    def fix_poor_transfer_quality():
        """Rozwiązuje problemy z jakością transferu"""
        print("🔧 Rozwiązywanie problemów z jakością transferu:")
        
        fixes = {
            "Transfer zbyt słaby": [
                "Zwiększ wagi kanałów (channel_weights)",
                "Sprawdź czy obrazy nie są zbyt podobne",
                "Użyj metody 'weighted' zamiast 'basic'",
                "Sprawdź czy target ma wystarczającą różnorodność kolorów"
            ],
            "Transfer zbyt silny": [
                "Zmniejsz wagi kanałów",
                "Użyj metody 'selective' tylko dla kanałów a, b",
                "Sprawdź czy obrazy nie są zbyt różne",
                "Rozważ preprocessing obrazów"
            ],
            "Utrata szczegółów": [
                "Zmniejsz wagę kanału L",
                "Użyj selective transfer",
                "Sprawdź rozdzielczość obrazów",
                "Rozważ lokalny transfer z maskami"
            ],
            "Artefakty kolorowe": [
                "Sprawdź jakość obrazu docelowego",
                "Użyj adaptacyjnej metody",
                "Sprawdź czy obrazy są z tej samej domeny",
                "Rozważ preprocessing (denoising)"
            ]
        }
        
        for category, solutions in fixes.items():
            print(f"\n📋 {category}:")
            for solution in solutions:
                print(f"   • {solution}")
    
    @staticmethod
    def fix_performance_issues():
        """Rozwiązuje problemy z wydajnością"""
        print("🔧 Rozwiązywanie problemów z wydajnością:")
        
        fixes = {
            "Wolna konwersja kolorów": [
                "Użyj vectorized operations (NumPy)",
                "Sprawdź czy używasz float32 zamiast float64",
                "Rozważ batch processing",
                "Optymalizuj rozmiar obrazów"
            ],
            "Wysokie zużycie pamięci": [
                "Przetwarzaj obrazy w kawałkach (chunks)",
                "Zwolnij niepotrzebne zmienne",
                "Użyj in-place operations gdzie możliwe",
                "Sprawdź czy nie trzymasz kopii obrazów"
            ],
            "Wolny transfer": [
                "Użyj metody 'basic' dla szybkości",
                "Zmniejsz rozdzielczość dla testów",
                "Rozważ downsampling target image",
                "Optymalizuj obliczenia statystyk"
            ]
        }
        
        for category, solutions in fixes.items():
            print(f"\n📋 {category}:")
            for solution in solutions:
                print(f"   • {solution}")
    
    @staticmethod
    def run_automated_fixes(source_path, target_path, output_path):
        """Uruchamia automatyczne poprawki"""
        print("🤖 Uruchamiam automatyczne poprawki...")
        
        try:
            # Wczytaj i sprawdź obrazy
            source_image = Image.open(source_path).convert('RGB')
            target_image = Image.open(target_path).convert('RGB')
            
            source_rgb = np.array(source_image)
            target_rgb = np.array(target_image)
            
            print(f"✅ Obrazy wczytane pomyślnie")
            
            # Automatyczne poprawki
            fixes_applied = []
            
            # 1. Sprawdź i popraw zakresy
            if np.any(source_rgb < 0) or np.any(source_rgb > 255):
                source_rgb = np.clip(source_rgb, 0, 255)
                fixes_applied.append("Clipped source RGB values to [0, 255]")
            
            if np.any(target_rgb < 0) or np.any(target_rgb > 255):
                target_rgb = np.clip(target_rgb, 0, 255)
                fixes_applied.append("Clipped target RGB values to [0, 255]")
            
            # 2. Sprawdź rozmiary i dostosuj jeśli potrzeba
            if source_rgb.shape != target_rgb.shape:
                # Zmień rozmiar target do source
                target_image_resized = Image.fromarray(target_rgb).resize(
                    (source_rgb.shape[1], source_rgb.shape[0]), 
                    Image.Resampling.LANCZOS
                )
                target_rgb = np.array(target_image_resized)
                fixes_applied.append(f"Resized target to match source: {source_rgb.shape}")
            
            # 3. Sprawdź podobieństwo obrazów
            similarity = np.corrcoef(
                source_rgb.flatten(), 
                target_rgb.flatten()
            )[0, 1]
            
            # 4. Wybierz optymalną konfigurację na podstawie podobieństwa
            if similarity > 0.8:
                # Obrazy bardzo podobne - użyj delikatnego transferu
                config = LABTransferConfig()
                config.method = 'weighted'
                config.channel_weights = {'L': 0.3, 'a': 0.5, 'b': 0.5}
                fixes_applied.append("Used gentle transfer for similar images")
                
            elif similarity < 0.3:
                # Obrazy bardzo różne - użyj selektywnego transferu
                config = LABTransferConfig()
                config.method = 'selective'
                config.transfer_channels = ['a', 'b']
                fixes_applied.append("Used selective transfer for very different images")
                
            else:
                # Obrazy umiarkowanie różne - standardowy transfer
                config = LABTransferConfig()
                config.method = 'weighted'
                config.channel_weights = {'L': 0.6, 'a': 0.8, 'b': 0.8}
                fixes_applied.append("Used standard weighted transfer")
            
            # 5. Wykonaj transfer
            transfer = LABColorTransferAdvanced(config)
            
            # Zapisz poprawione obrazy tymczasowo
            temp_source = "temp_source_fixed.jpg"
            temp_target = "temp_target_fixed.jpg"
            
            Image.fromarray(source_rgb).save(temp_source)
            Image.fromarray(target_rgb).save(temp_target)
            
            success = transfer.process_with_config(
                temp_source, temp_target, output_path
            )
            
            # Cleanup
            os.remove(temp_source)
            os.remove(temp_target)
            
            if success:
                print(f"✅ Transfer zakończony pomyślnie")
                print(f"📄 Wynik zapisany: {output_path}")
                
                if fixes_applied:
                    print(f"\n🔧 Zastosowane poprawki:")
                    for fix in fixes_applied:
                        print(f"   • {fix}")
                
                return True
            else:
                print(f"❌ Transfer nie powiódł się")
                return False
                
        except Exception as e:
            print(f"❌ Błąd podczas automatycznych poprawek: {str(e)}")
            return False
```

---

## Nawigacja

**◀️ Poprzednia część**: [Testy i Benchmarki](gatto-WORKING-03-algorithms-05-medium-02-lab-transfer-3aof3.md)  
**▶️ Następna część**: [Integracja i Podsumowanie](gatto-WORKING-03-algorithms-05-medium-02-lab-transfer-3cof3.md)  
**🏠 Powrót do**: [Spis Treści Algorytmów](gatto-WORKING-03-algorithms-toc.md)

---

*Ostatnia aktualizacja: 2024-01-20*  
*Autor: GattoNero AI Assistant*  
*Wersja: 2.0*  
*Status: Część 3b - Przypadki użycia i diagnostyka* ✅