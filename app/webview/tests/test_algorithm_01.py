#!/usr/bin/env python3
"""
Testy dla Algorithm 01 - Palette w WebView
Testowanie interfejsu webowego dla ekstrakcji palety kolorów.
"""

import os
import sys
import unittest
import tempfile
from PIL import Image
import numpy as np

# Dodaj ścieżkę do głównego katalogu projektu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from tests.base_test_case import BaseAlgorithmTestCase
except ImportError:
    # Fallback jeśli BaseAlgorithmTestCase nie jest dostępny
    import tempfile
    import shutil
    class BaseAlgorithmTestCase(unittest.TestCase):
        def setUp(self):
            self.test_dir = tempfile.mkdtemp()
            
        def tearDown(self):
            if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
                
        def create_test_image(self, filename, shape=(100, 100, 3), color=None, arr_data=None):
            """Tworzy testowy obraz."""
            if arr_data is not None:
                image_array = arr_data
            elif color is None:
                # Losowe kolory
                image_array = np.random.randint(0, 256, shape, dtype=np.uint8)
            else:
                # Jednolity kolor
                image_array = np.full(shape, color, dtype=np.uint8)
            
            image = Image.fromarray(image_array)
            filepath = os.path.join(self.test_dir, filename)
            image.save(filepath)
            return filepath

class TestAlgorithm01WebView(BaseAlgorithmTestCase):
    """Testy dla Algorithm 01 w WebView."""
    
    def setUp(self):
        """Przygotowanie testów."""
        super().setUp()  # Wywołaj setUp z klasy bazowej
        self.test_images = []
    
    def test_create_simple_palette_image(self):
        """Test: Tworzenie prostego obrazu do testowania palety."""
        # Obraz z 3 wyraźnymi kolorami
        image_path = self.create_test_image(
            "palette_test_simple.png", 
            shape=(60, 60, 3), 
            color=[255, 0, 0]  # Czerwony
        )
        # Nie dodawaj do self.test_images, bo BaseAlgorithmTestCase automatycznie czyści test_dir
        
        self.assertTrue(os.path.exists(image_path))
        print(f"✅ [WEBVIEW TEST] Utworzono prosty obraz testowy: {image_path}")
        print(f"   📝 Użyj tego obrazu w webview: http://localhost:5000/webview/algorithm_01")
    
    def test_create_complex_palette_image(self):
        """Test: Tworzenie złożonego obrazu z wieloma kolorami."""
        # Tworzenie obrazu z gradientem kolorów
        shape = (100, 100, 3)
        image_array = np.zeros(shape, dtype=np.uint8)
        
        # Gradient poziomy - różne kolory
        for x in range(shape[1]):
            for y in range(shape[0]):
                image_array[y, x] = [
                    int(255 * x / shape[1]),  # Czerwony gradient
                    int(255 * y / shape[0]),  # Zielony gradient
                    128  # Stały niebieski
                ]
        
        image_path = self.create_test_image("palette_test_complex.png", arr_data=image_array)
        # Nie dodawaj do self.test_images, bo BaseAlgorithmTestCase automatycznie czyści test_dir
        
        self.assertTrue(os.path.exists(image_path))
        print(f"✅ [WEBVIEW TEST] Utworzono złożony obraz testowy: {image_path}")
        print(f"   📝 Użyj tego obrazu w webview: http://localhost:5000/webview/algorithm_01")
    
    def test_create_noise_image(self):
        """Test: Tworzenie obrazu z szumem do testowania."""
        image_path = self.create_test_image(
            "palette_test_noise.png", 
            shape=(80, 80, 3)
        )
        # Nie dodawaj do self.test_images, bo BaseAlgorithmTestCase automatycznie czyści test_dir
        
        self.assertTrue(os.path.exists(image_path))
        print(f"✅ [WEBVIEW TEST] Utworzono obraz z szumem: {image_path}")
        print(f"   📝 Użyj tego obrazu w webview: http://localhost:5000/webview/algorithm_01")
    
    def test_create_palette_test_suite(self):
        """Test: Tworzenie pełnego zestawu obrazów testowych."""
        test_cases = [
            {
                'name': 'red_solid.png',
                'description': 'Jednolity czerwony obraz',
                'shape': (50, 50, 3),
                'color': [255, 0, 0]
            },
            {
                'name': 'green_solid.png', 
                'description': 'Jednolity zielony obraz',
                'shape': (50, 50, 3),
                'color': [0, 255, 0]
            },
            {
                'name': 'blue_solid.png',
                'description': 'Jednolity niebieski obraz', 
                'shape': (50, 50, 3),
                'color': [0, 0, 255]
            }
        ]
        
        created_images = []
        
        for test_case in test_cases:
            image_path = self.create_test_image(
                test_case['name'],
                test_case['shape'],
                test_case['color']
            )
            # Nie dodawaj do self.test_images, bo BaseAlgorithmTestCase automatycznie czyści test_dir
            created_images.append({
                'path': image_path,
                'description': test_case['description']
            })
            
            self.assertTrue(os.path.exists(image_path))
        
        print(f"\n🎯 [WEBVIEW TEST SUITE] Utworzono {len(created_images)} obrazów testowych:")
        for img in created_images:
            print(f"   📁 {img['path']} - {img['description']}")
        
        print(f"\n🌐 Testuj w webview:")
        print(f"   🔗 http://localhost:5000/webview/algorithm_01")
        print(f"\n📋 Parametry do testowania:")
        print(f"   • Liczba kolorów: 1-5 (dla obrazów jednolitych)")
        print(f"   • Metoda: K-Means vs Median Cut")
        print(f"   • Jakość: 1-10")
    
    def test_webview_instructions(self):
        """Test: Wyświetlenie instrukcji testowania w webview."""
        print(f"\n🧪 [INSTRUKCJE TESTOWANIA ALGORITHM 01 W WEBVIEW]")
        print(f"\n1. 🌐 Otwórz webview:")
        print(f"   http://localhost:5000/webview/algorithm_01")
        
        print(f"\n2. 📤 Upload obrazu:")
        print(f"   • Przeciągnij obraz do obszaru uploadu")
        print(f"   • Lub kliknij i wybierz plik")
        print(f"   • Obsługiwane: JPEG, PNG (max 10MB)")
        
        print(f"\n3. ⚙️ Konfiguracja parametrów:")
        print(f"   • Liczba kolorów: 1-20 (zalecane: 3-8)")
        print(f"   • Metoda: K-Means (szybka) vs Median Cut (dokładna)")
        print(f"   • Jakość: 1-10 (wyższa = dokładniejsza, ale wolniejsza)")
        print(f"   • Metadane: włącz dla dodatkowych informacji")
        
        print(f"\n4. 🧪 Przetwarzanie:")
        print(f"   • Kliknij 'Przetwórz Obraz'")
        print(f"   • Obserwuj pasek postępu")
        print(f"   • Sprawdź logi w czasie rzeczywistym")
        
        print(f"\n5. 📊 Analiza wyników:")
        print(f"   • Paleta kolorów z kodami HEX")
        print(f"   • Statystyki przetwarzania")
        print(f"   • Porównanie przed/po")
        print(f"   • Możliwość eksportu wyników")
        
        print(f"\n6. 🔄 Testowanie różnych scenariuszy:")
        print(f"   • Obrazy jednolite (1-2 kolory)")
        print(f"   • Obrazy z gradientem (5-10 kolorów)")
        print(f"   • Zdjęcia rzeczywiste (8-15 kolorów)")
        print(f"   • Obrazy z szumem (test odporności)")
        
        print(f"\n✅ Test instrukcji zakończony pomyślnie")

if __name__ == "__main__":
    # Uruchom testy
    unittest.main(verbosity=2)