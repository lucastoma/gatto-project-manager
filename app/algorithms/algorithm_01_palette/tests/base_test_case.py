# -*- coding: utf-8 -*-
import unittest
import tempfile
import shutil
import numpy as np
import cv2
import os

class BaseAlgorithmTestCase(unittest.TestCase):
    """
    Uniwersalna klasa bazowa dla wszystkich testów algorytmów.
    Automatycznie zarządza tworzeniem i czyszczeniem tymczasowego
    folderu na pliki testowe.
    """
    def setUp(self):
        """Metoda wywoływana przed każdym testem w klasie."""
        # Stwórz unikalny, tymczasowy folder dla tego zestawu testów
        self.test_dir = tempfile.mkdtemp()
        print(f"\n[TEST ENV] Stworzono folder tymczasowy: {self.test_dir}")

    def tearDown(self):
        """Metoda wywoływana po każdym teście w klasie."""
        # Usuń cały folder tymczasowy wraz z zawartością
        shutil.rmtree(self.test_dir)
        print(f"[TEST ENV] Usunięto folder tymczasowy: {self.test_dir}")

    def create_test_image(self, filename: str, shape: tuple = (64, 64, 3), color: list | None = None, arr_data=None) -> str:
        """
        Tworzy prosty obraz testowy i zapisuje go w folderze tymczasowym.

        Args:
            filename (str): Nazwa pliku do zapisu (np. 'master.png').
            shape (tuple): Kształt obrazu (wysokość, szerokość, kanały).
            color (list | None, optional): Kolor RGB do wypełnienia obrazu. 
                                    Jeśli None, generowany jest losowy szum.
            arr_data (np.ndarray, optional): Tablica danych obrazu do zapisania. Jeśli podana, nadpisuje color/shape.

        Returns:
            str: Pełna ścieżka do utworzonego pliku obrazu.
        """
        if arr_data is not None:
            image_array = arr_data
        elif color is not None:
            image_array = np.full(shape, color, dtype=np.uint8)
        else:
            image_array = np.random.randint(0, 256, shape, dtype=np.uint8)
        
        filepath = os.path.join(self.test_dir, filename)
        
        # Zapisz obraz za pomocą OpenCV
        cv2.imwrite(filepath, image_array)
        
        return filepath