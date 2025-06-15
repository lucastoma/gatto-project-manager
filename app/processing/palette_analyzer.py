import cv2
import numpy as np
from sklearn.cluster import KMeans

# Placeholder for palette analyzer logic

def analyze_palette(image_path, k=8):
    # ...existing code from processing.py...
    try:
        # 1. Wczytaj obraz za pomocą OpenCV (obsługuje PNG, TIFF, JPEG)
        print(f"Wczytywanie obrazu: {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Nie można wczytać obrazu.")

        # 2. Przekonwertuj obraz z BGR na RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. Zmień rozmiar obrazu dla wydajności (do szerokości 500px, zachowując proporcje)
        height, width = image_rgb.shape[:2]
        if width > 500:
            new_width = 500
            new_height = int(height * (new_width / width))
            image_rgb = cv2.resize(image_rgb, (new_width, new_height))

        # 4. Przekształć dane obrazu na listę pikseli (wymaganą przez KMeans)
        pixels = image_rgb.reshape((-1, 3))

        # 5. Użyj K-Means do znalezienia klastrów
        print(f"Tworzenie palety z {k} kolorów...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)

        # 6. Wyciągnij środki klastrów
        palette = kmeans.cluster_centers_

        # 7. Przekonwertuj wartości kolorów na liczby całkowite (0-255)
        palette_int = palette.astype('uint8')

        # 8. Zwróć listę list z kolorami RGB
        return palette_int.tolist()
    except Exception as e:
        print(f"Błąd podczas analizy palety: {e}")
        return []
