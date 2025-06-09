import shutil
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from app.core.file_handler import get_result_path

def simple_palette_mapping(master_path, target_path, k_colors=8):
    """POZIOM 1: Proste mapowanie palety w RGB space (max 30 linii)"""
    # Wczytaj obrazy
    master = cv2.imread(master_path)
    target = cv2.imread(target_path)
    
    # Reshape do 2D dla K-means
    master_pixels = master.reshape(-1, 3).astype(np.float32)
    target_pixels = target.reshape(-1, 3).astype(np.float32)
    
    # K-means na master image
    kmeans_master = KMeans(n_clusters=k_colors, random_state=42, n_init=10)
    kmeans_master.fit(master_pixels)
    master_colors = kmeans_master.cluster_centers_
    
    # K-means na target image
    kmeans_target = KMeans(n_clusters=k_colors, random_state=42, n_init=10)
    target_labels = kmeans_target.fit_predict(target_pixels)
    target_colors = kmeans_target.cluster_centers_
    
    # Proste mapowanie: znajdź najbliższy kolor z master dla każdego z target
    mapped_pixels = np.zeros_like(target_pixels)
    for i, target_color in enumerate(target_colors):
        # Znajdź najbliższy kolor w master palette
        distances = np.sum((master_colors - target_color) ** 2, axis=1)
        closest_idx = np.argmin(distances)
        mapped_pixels[target_labels == i] = master_colors[closest_idx]
    
    # Reshape z powrotem do obrazu
    result = mapped_pixels.reshape(target.shape).astype(np.uint8)
    
    # Zapisz wynik
    result_path = get_result_path(os.path.basename(target_path))
    cv2.imwrite(result_path, result)
    return result_path

def basic_statistical_transfer(master_path, target_path):
    """POZIOM 1: Podstawowy transfer statystyczny w LAB (max 30 linii)"""
    # Wczytaj obrazy
    master = cv2.imread(master_path)
    target = cv2.imread(target_path)
    
    # Konwersja do LAB
    master_lab = cv2.cvtColor(master, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Oblicz statystyki dla każdego kanału
    result_lab = target_lab.copy()
    for i in range(3):  # L, a, b channels
        master_mean = np.mean(master_lab[:, :, i])
        master_std = np.std(master_lab[:, :, i])
        target_mean = np.mean(target_lab[:, :, i])
        target_std = np.std(target_lab[:, :, i])
        
        # Normalizuj i przeskaluj
        if target_std > 0:
            result_lab[:, :, i] = (target_lab[:, :, i] - target_mean) * (master_std / target_std) + master_mean
    
    # Ogranicz wartości do prawidłowego zakresu LAB
    result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 100)  # L: 0-100
    result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], -127, 127)  # a: -127 to 127
    result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], -127, 127)  # b: -127 to 127
    
    # Konwersja z powrotem do BGR
    result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # Zapisz wynik
    result_path = get_result_path(os.path.basename(target_path))
    cv2.imwrite(result_path, result)
    return result_path

def simple_histogram_matching(master_path, target_path):
    """POZIOM 1: Proste dopasowanie histogramu tylko dla luminancji (max 30 linii)"""
    # Wczytaj obrazy
    master = cv2.imread(master_path)
    target = cv2.imread(target_path)
    
    # Konwersja do LAB (używamy tylko kanał L)
    master_lab = cv2.cvtColor(master, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    
    # Wyciągnij kanał luminancji (L)
    master_l = master_lab[:, :, 0]
    target_l = target_lab[:, :, 0]
    
    # Oblicz histogramy
    master_hist, _ = np.histogram(master_l.flatten(), 256, [0, 256])
    target_hist, _ = np.histogram(target_l.flatten(), 256, [0, 256])
    
    # Oblicz CDF (Cumulative Distribution Function)
    master_cdf = master_hist.cumsum()
    target_cdf = target_hist.cumsum()
    
    # Normalizuj CDF
    master_cdf = master_cdf / master_cdf[-1]
    target_cdf = target_cdf / target_cdf[-1]
    
    # Stwórz lookup table
    lookup_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        # Znajdź najbliższą wartość w master CDF
        closest_idx = np.argmin(np.abs(master_cdf - target_cdf[i]))
        lookup_table[i] = closest_idx
    
    # Zastosuj lookup table tylko do kanału L
    result_lab = target_lab.copy()
    result_lab[:, :, 0] = lookup_table[target_l]
    
    # Konwersja z powrotem do BGR
    result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    
    # Zapisz wynik
    result_path = get_result_path(os.path.basename(target_path))
    cv2.imwrite(result_path, result)
    return result_path

# Backward compatibility
def palette_mapping_method1(master_path, target_path, k_colors):
    return simple_palette_mapping(master_path, target_path, k_colors)

def run_color_matching(master_path, target_path, k_colors):
    return simple_palette_mapping(master_path, target_path, k_colors)
