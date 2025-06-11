#!/usr/bin/env python3
"""
Tworzenie prostych obrazów testowych dla webview
"""

from PIL import Image, ImageDraw
import numpy as np

def create_master_with_red():
    """Tworzy obraz master z czerwonym tłem"""
    img = Image.new('RGB', (100, 100), (255, 0, 0))  # Czerwony
    img.save('d:/projects/gatto-ps-ai/test_master_red.png')
    print("Utworzono test_master_red.png")

def create_target_with_blue():
    """Tworzy obraz target z niebieskim tłem"""
    img = Image.new('RGB', (100, 100), (0, 0, 255))  # Niebieski
    img.save('d:/projects/gatto-ps-ai/test_target_blue.png')
    print("Utworzono test_target_blue.png")

if __name__ == "__main__":
    create_master_with_red()
    create_target_with_blue()
    print("Obrazy testowe utworzone. Użyj ich w webview do testowania Color Focus.")
