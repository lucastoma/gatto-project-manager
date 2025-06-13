"""
LAB Color Space Transfer Algorithm
ID: algorithm_05_lab_transfer
API Number: 5

Zaawansowany algorytm transferu kolorów w przestrzeni LAB (CIELAB).
Wykorzystuje percepcyjnie jednolitą przestrzeń kolorów dla dokładniejszego
dopasowania kolorów między obrazami.
"""

from .algorithm import create_lab_transfer_algorithm

__all__ = ['create_lab_transfer_algorithm']
