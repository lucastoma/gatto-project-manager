#!/bin/bash
cd "$(dirname "$0")"
echo
echo "URUCHAMIANIE SELEKTORA KONFIGURACJI..."
echo
python3 config-selector.py
echo
echo "Nacisnij dowolny klawisz aby zamknac..."
read -n1 -r -p "" key
