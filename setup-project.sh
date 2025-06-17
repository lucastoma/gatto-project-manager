#!/bin/bash

# Ustawienie, aby skrypt przerwał działanie w przypadku błędu
set -e

# ==============================================================================
# GŁÓWNY SKRYPT KONFIGURACYJNY PROJEKTU DJANGO
# Uruchom w folderze 'gatto-project-manager'
# ==============================================================================

# Nazwa folderu ze starymi plikami
OLD_PROJECT_DIR=".doc-gen"

# Kolory dla lepszej czytelności
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Sprawdzanie, czy istnieje folder źródłowy '${OLD_PROJECT_DIR}'...${NC}"
if [ ! -d "$OLD_PROJECT_DIR" ]; then
    echo "BŁĄD: Nie znaleziono folderu '${OLD_PROJECT_DIR}'. Uruchom ten skrypt w folderze 'gatto-project-manager'."
    exit 1
fi
echo -e "${GREEN}Znaleziono. Zaczynam budowę projektu.${NC}"
echo ""

# --- Krok 1: Wirtualne środowisko i zależności ---
echo -e "${YELLOW}--- Krok 1: Tworzenie wirtualnego środowiska 'venv' i instalacja pakietów...${NC}"
python3 -m venv venv
source venv/bin/activate
pip install django python-decouple
pip freeze > requirements.txt
echo -e "${GREEN}Gotowe. Środowisko aktywne, zależności zapisane w requirements.txt.${NC}"
echo ""

# --- Krok 2: Struktura projektu Django ---
echo -e "${YELLOW}--- Krok 2: Tworzenie bazowej struktury projektu Django...${NC}"
django-admin startproject config .
mkdir -p apps/ static/ templates/ media/ configs/ output/ tests/
(cd apps && ../venv/bin/python ../manage.py startapp collector)
echo -e "${GREEN}Gotowe. Stworzono foldery i aplikację 'collector'.${NC}"
echo ""

# --- Krok 3: Migracja istniejących plików ---
echo -e "${YELLOW}--- Krok 3: Przenoszenie i porządkowanie Twoich plików z .doc-gen...${NC}"

# Logika biznesowa
echo "  -> Przenoszenie skryptów .py do 'apps/collector/'..."
if [ -f "$OLD_PROJECT_DIR"/advanced_context_collector.py ]; then
    mv "$OLD_PROJECT_DIR"/advanced_context_collector.py apps/collector/
else
    echo "    advanced_context_collector.py już przeniesiony lub nie istnieje"
fi
if [ -f "$OLD_PROJECT_DIR"/config-selector.py ]; then
    mv "$OLD_PROJECT_DIR"/config-selector.py apps/collector/
else
    echo "    config-selector.py już przeniesiony lub nie istnieje"
fi
if [ -f "$OLD_PROJECT_DIR"/quick_context_collector.py ]; then
    mv "$OLD_PROJECT_DIR"/quick_context_collector.py apps/collector/
else
    echo "    quick_context_collector.py już przeniesiony lub nie istnieje"
fi

# Pliki konfiguracyjne
echo "  -> Konsolidacja plików konfiguracyjnych w 'configs/'..."
if [ -d "$OLD_PROJECT_DIR"/config ] && [ "$(ls -A "$OLD_PROJECT_DIR"/config/)" ]; then
    mv "$OLD_PROJECT_DIR"/config/* configs/
else
    echo "    Folder config/ pusty lub już przeniesiony"
fi
# Przenoszenie plików z config-lists (włącznie z ukrytymi plikami zaczynającymi się od kropki)
if [ -d "$OLD_PROJECT_DIR"/config-lists ] && [ "$(ls -A "$OLD_PROJECT_DIR"/config-lists/)" ]; then
    mv "$OLD_PROJECT_DIR"/config-lists/.* configs/ 2>/dev/null || true
    mv "$OLD_PROJECT_DIR"/config-lists/* configs/ 2>/dev/null || true
else
    echo "    Folder config-lists/ pusty lub już przeniesiony"
fi

# Pliki wyjściowe
echo "  -> Przenoszenie wygenerowanych plików do 'output/'..."
if [ -d "$OLD_PROJECT_DIR"/export ] && [ "$(ls -A "$OLD_PROJECT_DIR"/export/)" ]; then
    mv "$OLD_PROJECT_DIR"/export/* output/
else
    echo "    Folder export/ pusty lub już przeniesiony"
fi
if [ -f "$OLD_PROJECT_DIR"/repomix-output.xml ]; then
    mv "$OLD_PROJECT_DIR"/repomix-output.xml output/
else
    echo "    repomix-output.xml już przeniesiony lub nie istnieje"
fi
if [ -f "$OLD_PROJECT_DIR"/repomix-stats.log ]; then
    mv "$OLD_PROJECT_DIR"/repomix-stats.log output/
else
    echo "    repomix-stats.log już przeniesiony lub nie istnieje"
fi

# README
echo "  -> Kopiowanie README.md..."
if [ -f "$OLD_PROJECT_DIR"/README.md ]; then
    cp "$OLD_PROJECT_DIR"/README.md .
else
    echo "    README.md już skopiowany lub nie istnieje"
fi

echo -e "${GREEN}Gotowe. Pliki zostały przeniesione do nowej struktury.${NC}"
echo ""

# --- Krok 4: Finalizacja konfiguracji ---
echo -e "${YELLOW}--- Krok 4: Tworzenie .env, .gitignore i miejsca na komendę Django...${NC}"
touch .env

echo "# Środowisko wirtualne
venv/

# Pliki Pythona
__pycache__/
*.pyc

# Baza danych
db.sqlite3

# Sekrety
.env

# Pliki wgrywane i generowane
media/
output/

# Ustawienia IDE
.vscode/
.idea/
" > .gitignore
echo "  -> Stworzono i wypełniono .gitignore."

# Tworzenie miejsca na komendę, która zastąpi run-*.sh
mkdir -p apps/collector/management/commands/
touch apps/collector/management/commands/run_collector.py
echo "  -> Stworzono placeholder dla komendy Django: apps/collector/management/commands/run_collector.py"

echo -e "${GREEN}Gotowe.${NC}"
echo ""

echo -e "${YELLOW}======================================================"
echo -e "      PROJEKT DJANGO ZOSTAŁ POMYŚLNIE ZBUDOWANY!      "
echo -e "======================================================${NC}"
echo ""
echo -e "Folder '${OLD_PROJECT_DIR}' pozostał jako kopia zapasowa. Możesz go usunąć po weryfikacji."
echo ""
echo -e "Pamiętaj, aby teraz pracować w nowym środowisku: ${GREEN}source venv/bin/activate${NC}"