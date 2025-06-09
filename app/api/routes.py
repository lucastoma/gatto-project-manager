from flask import Blueprint, request, jsonify
import os
from app.core.file_handler import save_temp_file
from app.core.development_logger import get_logger
from app.algorithms import get_algorithm

# Create Blueprint instead of Flask app
app = Blueprint('api', __name__)

# Initialize logger
logger = get_logger()

@app.route('/api/colormatch', methods=['POST'])
def colormatch_endpoint():
    """Endpoint API do dopasowywania kolorów - w pełni dynamiczny."""
    if 'master_image' not in request.files or 'target_image' not in request.files:
        logger.error("Brak wymaganych plików 'master_image' i 'target_image'")
        return "error,Request musi zawierać 'master_image' i 'target_image'"

    master_file = request.files['master_image']
    target_file = request.files['target_image']
    
    # Mapowanie 'method' na 'algorithm_id'
    method = request.form.get('method', default='1', type=str)
    algorithm_map = {
        '1': 'algorithm_01_palette',
        '2': 'algorithm_02_statistical',
        '3': 'algorithm_03_histogram'
    }
    algorithm_id = algorithm_map.get(method)
    
    if not algorithm_id:
        logger.error(f"Nieznana metoda: {method}")
        return f"error,Nieznana metoda: {method}. Dostępne: 1, 2, 3"

    # Przygotowanie parametrów
    params = {}
    if algorithm_id == 'algorithm_01_palette':
        params['k_colors'] = request.form.get('k', default=8, type=int)

    logger.info(f"Przetwarzanie przez algorytm: {algorithm_id} z parametrami: {params}")

    try:
        master_path = save_temp_file(master_file)
        target_path = save_temp_file(target_file)

        if not master_path or not target_path:
            raise RuntimeError("Nie udało się zapisać plików tymczasowych")

        # Dynamiczne pobranie i uruchomienie algorytmu
        algorithm = get_algorithm(algorithm_id)
        result_file_path = algorithm.process(master_path, target_path, **params)

        result_filename = os.path.basename(result_file_path)
        logger.success(f"Dopasowywanie kolorów zakończone: {result_filename}")
        # Zwracamy 'method{X}' dla kompatybilności z JSX
        return f"success,method{method},{result_filename}"

    except Exception as e:
        logger.error(f"Dopasowywanie kolorów nie powiodło się: {str(e)}", exc_info=True)
        return f"error,{str(e)}"
    finally:
        # Można dodać logikę czyszczenia plików tymczasowych, jeśli jest potrzebna
        pass

@app.route('/api/analyze_palette', methods=['POST'])
def analyze_palette_endpoint():
    """Endpoint API do analizy palety kolorów."""
    if 'source_image' not in request.files:
        return "error,Brak pliku source_image"
    file = request.files['source_image']
    k = request.form.get('k', default=8, type=int)
    from app.core.file_handler import save_temp_file
    from app.processing.palette_analyzer import analyze_palette
    try:
        temp_path = save_temp_file(file)
        palette = analyze_palette(temp_path, k)
        if not palette or len(palette) == 0:
            return "error,Brak kolorów lub błąd analizy"
        # Spłaszcz listę kolorów do CSV
        flat = [str(x) for color in palette for x in color]
        response = ["success", str(len(palette))] + flat
        return ",".join(response)
    except Exception as e:
        return f"error,{str(e)}"
