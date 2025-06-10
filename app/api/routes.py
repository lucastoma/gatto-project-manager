from flask import Blueprint, request, jsonify
import os
from app.core.file_handler import save_temp_file
from app.core.file_handler import get_result_path
from app.core.development_logger import get_logger
from app.algorithms import get_algorithm
from typing import Any

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
    params: dict[str, Any] = {}
    if algorithm_id == 'algorithm_01_palette':
        params['num_colors'] = int(request.form.get('k', 16)) # k_colors, default 16
        params['distance_metric'] = request.form.get('distance_metric', 'weighted_rgb') # default weighted_rgb
        params['use_dithering'] = request.form.get('use_dithering', 'false').lower() == 'true'
        params['preserve_luminance'] = request.form.get('preserve_luminance', 'false').lower() == 'true'
        
        # Parametry dithering i extremes (istniejące)
        params['dithering_method'] = request.form.get('dithering_method', 'none')
        params['inject_extremes'] = request.form.get('inject_extremes', 'false').lower() == 'true'
        params['preserve_extremes'] = request.form.get('preserve_extremes', 'false').lower() == 'true'
        params['extremes_threshold'] = int(request.form.get('extremes_threshold', 10))
        
        # === NOWE PARAMETRY EDGE BLENDING ===
        params['edge_blur_enabled'] = request.form.get('enable_edge_blending', 'false').lower() == 'true'
        params['edge_detection_threshold'] = float(request.form.get('edge_detection_threshold', 25))
        params['edge_blur_radius'] = float(request.form.get('edge_blur_radius', 1.5))
        params['edge_blur_strength'] = float(request.form.get('edge_blur_strength', 0.3))
        # exclude_colors and palette_source_area will be handled in Phase 2b

    logger.info(f"Przetwarzanie przez algorytm: {algorithm_id} z parametrami: {params}")

    master_path = None
    target_path = None
    try:
        master_path = save_temp_file(master_file)
        target_path = save_temp_file(target_file)

        if not master_path or not target_path:
            raise RuntimeError("Nie udało się zapisać plików tymczasowych")

        algorithm = get_algorithm(algorithm_id)
        if algorithm_id == 'algorithm_01_palette':
            # Użyj unikalnej nazwy pliku tymczasowego target_path, aby wynik był poprawny
            output_filename = os.path.basename(target_path)
            result_file_path = get_result_path(output_filename)
            algorithm.process_images(master_path, target_path, output_path=result_file_path, **params)
        else:
            result_file_path = algorithm.process(master_path, target_path)

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

@app.route('/api/colormatch/preview', methods=['POST'])
def colormatch_preview_endpoint():
    """Endpoint API do generowania podglądu dopasowania kolorów."""
    if 'master_image' not in request.files or 'target_image' not in request.files:
        logger.error("Brak wymaganych plików 'master_image' i 'target_image' dla podglądu")
        return "error,Request musi zawierać 'master_image' i 'target_image'"

    master_file = request.files['master_image']
    target_file = request.files['target_image']
    
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

    params: dict[str, Any] = {'preview_mode': True}
    if algorithm_id == 'algorithm_01_palette':
        params['num_colors'] = int(request.form.get('k', 16))
        params['distance_metric'] = request.form.get('distance_metric', 'weighted_rgb')
        params['use_dithering'] = request.form.get('use_dithering', 'false').lower() == 'true'
        params['preserve_luminance'] = request.form.get('preserve_luminance', 'false').lower() == 'true'
        
        # Parametry dithering i extremes (istniejące)
        params['dithering_method'] = request.form.get('dithering_method', 'none')
        params['inject_extremes'] = request.form.get('inject_extremes', 'false').lower() == 'true'
        params['preserve_extremes'] = request.form.get('preserve_extremes', 'false').lower() == 'true'
        params['extremes_threshold'] = int(request.form.get('extremes_threshold', 10))
        
        # === NOWE PARAMETRY EDGE BLENDING ===
        params['edge_blur_enabled'] = request.form.get('enable_edge_blending', 'false').lower() == 'true'
        params['edge_detection_threshold'] = float(request.form.get('edge_detection_threshold', 25))
        params['edge_blur_radius'] = float(request.form.get('edge_blur_radius', 1.5))
        params['edge_blur_strength'] = float(request.form.get('edge_blur_strength', 0.3))
        # preview_thumbnail_size can be passed from JSX if needed, otherwise default from config

    logger.info(f"Przetwarzanie podglądu przez algorytm: {algorithm_id} z parametrami: {params}")

    master_path = None
    target_path = None
    try:
        master_path = save_temp_file(master_file)
        target_path = save_temp_file(target_file)

        if not master_path or not target_path:
            raise RuntimeError("Nie udało się zapisać plików tymczasowych dla podglądu")

        algorithm = get_algorithm(algorithm_id)
        if algorithm_id == 'algorithm_01_palette':
            # Użyj unikalnej nazwy pliku tymczasowego target_path, aby wynik był poprawny
            output_filename = os.path.basename(target_path)
            result_file_path = get_result_path(output_filename)
            algorithm.process_images(master_path, target_path, output_path=result_file_path, **params)
        else:
            result_file_path = algorithm.process(master_path, target_path)

        result_filename = os.path.basename(result_file_path)
        logger.success(f"Podgląd dopasowywania kolorów zakończony: {result_filename}")
        return f"success,preview,{result_filename}"

    except Exception as e:
        logger.error(f"Podgląd dopasowywania kolorów nie powiódł się: {str(e)}", exc_info=True)
        return f"error,{str(e)}"
    finally:
        # Clean up temporary files created by save_temp_file
        if 'master_path' in locals() and master_path is not None and os.path.exists(master_path):
            os.remove(master_path)
        if 'target_path' in locals() and target_path is not None and os.path.exists(target_path):
            os.remove(target_path)


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
        logger.error(f"Analiza palety nie powiodła się: {str(e)}", exc_info=True)
        return f"error,{str(e)}"
