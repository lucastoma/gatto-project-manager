"""WebView Routes

Flask routes for the WebView interface.
Provides web-based testing and debugging for algorithms.
"""

import os
import json
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# --- UWAGA: Upewniamy się, że używamy prawdziwego algorytmu ---

# Zakładamy, że reszta aplikacji jest poprawnie skonfigurowana.

try:
    from ..algorithms.algorithm_01_palette.algorithm import PaletteMappingAlgorithm
except ImportError as e: # W przypadku błędu importu, rzucamy wyjątek, aby wyraźnie pokazać problem
    raise ImportError(f"CRITICAL: Failed to import PaletteMappingAlgorithm. Ensure the module exists and is correct. Error: {e}")

# Create Blueprint
webview_bp = Blueprint('webview', __name__,
                       template_folder='templates',
                       static_folder='static',
                       url_prefix='/webview')

# --- NOWA KONFIGURACJA ZGODNA Z PROŚBĄ ---

# Zwiększony limit rozmiaru pliku (np. do 100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024

# Dodane rozszerzenia TIF/TIFF i usunięte limity
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# Używamy podkatalogu w 'static', aby pliki były publicznie dostępne przez URL
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'results')

# Używamy osobnego folderu na tymczasowe pliki
UPLOADS_FOLDER = os.path.join(os.path.dirname(__file__), 'temp_uploads')

def allowed_file(filename):
    """Sprawdza, czy rozszerzenie pliku jest dozwolone."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_folders():
    """Upewnia się, że foldery na upload i wyniki istnieją."""
    os.makedirs(UPLOADS_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

def log_activity(action, details=None, level='info'):
    """Prosta funkcja do logowania aktywności WebView."""
    timestamp = datetime.now().isoformat()
    log_message = f"WebView: {action} - {json.dumps(details) if details else ''}"
    if hasattr(current_app, 'logger'):
        if level == 'error':
            current_app.logger.error(log_message)
        else:
            current_app.logger.info(log_message)
    else:
        print(f"[{level.upper()}] {log_message}")

# --- Istniejące trasy ---

@webview_bp.route('/')
def index():
    """WebView main page."""
    log_activity('page_view', {'page': 'index'})
    return render_template('index.html')

@webview_bp.route('/algorithm_01')
def algorithm_01_palette_extraction(): # Zmieniona nazwa funkcji i szablonu
    """Strona testowania ekstrakcji palety (istniejąca)."""
    log_activity('page_view', {'page': 'algorithm_01_palette_extraction'})
    return render_template('algorithm_01_palette_extraction.html') # Zmieniony szablon

# --- NOWA TRASA DO TESTOWANIA TRANSFERU PALETY ---

@webview_bp.route('/algorithm_01/transfer')
def algorithm_01_palette_transfer():
    """Strona testowania transferu palety (nowa)."""
    log_activity('page_view', {'page': 'algorithm_01_palette_transfer'})
    return render_template('algorithm_01_transfer.html')

@webview_bp.route('/results/<filename>')
def get_result_file(filename):
    """Serwuje przetworzony obraz z folderu wyników."""
    return send_from_directory(RESULTS_FOLDER, filename)


@webview_bp.route('/api/health')
def health_check():
    """Health check endpoint for WebView."""
    # Uproszczone, ponieważ USE_MOCK_DATA zostało usunięte
    palette_algorithm_available = False
    try:
        # Sprawdź, czy PaletteMappingAlgorithm jest dostępne
        if PaletteMappingAlgorithm:
            palette_algorithm_available = True
    except NameError:
        pass # PaletteMappingAlgorithm nie jest zdefiniowane
        
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'webview_version': '1.0.0',
        'mode': 'LIVE',
        'services': {
            'algorithm_01_palette': palette_algorithm_available
        }
    })

@webview_bp.route('/api/process', methods=['POST'])
def process_algorithm():
    """Process algorithm with uploaded image and parameters."""
    try:
        if 'image_file' not in request.files:
            log_activity('process_error', {'error': 'No file uploaded'}, 'error')
            return jsonify({'success': False, 'error': 'Nie wybrano pliku do przetworzenia'}), 400
        
        file = request.files['image_file']
        if file.filename == '' or not allowed_file(file.filename):
            log_activity('process_error', {'error': 'Invalid file'}, 'error')
            return jsonify({'success': False, 'error': f'Nieprawidłowy plik lub typ pliku. Dozwolone: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        algorithm = request.form.get('algorithm', 'algorithm_01')
        params = {
            'num_colors': int(request.form.get('num_colors', 5)),
            'method': request.form.get('method', 'kmeans'),
            'quality': int(request.form.get('quality', 5)),
            'include_metadata': request.form.get('include_metadata') == 'on'
        }
        
        log_activity('process_start', {'algorithm': algorithm, 'filename': file.filename, 'params': params})
        
        ensure_folders() # Zmieniono na ensure_folders
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f"{timestamp}_{filename}"
        temp_path = os.path.join(UPLOADS_FOLDER, temp_filename) # Zmieniono na UPLOADS_FOLDER
        file.save(temp_path)
        
        try:
            if algorithm == 'algorithm_01': # To jest dla ekstrakcji palety
                result = process_algorithm_01_extraction(temp_path, params) # Zmieniono nazwę funkcji
            else:
                raise ValueError(f"Nieznany algorytm: {algorithm}")
            
            log_activity('process_success', {'algorithm': algorithm, 'filename': filename})
            return jsonify({'success': True, 'result': result, 'algorithm': algorithm, 'timestamp': datetime.now().isoformat()})
            
        finally:
            # Nie usuwamy pliku od razu, jeśli chcemy go użyć np. w process_images
            # Rozważ logikę czyszczenia później lub jeśli plik nie jest już potrzebny
            if os.path.exists(temp_path) and algorithm == 'algorithm_01': # Usuń tylko jeśli to była ekstrakcja i nie transfer
                 os.remove(temp_path)
    
    except RequestEntityTooLarge:
        log_activity('process_error', {'error': 'File too large'}, 'error')
        return jsonify({'success': False, 'error': f'Plik zbyt duży. Maksymalny rozmiar: {MAX_FILE_SIZE // (1024*1024)}MB'}), 413
    
    except Exception as e:
        log_activity('process_error', {'error': str(e)}, 'error')
        current_app.logger.error(f"WebView processing error: {e}")
        return jsonify({'success': False, 'error': 'Wystąpił błąd podczas przetwarzania obrazu'}), 500

# Zmieniona nazwa funkcji, aby odróżnić od transferu
def process_algorithm_01_extraction(image_path, params):
    """Process Algorithm 01 - Palette extraction using the actual algorithm."""
    # Ta funkcja jest teraz wywoływana tylko, gdy USE_MOCK_DATA = False
    try:
        # 1. Inicjalizuj algorytm
        algorithm = PaletteMappingAlgorithm()

        # 2. Przekaż parametry z UI do konfiguracji algorytmu
        algorithm.config['quality'] = params.get('quality', 5)

        # 3. Wywołaj właściwą funkcję ekstrakcji palety
        palette_rgb = algorithm.extract_palette(
            image_path=image_path,
            num_colors=params['num_colors'],
            method=params['method']
        )

        # 4. Przetwórz wynik do formatu oczekiwanego przez UI (z HEX, HSL itp.)
        # Ta część jest opcjonalna, ale dobra dla spójności z mock data
        colors = []
        for r, g, b in palette_rgb:
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            hsl_color = global_rgb_to_hsl(r, g, b) # Używamy globalnej funkcji
            
            colors.append({
                'hex': hex_color,
                'rgb': [r, g, b],
                'hsl': hsl_color
                # Procentowy udział wymagałby dodatkowej logiki, więc pomijamy go dla uproszczenia
            })

        return {
            'palette': colors,
            'algorithm': 'algorithm_01',
            'method': params['method'],
            'num_colors': params['num_colors'],
            'quality': params['quality'],
            'mock': False  # Wyraźnie zaznaczamy, że to nie są dane testowe
        }

    except Exception as e:
        # Użyj loggera, jeśli jest dostępny
        if hasattr(current_app, 'logger'):
            current_app.logger.error(f"Algorithm 01 real processing error: {e}", exc_info=True)
        else:
            print(f"Algorithm 01 real processing error: {e}")
        raise ValueError(f"Błąd przetwarzania algorytmu: {str(e)}")

# global_rgb_to_hsl, aby uniknąć konfliktu nazw, jeśli lokalna była używana gdzie indziej
def global_rgb_to_hsl(r, g, b):
    """Convert RGB to HSL."""
    r, g, b = r/255.0, g/255.0, b/255.0
    max_val, min_val = max(r, g, b), min(r, g, b)
    h, s, l = 0, 0, (max_val + min_val) / 2
    if max_val != min_val:
        d = max_val - min_val
        s = d / (2 - max_val - min_val) if l > 0.5 else d / (max_val + min_val)
        if max_val == r: h = (g - b) / d + (6 if g < b else 0)
        elif max_val == g: h = (b - r) / d + 2
        else: h = (r - g) / d + 4
        h /= 6
    return [round(h * 360), round(s * 100), round(l * 100)]

def get_image_metadata(image_path):
    """Extract basic image metadata."""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return {'filename': os.path.basename(image_path), 'format': img.format, 'mode': img.mode, 'size': img.size, 'width': img.width, 'height': img.height, 'file_size': os.path.getsize(image_path)}
    except Exception as e:
        current_app.logger.warning(f"Could not extract metadata: {e}")
        return {'filename': os.path.basename(image_path), 'file_size': os.path.getsize(image_path), 'error': 'Could not extract detailed metadata'}

# Reszta pliku bez zmian (list_algorithms, get_logs, errorhandlers, etc.)
# ... (pozostała część pliku pozostaje bez zmian)

# --- NOWA TRASA API DO OBSŁUGI TRANSFERU PALETY ---

@webview_bp.route('/api/algorithm_01/transfer', methods=['POST'])
def handle_palette_transfer():
    """
    Przyjmuje obraz master, target i parametry, przetwarza je i zwraca
    ścieżkę do obrazu wynikowego.
    """
    ensure_folders()
    log_activity('transfer_request_start')

    try:
        # Sprawdzenie plików
        if 'master_image' not in request.files or 'target_image' not in request.files:
            log_activity('transfer_error', {'error': 'Missing master or target image'}, 'error')
            return jsonify({'success': False, 'error': 'Brak pliku master_image lub target_image'}), 400

        master_file = request.files['master_image']
        target_file = request.files['target_image']

        if master_file.filename == '' or target_file.filename == '':
            log_activity('transfer_error', {'error': 'Empty filename for master or target'}, 'error')
            return jsonify({'success': False, 'error': 'Nie wybrano plików'}), 400

        if not allowed_file(master_file.filename) or not allowed_file(target_file.filename):
            log_activity('transfer_error', {'error': f'Invalid file type. Master: {master_file.filename}, Target: {target_file.filename}'}, 'error')
            return jsonify({'success': False, 'error': f'Niedozwolony typ pliku. Dozwolone: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

        # Zapisanie plików tymczasowych
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        master_filename_secure = secure_filename(master_file.filename)
        target_filename_secure = secure_filename(target_file.filename)
        
        master_filename = f"{timestamp}_master_{master_filename_secure}"
        target_filename = f"{timestamp}_target_{target_filename_secure}"
        master_path = os.path.join(UPLOADS_FOLDER, master_filename)
        target_path = os.path.join(UPLOADS_FOLDER, target_filename)
        master_file.save(master_path)
        target_file.save(target_path)
        log_activity('files_saved', {'master': master_path, 'target': target_path})

        # Zebranie parametrów z formularza
        params = {
            'num_colors': int(request.form.get('num_colors', 16)),
            'dithering_method': request.form.get('dithering_method', 'none'),
            'inject_extremes': request.form.get('inject_extremes') == 'on',
            'preserve_extremes': request.form.get('preserve_extremes') == 'on',
            'extremes_threshold': int(request.form.get('extremes_threshold', 10)),
            'edge_blur_enabled': request.form.get('edge_blur_enabled') == 'on',
            'edge_detection_threshold': float(request.form.get('edge_detection_threshold', 25)),
            'edge_blur_radius': float(request.form.get('edge_blur_radius', 1.5)),
            'edge_blur_strength': float(request.form.get('edge_blur_strength', 0.3)),
            'quality': int(request.form.get('quality', 5)), # Parametr z p1.md
            'distance_metric': request.form.get('distance_metric', 'weighted_rgb') # Parametr z p1.md
        }
        log_activity('parameters_collected', params)

        # Przetwarzanie
        algorithm = PaletteMappingAlgorithm()
        # Używamy oryginalnej nazwy pliku docelowego dla wyniku, aby było bardziej czytelne
        output_filename = f"result_{timestamp}_{target_filename_secure}"
        output_path = os.path.join(RESULTS_FOLDER, output_filename)

        log_activity('processing_start', {'output_path': output_path, 'params': params})
        
        # Zakładamy, że metoda process_images istnieje w PaletteMappingAlgorithm
        # i przyjmuje te parametry.
        # To może wymagać aktualizacji pliku algorithm.py
        success = algorithm.process_images(
            master_path=master_path,
            target_path=target_path,
            output_path=output_path,
            **params
        )
        log_activity('processing_end', {'success': success})

        if not success:
            # Dodatkowe logowanie, jeśli algorytm zwrócił False
            log_activity('algorithm_processing_failed', {'master': master_path, 'target': target_path, 'params': params}, 'error')
            raise RuntimeError("Algorithm processing failed. Check server logs for details.")

        # Zwrócenie ścieżki do pliku wynikowego
        result_url = f"/webview/results/{output_filename}" # Poprawiony URL, aby pasował do definicji trasy
        log_activity('transfer_request_success', {'result_url': result_url})

        return jsonify({
            'success': True,
            'result_url': result_url,
            'message': 'Obraz przetworzony pomyślnie!'
        })

    except RequestEntityTooLarge:
        log_activity('transfer_error', {'error': 'File too large'}, 'error')
        return jsonify({'success': False, 'error': f'Plik zbyt duży. Maksymalny rozmiar: {MAX_FILE_SIZE // (1024*1024)}MB'}), 413
    except Exception as e:
        log_activity('transfer_error', {'error': str(e)}, 'error')
        if hasattr(current_app, 'logger'):
            current_app.logger.exception("An error occurred during palette transfer.")
        return jsonify({'success': False, 'error': f'Wystąpił wewnętrzny błąd serwera: {str(e)}'}), 500
    finally:
        # Czyszczenie plików tymczasowych po przetworzeniu
        if 'master_path' in locals() and os.path.exists(master_path):
            os.remove(master_path)
        if 'target_path' in locals() and os.path.exists(target_path):
            os.remove(target_path)
        log_activity('temp_files_cleaned', {'master_path': locals().get('master_path'), 'target_path': locals().get('target_path')})

@webview_bp.route('/api/algorithms')
def list_algorithms():
    """List available algorithms."""
    algorithms = [
        {
            'id': 'algorithm_01_extraction',
            'name': 'Palette Extraction',
            'description': 'Ekstrakcja palety kolorów z obrazu.',
            'status': 'available', # Usunięto zależność od USE_MOCK_DATA
            'parameters': {
                'num_colors': {'type': 'int', 'min': 1, 'max': 20, 'default': 5},
                'method': {'type': 'select', 'options': ['kmeans', 'median_cut'], 'default': 'kmeans'},
                'quality': {'type': 'int', 'min': 1, 'max': 10, 'default': 5},
                # 'include_metadata': {'type': 'bool', 'default': True} # Usunięto, jeśli nie jest używane
            },
            'endpoint': '/api/process' # Wskazuje na istniejący endpoint dla ekstrakcji
        },
        {
            'id': 'algorithm_01_transfer',
            'name': 'Palette Transfer',
            'description': 'Transfer palety kolorów między obrazami.',
            'status': 'available',
            'parameters': {
                'num_colors': {'type': 'int', 'min': 1, 'max': 64, 'default': 16},
                'dithering_method': {'type': 'select', 'options': ['none', 'floyd_steinberg'], 'default': 'none'},
                'inject_extremes': {'type': 'bool', 'default': False},
                'preserve_extremes': {'type': 'bool', 'default': False},
                'extremes_threshold': {'type': 'int', 'min':0, 'max': 100, 'default': 10},
                'edge_blur_enabled': {'type': 'bool', 'default': False},
                'edge_detection_threshold': {'type': 'float', 'min':0, 'max': 255, 'default': 25},
                'edge_blur_radius': {'type': 'float', 'min':0.1, 'max': 10, 'default': 1.5},
                'edge_blur_strength': {'type': 'float', 'min':0, 'max': 1, 'default': 0.3},
                'quality': {'type': 'int', 'min': 1, 'max': 10, 'default': 5},
                'distance_metric': {'type': 'select', 'options': ['rgb', 'lab', 'weighted_rgb'], 'default': 'weighted_rgb'}
            },
            'endpoint': '/api/algorithm_01/transfer'
        }
    ]
    return jsonify({'algorithms': algorithms, 'count': len(algorithms)})

@webview_bp.route('/api/logs')
def get_logs():
    return jsonify({'logs': [{'timestamp': datetime.now().isoformat(),'level': 'info','message': 'WebView system operational'}]})

@webview_bp.errorhandler(413)
def too_large(e):
    log_activity('error', {'error': 'File too large'}, 'error')
    return jsonify({'success': False, 'error': f'Plik zbyt duży. Maksymalny rozmiar: {MAX_FILE_SIZE // (1024*1024)}MB'}), 413

@webview_bp.errorhandler(404)
def not_found(e):
    log_activity('error', {'error': '404 Not Found', 'path': request.path}, 'warning')
    return render_template('404.html'), 404

@webview_bp.errorhandler(500)
def internal_error(e):
    log_activity('error', {'error': '500 Internal Server Error'}, 'error')
    return render_template('500.html'), 500

@webview_bp.context_processor
def inject_webview_context():
    return {'webview_version': '1.0.0','current_time': datetime.now(),'max_file_size_mb': MAX_FILE_SIZE // (1024 * 1024),'allowed_extensions': list(ALLOWED_EXTENSIONS)}

@webview_bp.app_template_filter('filesize')
def filesize_filter(size_bytes):
    if size_bytes == 0: return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"

@webview_bp.app_template_filter('datetime')
def datetime_filter(dt, format='%Y-%m-%d %H:%M:%S'):
    if isinstance(dt, str):
        try: dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except: return dt
    return dt.strftime(format)