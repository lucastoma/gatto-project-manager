"""WebView Routes

Flask routes for the WebView interface.
Provides web-based testing and debugging for algorithms.
"""

import os
import json
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# === ZMIANA: WPROWADZAMY JAWNĄ FLAGĘ DO KONTROLOWANIA DANYCH TESTOWYCH ===
# Ustaw na 'False', gdy dodasz brakujące moduły 'core' i 'algorithms'
USE_MOCK_DATA = False

# === ZMIANA: Komentujemy importy, które powodują błąd ===
# Zostaną one zastąpione przez logikę opartą na fladze USE_MOCK_DATA
if not USE_MOCK_DATA:
    try:
        # Zamiast ImageProcessor, importujemy bezpośrednio algorytm
        from ..algorithms.algorithm_01_palette.algorithm import PaletteMappingAlgorithm
        # Poniższe importy mogą być potrzebne, jeśli inne części kodu ich używają
        # from ..core.image_processor import ImageProcessor 
        # from ..api.utils import validate_image, create_response
    except ImportError as e:
        raise ImportError(f"Nie udało się zaimportować modułów. Upewnij się, że istnieją. Błąd: {e}")
else:
    # Definiujemy puste zmienne, gdy używamy danych testowych
    PaletteMappingAlgorithm = None
    # ImageProcessor = None # Jeśli nie jest używany gdzie indziej, można usunąć
    # process_palette = None # Zastąpione przez PaletteMappingAlgorithm


# Create Blueprint
webview_bp = Blueprint('webview', __name__, 
                      template_folder='templates',
                      static_folder='static',
                      url_prefix='/webview')

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'temp_uploads'

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_upload_folder():
    """Ensure upload folder exists."""
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

def log_activity(action, details=None, level='info'):
    """Log WebView activity."""
    timestamp = datetime.now().isoformat()
    log_entry = {
        'timestamp': timestamp,
        'action': action,
        'details': details or {},
        'level': level
    }
    
    if hasattr(current_app, 'logger'):
        if level == 'error':
            current_app.logger.error(f"WebView: {action} - {details}")
        elif level == 'warning':
            current_app.logger.warning(f"WebView: {action} - {details}")
        else:
            current_app.logger.info(f"WebView: {action} - {details}")
    
    return log_entry

@webview_bp.route('/')
def index():
    """WebView main page."""
    log_activity('page_view', {'page': 'index'})
    return render_template('index.html')

@webview_bp.route('/algorithm_01')
def algorithm_01():
    """Algorithm 01 - Palette testing page."""
    log_activity('page_view', {'page': 'algorithm_01'})
    return render_template('algorithm_01.html')

@webview_bp.route('/api/health')
def health_check():
    """Health check endpoint for WebView."""
    services_available = not USE_MOCK_DATA
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'webview_version': '1.0.0',
        'mode': 'MOCK_DATA' if USE_MOCK_DATA else 'LIVE',
        'services': {
            'image_processor': services_available and (ImageProcessor is not None),
            'algorithm_01': services_available and (process_palette is not None)
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
        
        ensure_upload_folder()
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f"{timestamp}_{filename}"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        file.save(temp_path)
        
        try:
            if algorithm == 'algorithm_01':
                result = process_algorithm_01(temp_path, params)
            else:
                raise ValueError(f"Nieznany algorytm: {algorithm}")
            
            log_activity('process_success', {'algorithm': algorithm, 'filename': filename})
            return jsonify({'success': True, 'result': result, 'algorithm': algorithm, 'timestamp': datetime.now().isoformat()})
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except RequestEntityTooLarge:
        log_activity('process_error', {'error': 'File too large'}, 'error')
        return jsonify({'success': False, 'error': f'Plik zbyt duży. Maksymalny rozmiar: {MAX_FILE_SIZE // (1024*1024)}MB'}), 413
    
    except Exception as e:
        log_activity('process_error', {'error': str(e)}, 'error')
        current_app.logger.error(f"WebView processing error: {e}")
        return jsonify({'success': False, 'error': 'Wystąpił błąd podczas przetwarzania obrazu'}), 500

# === ZMIANA: Logika funkcji process_algorithm_01 jest teraz kontrolowana przez flagę ===
def process_algorithm_01(image_path, params):
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
            # Zakładamy, że rgb_to_hsl jest zdefiniowane gdzieś globalnie lub w utils
            # Jeśli nie, trzeba będzie je dodać lub zaimportować
            try:
                hsl_color = rgb_to_hsl(r, g, b) # Załóżmy, że ta funkcja istnieje
            except NameError: # Jeśli rgb_to_hsl nie jest zdefiniowane
                hsl_color = [0,0,0] # Placeholder
                if hasattr(current_app, 'logger'):
                    current_app.logger.warning("rgb_to_hsl function not found, using placeholder HSL values.")
                else:
                    print("Warning: rgb_to_hsl function not found, using placeholder HSL values.")
            
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

# Funkcja pomocnicza do konwersji RGB na HSL (jeśli nie istnieje globalnie)
# Można ją przenieść do app/webview/utils/color_converter.py lub podobnego miejsca
def rgb_to_hsl(r, g, b):
    r /= 255.0
    g /= 255.0
    b /= 255.0
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    h, s, l = 0, 0, (max_val + min_val) / 2

    if max_val == min_val:
        h = s = 0  # achromatic
    else:
        d = max_val - min_val
        s = d / (2 - max_val - min_val) if l > 0.5 else d / (max_val + min_val)
        if max_val == r:
            h = (g - b) / d + (6 if g < b else 0)
        elif max_val == g:
            h = (b - r) / d + 2
        elif max_val == b:
            h = (r - g) / d + 4
        h /= 6
    return [round(h * 360), round(s * 100), round(l * 100)]

def create_mock_palette_result(params, image_path):
    """Create mock result for testing. Ulepszona wersja, aby dane wyglądały bardziej sensownie."""
    import random
    from PIL import Image

    # Spróbujmy wygenerować kolory, które mają sens w kontekście obrazka
    try:
        with Image.open(image_path) as img:
            img.thumbnail((100, 100)) # Zmniejsz obrazek do szybkiej analizy
            img_colors = img.getcolors(img.size[0] * img.size[1])
            # Posortuj kolory wg występowania i weź najczęstsze
            dominant_colors = sorted(img_colors, key=lambda x: x[0], reverse=True)
            base_colors = [c[1] for c in dominant_colors[:params['num_colors']]]
    except:
        # Jeśli się nie uda, generuj losowe, jak wcześniej
        base_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(params['num_colors'])]

    colors = []
    for r, g, b in base_colors:
        # Dodajmy lekką losowość, aby nie były to zawsze te same kolory z getcolors
        r = min(255, max(0, r + random.randint(-10, 10)))
        g = min(255, max(0, g + random.randint(-10, 10)))
        b = min(255, max(0, b + random.randint(-10, 10)))
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        colors.append({
            'hex': hex_color,
            'rgb': [r, g, b],
            'hsl': rgb_to_hsl(r, g, b),
            'percentage': random.uniform(5, 25)
        })
    
    return {
        'palette': colors,
        'algorithm': 'algorithm_01',
        'method': params['method'],
        'num_colors': params['num_colors'],
        'quality': params['quality'],
        'processing_time': random.uniform(0.1, 0.5),
        'mock': True
    }

def rgb_to_hsl(r, g, b):
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

@webview_bp.route('/api/algorithms')
def list_algorithms():
    """List available algorithms."""
    algorithms = [
        {
            'id': 'algorithm_01',
            'name': 'Palette Extraction',
            'description': 'Ekstrakcja palety kolorów z obrazu',
            'status': 'mock' if USE_MOCK_DATA else 'available',
            'parameters': {
                'num_colors': {'type': 'int', 'min': 1, 'max': 20, 'default': 5},
                'method': {'type': 'select', 'options': ['kmeans', 'median_cut'], 'default': 'kmeans'},
                'quality': {'type': 'int', 'min': 1, 'max': 10, 'default': 5},
                'include_metadata': {'type': 'bool', 'default': True}
            }
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