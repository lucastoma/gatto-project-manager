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

# Import core services
try:
    from ..core.image_processor import ImageProcessor
    from ..algorithms.algorithm_01_palette.main import process_image as process_palette
    from ..api.utils import validate_image, create_response
except ImportError as e:
    print(f"Warning: Could not import core services: {e}")
    # Fallback imports or mock functions can be added here
    ImageProcessor = None
    process_palette = None

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
    
    # Log to Flask logger
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
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'webview_version': '1.0.0',
        'services': {
            'image_processor': ImageProcessor is not None,
            'algorithm_01': process_palette is not None
        }
    })

@webview_bp.route('/api/process', methods=['POST'])
def process_algorithm():
    """Process algorithm with uploaded image and parameters."""
    try:
        # Check if file is present
        if 'image_file' not in request.files:
            log_activity('process_error', {'error': 'No file uploaded'}, 'error')
            return jsonify({
                'success': False,
                'error': 'Nie wybrano pliku do przetworzenia'
            }), 400
        
        file = request.files['image_file']
        if file.filename == '':
            log_activity('process_error', {'error': 'Empty filename'}, 'error')
            return jsonify({
                'success': False,
                'error': 'Nie wybrano pliku do przetworzenia'
            }), 400
        
        # Validate file
        if not allowed_file(file.filename):
            log_activity('process_error', {'error': 'Invalid file type', 'filename': file.filename}, 'error')
            return jsonify({
                'success': False,
                'error': f'Nieprawidłowy typ pliku. Dozwolone: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Get algorithm type
        algorithm = request.form.get('algorithm', 'algorithm_01')
        
        # Get parameters
        params = {
            'num_colors': int(request.form.get('num_colors', 5)),
            'method': request.form.get('method', 'kmeans'),
            'quality': int(request.form.get('quality', 5)),
            'include_metadata': request.form.get('include_metadata') == 'on'
        }
        
        log_activity('process_start', {
            'algorithm': algorithm,
            'filename': file.filename,
            'params': params
        })
        
        # Ensure upload folder exists
        ensure_upload_folder()
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f"{timestamp}_{filename}"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        file.save(temp_path)
        
        try:
            # Process based on algorithm type
            if algorithm == 'algorithm_01':
                result = process_algorithm_01(temp_path, params)
            else:
                raise ValueError(f"Nieznany algorytm: {algorithm}")
            
            log_activity('process_success', {
                'algorithm': algorithm,
                'filename': filename,
                'result_colors': len(result.get('palette', []))
            })
            
            return jsonify({
                'success': True,
                'result': result,
                'algorithm': algorithm,
                'timestamp': datetime.now().isoformat()
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except RequestEntityTooLarge:
        log_activity('process_error', {'error': 'File too large'}, 'error')
        return jsonify({
            'success': False,
            'error': f'Plik zbyt duży. Maksymalny rozmiar: {MAX_FILE_SIZE // (1024*1024)}MB'
        }), 413
    
    except ValueError as e:
        log_activity('process_error', {'error': str(e)}, 'error')
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    
    except Exception as e:
        log_activity('process_error', {'error': str(e)}, 'error')
        current_app.logger.error(f"WebView processing error: {e}")
        return jsonify({
            'success': False,
            'error': 'Wystąpił błąd podczas przetwarzania obrazu'
        }), 500

def process_algorithm_01(image_path, params):
    """Process Algorithm 01 - Palette extraction."""
    try:
        # Check if algorithm is available
        if process_palette is None:
            # Fallback implementation for testing
            return create_mock_palette_result(params)
        
        # Call the actual algorithm
        result = process_palette(
            image_path=image_path,
            num_colors=params['num_colors'],
            method=params['method'],
            quality=params['quality']
        )
        
        # Add metadata if requested
        if params['include_metadata']:
            result['metadata'] = get_image_metadata(image_path)
        
        # Ensure result format consistency
        if 'palette' not in result and 'colors' in result:
            result['palette'] = result['colors']
        
        return result
        
    except Exception as e:
        current_app.logger.error(f"Algorithm 01 processing error: {e}")
        raise ValueError(f"Błąd przetwarzania algorytmu: {str(e)}")

def create_mock_palette_result(params):
    """Create mock result for testing when algorithm is not available."""
    import random
    
    # Generate mock colors
    colors = []
    for i in range(params['num_colors']):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        
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
        'processing_time': random.uniform(0.5, 2.0),
        'mock': True  # Indicate this is mock data
    }

def rgb_to_hsl(r, g, b):
    """Convert RGB to HSL."""
    r, g, b = r/255.0, g/255.0, b/255.0
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

def get_image_metadata(image_path):
    """Extract basic image metadata."""
    try:
        from PIL import Image
        
        with Image.open(image_path) as img:
            metadata = {
                'filename': os.path.basename(image_path),
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height,
                'file_size': os.path.getsize(image_path)
            }
            
            # Add EXIF data if available
            if hasattr(img, '_getexif') and img._getexif():
                metadata['exif'] = dict(img._getexif())
            
            return metadata
            
    except Exception as e:
        current_app.logger.warning(f"Could not extract metadata: {e}")
        return {
            'filename': os.path.basename(image_path),
            'file_size': os.path.getsize(image_path),
            'error': 'Could not extract detailed metadata'
        }

@webview_bp.route('/api/algorithms')
def list_algorithms():
    """List available algorithms."""
    algorithms = [
        {
            'id': 'algorithm_01',
            'name': 'Palette Extraction',
            'description': 'Ekstrakcja palety kolorów z obrazu',
            'status': 'available' if process_palette else 'mock',
            'parameters': {
                'num_colors': {'type': 'int', 'min': 1, 'max': 20, 'default': 5},
                'method': {'type': 'select', 'options': ['kmeans', 'median_cut'], 'default': 'kmeans'},
                'quality': {'type': 'int', 'min': 1, 'max': 10, 'default': 5},
                'include_metadata': {'type': 'bool', 'default': True}
            }
        }
    ]
    
    return jsonify({
        'algorithms': algorithms,
        'count': len(algorithms)
    })

@webview_bp.route('/api/logs')
def get_logs():
    """Get recent WebView activity logs."""
    # This is a simplified implementation
    # In a real application, you might want to store logs in a database or file
    return jsonify({
        'logs': [
            {
                'timestamp': datetime.now().isoformat(),
                'level': 'info',
                'message': 'WebView system operational'
            }
        ]
    })

@webview_bp.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    log_activity('error', {'error': 'File too large'}, 'error')
    return jsonify({
        'success': False,
        'error': f'Plik zbyt duży. Maksymalny rozmiar: {MAX_FILE_SIZE // (1024*1024)}MB'
    }), 413

@webview_bp.errorhandler(404)
def not_found(e):
    """Handle 404 errors in WebView."""
    log_activity('error', {'error': '404 Not Found', 'path': request.path}, 'warning')
    return render_template('404.html'), 404

@webview_bp.errorhandler(500)
def internal_error(e):
    """Handle 500 errors in WebView."""
    log_activity('error', {'error': '500 Internal Server Error'}, 'error')
    return render_template('500.html'), 500

# Context processors for templates
@webview_bp.context_processor
def inject_webview_context():
    """Inject common context variables into templates."""
    return {
        'webview_version': '1.0.0',
        'current_time': datetime.now(),
        'max_file_size_mb': MAX_FILE_SIZE // (1024 * 1024),
        'allowed_extensions': list(ALLOWED_EXTENSIONS)
    }

# Template filters
@webview_bp.app_template_filter('filesize')
def filesize_filter(size_bytes):
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

@webview_bp.app_template_filter('datetime')
def datetime_filter(dt, format='%Y-%m-%d %H:%M:%S'):
    """Format datetime object."""
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except:
            return dt
    
    return dt.strftime(format)