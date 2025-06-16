"""WebView Routes

Flask routes for the WebView interface.
Provides web-based testing and debugging for algorithms.
"""

import os
import json
from datetime import datetime
from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    current_app,
    send_from_directory,
)
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import time  # Dodano do pomiaru czasu przetwarzania

# --- UWAGA: Upewniamy się, że używamy prawdziwego algorytmu ---
# Zakładamy, że reszta aplikacji jest poprawnie skonfigurowana.
try:
    from ..algorithms.algorithm_01_palette.algorithm import PaletteMappingAlgorithm
    # --- Imports for Algorithm 05: LAB Color Transfer ---
    from ..algorithms.algorithm_05_lab_transfer.core import LABColorTransfer
    from ..algorithms.algorithm_05_lab_transfer.config import LABTransferConfig
    import numpy as np
    from PIL import Image
    # --- End Imports for Algorithm 05 ---
except ImportError as e:
    # W przypadku błędu importu, rzucamy wyjątek, aby wyraźnie pokazać problem
    raise ImportError(
        f"CRITICAL: Failed to import PaletteMappingAlgorithm. Ensure the module exists and is correct. Error: {e}"
    )

# Import GPU variant, może być niedostępny na systemach bez OpenCL
try:
    from ..algorithms.algorithm_01_palette import PaletteMappingAlgorithmGPU, OPENCL_AVAILABLE
except ImportError:
    PaletteMappingAlgorithmGPU = None
    OPENCL_AVAILABLE = False

# Create Blueprint
webview_bp = Blueprint(
    "webview",
    __name__,
    template_folder="templates",
    static_folder="static",
    url_prefix="/webview",
)

# --- KONFIGURACJA ---
MAX_FILE_SIZE = 100 * 1024 * 1024  # Zwiększony limit do 100MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff"}  # Dodano TIF/TIFF
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "static", "results")
UPLOADS_FOLDER = os.path.join(os.path.dirname(__file__), "temp_uploads")


# --- FUNKCJE POMOCNICZE ---


def allowed_file(filename):
    """Sprawdza, czy rozszerzenie pliku jest dozwolone."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_folders():
    """Upewnia się, że foldery na upload i wyniki istnieją."""
    os.makedirs(UPLOADS_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)


def log_activity(action, details=None, level="info"):
    """Prosta funkcja do logowania aktywności WebView."""
    timestamp = datetime.now().isoformat()
    log_message = f"WebView: {action} - {json.dumps(details) if details else ''}"
    if hasattr(current_app, "logger"):
        if level == "error":
            current_app.logger.error(log_message)
        else:
            current_app.logger.info(log_message)
    else:
        print(f"[{level.upper()}] {log_message}")


def rgb_to_hsl(r, g, b):
    """Konwertuje kolor z RGB na HSL, potrzebne dla starego panelu."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    max_val, min_val = max(r, g, b), min(r, g, b)
    h, s, l = 0, 0, (max_val + min_val) / 2
    if max_val != min_val:
        d = max_val - min_val
        s = d / (2 - max_val - min_val) if l > 0.5 else d / (max_val + min_val)
        if max_val == r:
            h = (g - b) / d + (6 if g < b else 0)
        elif max_val == g:
            h = (b - r) / d + 2
        else:
            h = (r - g) / d + 4
        h /= 6
    return [round(h * 360), round(s * 100), round(l * 100)]


# --- GŁÓWNE TRASY I ENDPOINTY ---


@webview_bp.route("/")
def index():
    """Główna strona WebView."""
    log_activity("page_view", {"page": "index", "template_vars": {"now": datetime.now().year}})
    return render_template("index.html", now=datetime.now())


@webview_bp.route("/algorithm_01")
def algorithm_01():
    """Strona testowania ekstrakcji palety (stary panel)."""
    log_activity("page_view", {"page": "algorithm_01_extraction"})
    return render_template("algorithm_01.html")


@webview_bp.route("/algorithm_01/transfer")
def algorithm_01_palette_transfer():
    """Strona testowania transferu palety (nowy panel)."""
    log_activity("page_view", {"page": "algorithm_01_palette_transfer"})
    return render_template("algorithm_01_transfer.html")


@webview_bp.route("/results/<filename>")
def get_result_file(filename):
    """Serwuje przetworzony obraz z folderu wyników."""
    return send_from_directory(RESULTS_FOLDER, filename)


# --- API ENDPOINTS ---


@webview_bp.route("/api/process", methods=["POST"])
def process_algorithm():
    """API dla starszego panelu ekstrakcji palety."""
    try:
        if "image_file" not in request.files:
            return jsonify({"success": False, "error": "Nie wybrano pliku"}), 400
        file = request.files["image_file"]
        if file.filename == "" or not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Nieprawidłowy plik"}), 400

        params = {
            "num_colors": int(request.form.get("num_colors", 8)),
            "method": request.form.get("method", "kmeans"),
            "quality": int(request.form.get("quality", 5)),
        }
        log_activity(
            "extraction_request", {"filename": file.filename, "params": params}
        )

        ensure_folders()
        temp_path = os.path.join(UPLOADS_FOLDER, secure_filename(file.filename))
        file.save(temp_path)

        try:
            result = process_palette_extraction(temp_path, params)
            return jsonify({"success": True, "result": result})
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    except Exception as e:
        log_activity("extraction_error", {"error": str(e)}, "error")
        return (
            jsonify({"success": False, "error": "Błąd serwera przy ekstrakcji palety"}),
            500,
        )


@webview_bp.route("/api/algorithm_01/transfer", methods=["POST"])
def handle_palette_transfer():
    """API dla nowego panelu transferu palety."""
    ensure_folders()
    log_activity("transfer_request_start")
    master_path, target_path = None, None
    try:
        if "master_image" not in request.files or "target_image" not in request.files:
            return (
                jsonify({"success": False, "error": "Brak obrazu master lub target"}),
                400,
            )

        master_file = request.files["master_image"]
        target_file = request.files["target_image"]

        if (
            not master_file.filename
            or not target_file.filename
            or not allowed_file(master_file.filename)
            or not allowed_file(target_file.filename)
        ):
            return jsonify({"success": False, "error": "Nieprawidłowe pliki"}), 400

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        master_filename = f"{timestamp}_master_{secure_filename(master_file.filename)}"
        target_filename = f"{timestamp}_target_{secure_filename(target_file.filename)}"
        master_path = os.path.join(UPLOADS_FOLDER, master_filename)
        target_path = os.path.join(UPLOADS_FOLDER, target_filename)
        master_file.save(master_path)
        target_file.save(target_path)

        params = {
            "num_colors": int(request.form.get("num_colors", 16)),
            "palette_method": request.form.get("palette_method", "kmeans"),
            "quality": int(request.form.get("quality", 5)),
            "distance_metric": request.form.get("distance_metric", "weighted_hsv"),
            # Wagi HSV
            "hue_weight": float(request.form.get("hue_weight", 3.0)),
            "saturation_weight": float(request.form.get("saturation_weight", 1.0)),
            "value_weight": float(request.form.get("value_weight", 1.0)),
            # Dithering
            "dithering_method": request.form.get("dithering_method", "none"),
            "dithering_strength": float(request.form.get("dithering_strength", 8.0)),
            # Ekstremy
            "inject_extremes": request.form.get("inject_extremes") == "on",
            "preserve_extremes": request.form.get("preserve_extremes") == "on",
            "extremes_threshold": int(request.form.get("extremes_threshold", 10)),
            # Edge blur
            "edge_blur_enabled": request.form.get("edge_blur_enabled") == "on",
            "edge_detection_threshold": float(request.form.get("edge_detection_threshold", 25)),
            "edge_blur_radius": float(request.form.get("edge_blur_radius", 1.5)),
            "edge_blur_strength": float(request.form.get("edge_blur_strength", 0.3)),
            "edge_blur_device": request.form.get("edge_blur_device", "auto").lower(),
            # Color focus
            "use_color_focus": request.form.get("use_color_focus") == "on",
            # focus_ranges dodany poniżej (JSON)
            # Zaawansowane GPU/Wydajność
            "force_cpu": request.form.get("force_cpu") == "on",
            "gpu_batch_size": int(request.form.get("gpu_batch_size", 2_000_000)),
            "enable_kernel_fusion": request.form.get("enable_kernel_fusion") == "on",
            "gpu_memory_cleanup": request.form.get("gpu_memory_cleanup") == "on",
            "use_64bit_indices": request.form.get("use_64bit_indices") == "on",
        }
        focus_ranges_str = request.form.get('focus_ranges_json', '[]')
        try:
            params["focus_ranges"] = json.loads(focus_ranges_str)
            if not isinstance(params["focus_ranges"], list):
                raise ValueError("focus_ranges musi być listą (JSON array).")
        except (json.JSONDecodeError, ValueError) as e:
            log_activity("transfer_error", {"error": f"Invalid focus_ranges format: {e}"}, "error")
            return jsonify({"success": False, "error": f"Nieprawidłowy format JSON w 'Color Focus Ranges': {e}"}), 400

        # === DEBUG: Color Focus parameters ===
        print(f"DEBUG WEBVIEW: use_color_focus = {params['use_color_focus']}")
        print(f"DEBUG WEBVIEW: focus_ranges = {params['focus_ranges']}")
        if params['use_color_focus']:
            print(f"DEBUG WEBVIEW: Color Focus ENABLED with {len(params['focus_ranges'])} ranges")
        else:
            print("DEBUG WEBVIEW: Color Focus DISABLED")

        # Also log to the application logger
        log_activity("color_focus_debug", {
            "use_color_focus": params['use_color_focus'],
            "focus_ranges_count": len(params['focus_ranges']) if params['focus_ranges'] else 0,
            "focus_ranges": params['focus_ranges']
        })

        # --- ENGINE CHOICE ---
        engine_choice = request.form.get("engine", "auto").lower()
        device_used = "cpu"  # domyślnie
        if engine_choice == "cpu":
            AlgoClass = PaletteMappingAlgorithm
            device_used = "cpu"
        elif engine_choice == "gpu":
            if PaletteMappingAlgorithmGPU:
                AlgoClass = PaletteMappingAlgorithmGPU
                device_used = "gpu"
            else:
                return jsonify({"success": False, "error": "Tryb GPU jest niedostępny na tym serwerze."}), 400
        else:  # auto
            if PaletteMappingAlgorithmGPU:
                AlgoClass = PaletteMappingAlgorithmGPU
                device_used = "gpu"
            else:
                AlgoClass = PaletteMappingAlgorithm
                device_used = "cpu"
        params["engine"] = engine_choice  # echo back

        log_activity("parameters_collected", params)

        algorithm = AlgoClass()
        output_filename = f"result_{target_filename}"
        output_path = os.path.join(RESULTS_FOLDER, output_filename)

        log_activity("processing_start", {"output_path": output_path, "params": params, "device": device_used})
        # --- START TIMING ---
        start_t = time.perf_counter()
        success = algorithm.process_images(
            master_path=master_path,
            target_path=target_path,
            output_path=output_path,
            **params,
        )
        processing_time_ms = round((time.perf_counter() - start_t) * 1000.0, 1)

        if not success:
            raise RuntimeError("Przetwarzanie algorytmu nie powiodło się.")

        result_url = f"/webview/results/{output_filename}"
        log_activity("transfer_request_success", {"result_url": result_url, "time_ms": processing_time_ms, "device": device_used})
        return jsonify(
            {
                "success": True,
                "result_url": result_url,
                "message": "Obraz przetworzony pomyślnie!",
                "processing_time_ms": processing_time_ms,
                "device_used": device_used,
                "params_echo": params,
            }
        )

    except Exception as e:
        log_activity("transfer_error", {"error": str(e)}, "error")
        if hasattr(current_app, "logger"):
            current_app.logger.exception("Błąd podczas transferu palety.")
        return (
            jsonify({"success": False, "error": f"Błąd wewnętrzny serwera: {str(e)}"}),
            500,
        )
    finally:
        # Czyszczenie plików tymczasowych
        if master_path and os.path.exists(master_path):
            os.remove(master_path)
        if target_path and os.path.exists(target_path):
            os.remove(target_path)
        log_activity(
            "temp_files_cleaned",
            {"master_path": master_path, "target_path": target_path},
        )



# --- Algorithm 05: LAB Color Transfer Routes ---

@webview_bp.route('/algorithm_05/transfer')
def algorithm_05_lab_transfer_page():
    """Serves the page for LAB Color Transfer (Algorithm 05)."""
    return render_template('algorithm_05_lab_transfer.html', now=datetime.now())

@webview_bp.route('/run_algorithm_05_lab_transfer', methods=['POST'])
def run_algorithm_05_lab_transfer():
    """Handles processing for LAB Color Transfer (Algorithm 05)."""
    start_time = time.time()
    source_image_path = None
    target_image_path = None
    mask_image_path = None

    try:
        ensure_folders()
        logger = current_app.logger # Use Flask's app logger if available

        if 'source_image' not in request.files or 'target_image' not in request.files:
            return jsonify({'error': 'Brak obrazu źródłowego lub docelowego.'}), 400

        source_file = request.files['source_image']
        target_file = request.files['target_image']

        if not source_file.filename or not target_file.filename:
            return jsonify({'error': 'Nie wybrano plików obrazów.'}), 400

        if not (allowed_file(source_file.filename) and allowed_file(target_file.filename)):
            return jsonify({'error': 'Niedozwolony format pliku. Dozwolone: png, jpg, jpeg, tif, tiff.'}), 400

        source_filename = secure_filename(f"src_{int(time.time())}_{source_file.filename}")
        target_filename = secure_filename(f"tgt_{int(time.time())}_{target_file.filename}")
        source_image_path = os.path.join(UPLOADS_FOLDER, source_filename)
        target_image_path = os.path.join(UPLOADS_FOLDER, target_filename)
        source_file.save(source_image_path)
        target_file.save(target_image_path)

        # Load images as NumPy arrays
        source_img_pil = Image.open(source_image_path).convert('RGB')
        target_img_pil = Image.open(target_image_path).convert('RGB')
        source_img_np = np.array(source_img_pil)
        target_img_np = np.array(target_img_pil)
        mask_img_np = None

        # Parameters
        method = request.form.get('method', 'basic')
        use_gpu = request.form.get('use_gpu') == 'on'
        tile_size = int(request.form.get('tile_size', 512))
        overlap = int(request.form.get('overlap', 64))

        method_params = {}
        if method == 'linear_blend':
            weights_json = request.form.get('channel_weights_json')
            if weights_json:
                method_params['channel_weights'] = json.loads(weights_json)
        elif method == 'selective':
            channels_json = request.form.get('selective_channels_json')
            if channels_json:
                method_params['selective_channels'] = json.loads(channels_json)
            method_params['blend_factor'] = float(request.form.get('blend_factor', 0.5))
            
            if 'mask_image' in request.files:
                mask_file = request.files['mask_image']
                if mask_file and mask_file.filename and allowed_file(mask_file.filename):
                    mask_filename = secure_filename(f"mask_{int(time.time())}_{mask_file.filename}")
                    mask_image_path = os.path.join(UPLOADS_FOLDER, mask_filename)
                    mask_file.save(mask_image_path)
                    mask_img_pil = Image.open(mask_image_path).convert('RGB') # or 'L' if mask is grayscale
                    mask_img_np = np.array(mask_img_pil)
                elif mask_file and mask_file.filename: # File provided but wrong type
                     return jsonify({'error': 'Niedozwolony format pliku maski.'}), 400
            # If mask is required for selective but not provided or invalid, could error here or let algorithm handle it
            if mask_img_np is None and method == 'selective': # Ensure mask is present for selective mode
                return jsonify({'error': 'Tryb selektywny wymaga obrazu maski.'}), 400

        elif method == 'adaptive':
            method_params['adaptation_method'] = request.form.get('adaptation_method', 'none')
            method_params['num_segments'] = int(request.form.get('num_segments', 5))
            method_params['delta_e_threshold'] = float(request.form.get('delta_e_threshold', 10.0))
            method_params['min_segment_size_perc'] = float(request.form.get('min_segment_size_perc', 0.01)) # Already 0-1
        
        # Config and Algorithm Instantiation
        lab_config = LABTransferConfig(use_gpu=use_gpu, tile_size=tile_size, overlap=overlap, method=method)
        # Potentially update lab_config with specific method params if constructor takes them
        # or pass them directly to process_image_hybrid if that's how it's designed.
        # For now, process_image_hybrid takes method_params separately.

        algorithm = LABColorTransfer(config=lab_config)

        # --- New Processing Logic for Algorithm 05 ---
        current_app.logger.info(f"Algorithm 05: Method '{method}' selected with params: {method_params}")

        # Convert RGB inputs to LAB
        src_lab = algorithm.rgb_to_lab_optimized(source_img_np)
        tgt_lab = algorithm.rgb_to_lab_optimized(target_img_np)
        if mask_img_np is not None:
            # Assuming mask is also RGB and needs conversion for selective transfer if it expects LAB
            # However, selective_lab_transfer in core.py takes mask_rgb: np.ndarray
            # So, for selective, we pass mask_img_np directly (as RGB).
            # If other methods needed a LAB mask, we'd convert it here.
            pass # mask_img_np remains RGB for selective_lab_transfer

        result_lab = None

        if method == 'basic':
            result_lab = algorithm.basic_lab_transfer(src_lab, tgt_lab)
        elif method == 'adaptive':
            # adaptive_lab_transfer might use parameters from self.config (e.g., num_segments)
            # Ensure lab_config passed to LABColorTransfer in routes.py has these if needed.
            result_lab = algorithm.adaptive_lab_transfer(src_lab, tgt_lab)
        elif method == 'linear_blend':
            channel_weights = method_params.get('channel_weights', {'L': 0.5, 'a': 0.5, 'b': 0.5})
            result_lab = algorithm.weighted_lab_transfer(src_lab, tgt_lab, weights=channel_weights)
        elif method == 'selective':
            if mask_img_np is None:
                return jsonify({'error': 'Tryb selektywny wymaga obrazu maski.'}), 400
            selective_channels = method_params.get('selective_channels', ['L', 'a', 'b'])
            blend_factor = method_params.get('blend_factor', 0.5)
            # selective_lab_transfer expects an RGB mask (mask_rgb)
            result_lab = algorithm.selective_lab_transfer(
                src_lab, 
                tgt_lab, 
                mask_img_np, # Pass the original RGB mask
                selective_channels=selective_channels, 
                blend_factor=blend_factor
            )
        elif method == 'hybrid':
            # Hybrid mode logic: Fallback to basic or implement specific hybrid approach.
            # For now, using basic as a fallback.
            current_app.logger.info("Hybrid mode selected, falling back to basic LAB transfer.")
            result_lab = algorithm.basic_lab_transfer(src_lab, tgt_lab) # Fallback to basic
        else:
            return jsonify({'error': f'Nieznana metoda transferu: {method}'}), 400

        # Convert result LAB back to RGB
        if result_lab is not None:
            result_np = algorithm.lab_to_rgb_optimized(result_lab)
        else:
            # This case should ideally not be reached if method validation is correct
            return jsonify({'error': 'Nie udało się przetworzyć obrazu z wybraną metodą.'}), 500



        # Save result
        result_pil = Image.fromarray(result_np.astype(np.uint8))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        result_filename = f"lab_transfer_result_{timestamp}.png"
        result_filepath = os.path.join(RESULTS_FOLDER, result_filename)
        result_pil.save(result_filepath)

        processing_time = time.time() - start_time
        if logger: # Check if logger is available
             logger.info(f"Algorithm 05 ({method}) processed in {processing_time:.2f}s. Result: {result_filename}")

        from flask import url_for # Temporary explicit import
        return jsonify({
            'processed_image_url': url_for('webview.static', filename=f'results/{result_filename}', _external=True),
            'message': f'Obraz przetworzony pomyślnie w {processing_time:.2f}s.'
        })

    except RequestEntityTooLarge:
        if logger: logger.error("Uploaded file too large for Algorithm 05.")
        return jsonify({'error': f'Plik jest zbyt duży. Maksymalny rozmiar to {MAX_FILE_SIZE // (1024*1024)}MB.'}), 413
    except Exception as e:
        if logger: logger.error(f"Error in Algorithm 05 processing: {str(e)}", exc_info=True)
        return jsonify({'error': f'Wystąpił błąd serwera: {str(e)}'}), 500
    finally:
        # Cleanup temporary files
        for p in [source_image_path, target_image_path, mask_image_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as e_clean:
                    if logger:
                        logger.error(f"Failed to clean up temp file {p}: {str(e_clean)}")

# --- End Algorithm 05 Routes ---


# --- FUNKCJE WEWNĘTRZNE I OBSŁUGA BŁĘDÓW ---


def process_palette_extraction(image_path, params):
    """Logika dla ekstrakcji palety (dla starego panelu /api/process)."""
    try:
        algorithm = PaletteMappingAlgorithm()
        algorithm.config["quality"] = params.get("quality", 5)
        palette_rgb = algorithm.extract_palette(
            image_path=image_path,
            num_colors=params["num_colors"],
            method=params["method"],
        )

        colors = []
        for r, g, b in palette_rgb:
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            hsl_color = rgb_to_hsl(r, g, b)
            colors.append(
                {
                    "hex": hex_color,
                    "rgb": [r, g, b],
                    "hsl": hsl_color,
                }
            )

        return {
            "palette": colors,
            "method": params["method"],
            "num_colors": params["num_colors"],
        }
    except Exception as e:
        log_activity("extraction_logic_error", {"error": str(e)}, "error")
        raise


@webview_bp.errorhandler(404)
def not_found(e):
    """Obsługuje błąd 404 (nie znaleziono strony)."""
    return render_template("404.html", now=datetime.now()), 404


@webview_bp.errorhandler(500)
def internal_error(e):
    """Obsługuje wewnętrzne błędy serwera (500)."""
    log_activity("internal_server_error", {"error": str(e)}, "error")
    current_timestamp = datetime.now()
    return render_template(
        "500.html", 
        now=current_timestamp, 
        current_time=current_timestamp, 
        webview_version="1.1.0"
    ), 500