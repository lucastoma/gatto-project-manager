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

# --- UWAGA: Upewniamy się, że używamy prawdziwego algorytmu ---
# Zakładamy, że reszta aplikacji jest poprawnie skonfigurowana.
try:
    from ..algorithms.algorithm_01_palette.algorithm import PaletteMappingAlgorithm
except ImportError as e:
    # W przypadku błędu importu, rzucamy wyjątek, aby wyraźnie pokazać problem
    raise ImportError(
        f"CRITICAL: Failed to import PaletteMappingAlgorithm. Ensure the module exists and is correct. Error: {e}"
    )

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
            "dithering_method": request.form.get("dithering_method", "none"),
            "inject_extremes": request.form.get("inject_extremes") == "on",
            "preserve_extremes": request.form.get("preserve_extremes") == "on",
            "extremes_threshold": int(request.form.get("extremes_threshold", 10)),
            "edge_blur_enabled": request.form.get("edge_blur_enabled") == "on",
            "edge_detection_threshold": float(request.form.get("edge_detection_threshold", 25)),
            "edge_blur_radius": float(request.form.get("edge_blur_radius", 1.5)),
            "edge_blur_strength": float(request.form.get("edge_blur_strength", 0.3)),
            "edge_blur_device": request.form.get("edge_blur_device", "auto").lower(),
            "quality": int(request.form.get("quality", 5)),
            "distance_metric": request.form.get("distance_metric", "weighted_hsv"),
        }
        # === COLOR FOCUS PARAMS ===
        params['use_color_focus'] = request.form.get('use_color_focus') == "on"
        focus_ranges_str = request.form.get('focus_ranges_json', '[]')
        try:
            params['focus_ranges'] = json.loads(focus_ranges_str)
            if not isinstance(params['focus_ranges'], list):
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

        log_activity("parameters_collected", params)

        algorithm = PaletteMappingAlgorithm()
        output_filename = f"result_{target_filename}"
        output_path = os.path.join(RESULTS_FOLDER, output_filename)

        log_activity("processing_start", {"output_path": output_path, "params": params})
        success = algorithm.process_images(
            master_path=master_path,
            target_path=target_path,
            output_path=output_path,
            **params,
        )

        if not success:
            raise RuntimeError("Przetwarzanie algorytmu nie powiodło się.")

        result_url = f"/webview/results/{output_filename}"
        log_activity("transfer_request_success", {"result_url": result_url})
        return jsonify(
            {
                "success": True,
                "result_url": result_url,
                "message": "Obraz przetworzony pomyślnie!",
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