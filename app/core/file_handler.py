import os
import time
from werkzeug.utils import secure_filename

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(APP_DIR))
UPLOADS_DIR = os.path.join(PROJECT_ROOT, 'uploads')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

def save_temp_file(file_storage):
    """Zapisuje plik z requestu w folderze uploads z unikalną nazwą."""
    if not file_storage:
        return None
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    filename = secure_filename(file_storage.filename)
    base, extension = os.path.splitext(filename)
    unique_filename = f"{base}_{int(time.time())}{extension}"
    save_path = os.path.join(UPLOADS_DIR, unique_filename)
    file_storage.save(save_path)
    return save_path

def get_result_path(original_filename):
    """Generuje ścieżkę zapisu dla pliku wynikowego."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    base, extension = os.path.splitext(original_filename)
    return os.path.join(RESULTS_DIR, f"{base}_matched{extension}")
