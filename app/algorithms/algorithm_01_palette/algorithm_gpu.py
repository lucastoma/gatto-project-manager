import numpy as np
import pyopencl as cl
from PIL import Image, ImageFilter
import logging
import time
import threading
import os
from typing import Any, Dict, List, Optional

# Logika pomocnicza i fallbacki
from . import algorithm_gpu_utils as utils
from . import algorithm_gpu_cpu_fallback as cpu
from . import algorithm_gpu_exceptions as err
from . import algorithm_gpu_config as cfg

# Zależności
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class OpenCLManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(OpenCLManager, cls).__new__(cls)
                cls._instance._initialized = False
                cls._instance.ctx: Optional[cl.Context] = None
                cls._instance.queue: Optional[cl.CommandQueue] = None
                cls._instance.prg: Optional[cl.Program] = None
                cls._instance.logger = utils.get_logger()
        return cls._instance

    def ensure_initialized(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self.logger.info("OpenCLManager: Rozpoczynam leniwą inicjalizację...")
            try:
                platforms = cl.get_platforms()
                if not platforms:
                    raise RuntimeError("Nie znaleziono platform OpenCL. Sprawdź sterowniki.")
                gpu_devices = []
                for p in platforms:
                    try:
                        gpu_devices.extend(p.get_devices(device_type=cl.device_type.GPU))
                    except cl.LogicError:
                        continue
                if not gpu_devices:
                    self.logger.warning("Nie znaleziono GPU, próbuję użyć CPU jako urządzenia OpenCL.")
                    cpu_devices = []
                    for p in platforms:
                        try:
                            cpu_devices.extend(p.get_devices(device_type=cl.device_type.CPU))
                        except cl.LogicError:
                            continue
                    if not cpu_devices:
                        raise RuntimeError("Nie znaleziono żadnych urządzeń OpenCL (ani GPU, ani CPU).")
                    device = cpu_devices[0]
                else:
                    device = gpu_devices[0]
                self.ctx = cl.Context([device])
                self.queue = cl.CommandQueue(self.ctx)
                self.logger.info(f"Zainicjalizowano OpenCL na urządzeniu: {device.name}")
                self._compile_kernel_from_file()
                self._initialized = True
                self.logger.info("OpenCLManager: Leniwa inicjalizacja zakończona pomyślnie.")
            except Exception as e:
                self.logger.error(f"KRYTYCZNY BŁĄD: Inicjalizacja OpenCL nie powiodła się: {e}", exc_info=True)
                self.ctx = None
                self.queue = None
                self.prg = None
                self._initialized = False
                raise err.GPUProcessingError(f"Inicjalizacja OpenCL nie powiodła się: {e}") from e

    def _compile_kernel_from_file(self):
        try:
            kernel_file_path = os.path.join(os.path.dirname(__file__), 'palette_mapping.cl')
            with open(kernel_file_path, 'r', encoding='utf-8') as kernel_file:
                kernel_code = kernel_file.read()
            self.prg = cl.Program(self.ctx, kernel_code).build()
        except cl.LogicError as e:
            self.logger.error(f"Błąd kompilacji kernela OpenCL: {e}")
            raise err.GPUProcessingError(f"Błąd kompilacji kernela: {e}")
        except FileNotFoundError:
            self.logger.error(f"Plik kernela 'palette_mapping.cl' nie został znaleziony.")
            raise err.GPUProcessingError("Nie znaleziono pliku kernela OpenCL.")

    def get_context(self) -> cl.Context:
        self.ensure_initialized()
        if not self.ctx:
            raise err.GPUProcessingError("Kontekst OpenCL jest niedostępny.")
        return self.ctx

    def get_queue(self) -> cl.CommandQueue:
        self.ensure_initialized()
        if not self.queue:
            raise err.GPUProcessingError("Kolejka poleceń OpenCL jest niedostępna.")
        return self.queue

    def get_program(self) -> cl.Program:
        self.ensure_initialized()
        if not self.prg:
            raise err.GPUProcessingError("Program kernela OpenCL jest niedostępny.")
        return self.prg



class PaletteMappingAlgorithmGPU:
    def __init__(self, config_path: str = None, algorithm_id: str = "algorithm_01_palette_production"):
        self.algorithm_id = algorithm_id
        self.logger = utils.get_logger()
        self.profiler = utils.get_profiler()
        self.name = "Palette Mapping (OpenCL Production)"
        self.version = "12.0-Final"
        self.default_config = cfg.get_default_config()
        self.config = cfg.load_config(config_path, self.default_config) if config_path else self.default_config.copy()
        self.bayer_matrix_8x8 = np.array([[0,32,8,40,2,34,10,42],[48,16,56,24,50,18,58,26],[12,44,4,36,14,46,6,38],[60,28,52,20,62,30,54,22],[3,35,11,43,1,33,9,41],[51,19,59,27,49,17,57,25],[15,47,7,39,13,45,5,37],[63,31,55,23,61,29,53,21]])

    def _apply_ordered_dithering(self, image_array: np.ndarray, strength: float) -> np.ndarray:
        """Zoptymalizowana, wektorowa implementacja ditheringu."""
        self.logger.info(f"Stosuję dithering z siłą {strength}.")
        h, w, _ = image_array.shape
        bayer_norm = self.bayer_matrix_8x8 / 64.0 - 0.5
        tiled_bayer = np.tile(bayer_norm, (h // 8 + 1, w // 8 + 1))[:h, :w]
        dither_pattern = tiled_bayer[:, :, np.newaxis] * strength
        return np.clip(image_array.astype(np.float32) + dither_pattern, 0, 255)

    def _preserve_extremes(self, mapped_array: np.ndarray, original_array: np.ndarray, threshold: int) -> np.ndarray:
        """Ulepszona wersja oparta na luminancji."""
        self.logger.info("Zachowuję skrajne wartości czerni i bieli.")
        luminance = np.dot(original_array[..., :3], [0.2989, 0.5870, 0.1140])
        black_mask = luminance <= threshold
        white_mask = luminance >= (255 - threshold)
        mapped_array[black_mask] = [0, 0, 0]
        mapped_array[white_mask] = [255, 255, 255]
        return mapped_array

    def _gaussian_weights(self, radius: int) -> np.ndarray:
        sigma = max(radius / 2.0, 0.1)
        offsets = np.arange(-radius, radius + 1, dtype=np.float32)
        weights = np.exp(-offsets ** 2 / (2 * sigma * sigma))
        weights /= weights.sum()
        return weights.astype(np.float32)

    def _blur_image_gpu_gauss(self, image_array: np.ndarray, radius: int) -> Optional[np.ndarray]:
        """Separable Gaussian blur on GPU using two 1-D passes."""
        if radius <= 0:
            return image_array
        h, w, _ = image_array.shape
        flat = image_array.astype(np.uint8).reshape(-1)
        weights = self._gaussian_weights(radius)
        try:
            cl_mgr = OpenCLManager()
            ctx = cl_mgr.get_context()
            queue = cl_mgr.get_queue()
            prg = cl_mgr.get_program()
            mf = cl.mem_flags
            buf_in = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=flat)
            buf_temp = cl.Buffer(ctx, mf.READ_WRITE, flat.nbytes)
            buf_out = cl.Buffer(ctx, mf.READ_WRITE, flat.nbytes)
            buf_w = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weights)

            global_size = (h * w,)
            # horizontal pass
            prg.gaussian_blur_h(queue, global_size, None,
                                buf_in, buf_temp, buf_w,
                                np.int32(radius), np.int32(w), np.int32(h))
            # vertical pass
            prg.gaussian_blur_v(queue, global_size, None,
                                buf_temp, buf_out, buf_w,
                                np.int32(radius), np.int32(w), np.int32(h))

            cl.enqueue_copy(queue, flat, buf_out).wait()
            for b in (buf_in, buf_temp, buf_out, buf_w):
                b.release()
            return flat.reshape((h, w, 3))
        except Exception as e:
            self.logger.warning(f"GPU gaussian blur nie powiódł się: {e}. Użycie CPU jako fallback (jeśli zaimplementowano).")
            return None

    def _apply_edge_blending(self, mapped_image: Image.Image, config: Dict[str, Any]) -> Image.Image:
        if not SCIPY_AVAILABLE:
            # brak Scipy; spróbuj GPU box blur bez maski krawędzi? potrzebujemy ndimage do maski
            self.logger.warning("Scipy niedostępne – nie mogę wygenerować maski krawędzi. Pomijam edge blur.")
            return mapped_image
        self.logger.info("Stosuję zaawansowane wygładzanie krawędzi.")
        mapped_array = np.array(mapped_image, dtype=np.float64)
        gray = np.dot(mapped_array[..., :3], [0.2989, 0.5870, 0.1140])
        from scipy import ndimage  # import lokalny, jeśli dostępny
        grad_x = ndimage.sobel(gray, axis=1); grad_y = ndimage.sobel(gray, axis=0)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edge_mask = magnitude > config["edge_detection_threshold"]
        radius = int(config["edge_blur_radius"])
        if radius > 0:
            edge_mask = ndimage.binary_dilation(edge_mask, iterations=radius)
        
        use_gpu = config.get("edge_blur_device", "auto").lower() != "cpu" and not config.get("force_cpu")

        blurred_array = None
        device_used = "none"
        start_t = time.perf_counter()

        if radius > 0 and use_gpu:
            blurred_array = self._blur_image_gpu_gauss(mapped_array.astype(np.uint8), radius)
            if blurred_array is not None:
                device_used = "gpu"

        if blurred_array is None:
            # fallback CPU Gauss
            blurred_array = mapped_array.copy()
            for channel in range(3):
                blurred_array[:, :, channel] = ndimage.gaussian_filter(mapped_array[:, :, channel], sigma=radius)
            device_used = "cpu"

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        self._log_blur_benchmark(device_used, radius, elapsed_ms, mapped_array.shape[1], mapped_array.shape[0])

        blend_factor = (edge_mask * config["edge_blur_strength"])[:, :, np.newaxis]
        result_array = (mapped_array * (1 - blend_factor) + blurred_array * blend_factor)
        return Image.fromarray(np.clip(result_array, 0, 255).astype(np.uint8))

    def _map_pixels_to_palette_opencl(self, image_array: np.ndarray, palette: List[List[int]], config: Dict[str, Any]) -> np.ndarray:
        with self.profiler.profile_operation("map_pixels_to_palette_opencl", algorithm_id=self.algorithm_id):
            start_time = time.perf_counter()
            cl_mgr = OpenCLManager()
            ctx = cl_mgr.get_context()
            queue = cl_mgr.get_queue()
            prg = cl_mgr.get_program()
            palette_np_rgb = np.array(palette, dtype=np.float32)
            palette_np_hsv = cpu.rgb2hsv(palette_np_rgb / 255.0).astype(np.float32).flatten()
            pixels_flat = image_array.reshape(-1, 3).astype(np.float32)
            mf = cl.mem_flags
            pixels_g, palette_hsv_g, output_g = None, None, None
            try:
                pixels_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pixels_flat)
                palette_hsv_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=palette_np_hsv)
                output_g = cl.Buffer(ctx, mf.WRITE_ONLY, pixels_flat.shape[0] * 4)
                local_size = (64,)
                global_size_raw = pixels_flat.shape[0]
                rounded_global_size = (global_size_raw + local_size[0] - 1) // local_size[0] * local_size[0]
                global_size = (rounded_global_size,)
                prg.map_palette(
                    queue, global_size, local_size,
                    pixels_g, palette_hsv_g, output_g,
                    np.int32(len(palette)), np.float32(config.get('hue_weight', 3.0))
                )
                closest_indices = np.empty(pixels_flat.shape[0], dtype=np.int32)
                cl.enqueue_copy(queue, closest_indices, output_g).wait()
                result_array = np.array(palette, dtype=np.uint8)[closest_indices].reshape(image_array.shape)
                self.logger.info(f"Przetworzono na GPU (OpenCL) {pixels_flat.shape[0]:,} pikseli w {(time.perf_counter() - start_time) * 1000:.1f}ms")
                return result_array
            except Exception as e:
                self.logger.error(f"Błąd podczas wykonywania kernela OpenCL: {e}", exc_info=True)
                raise err.GPUProcessingError(f"Błąd wykonania OpenCL: {e}")
            finally:
                if pixels_g: pixels_g.release()
                if palette_hsv_g: palette_hsv_g.release()
                if output_g: output_g.release()
    
    def _map_pixels_to_palette(self, image_array: np.ndarray, palette: List[List[int]], config: Dict[str, Any]) -> np.ndarray:
        """Dispatcher wybierający między GPU a CPU."""
        try:
            if image_array.size > 100_000 and not config.get('force_cpu'):
                return self._map_pixels_to_palette_opencl(image_array, palette, config)
        except err.GPUProcessingError as e:
            self.logger.warning(f"Przetwarzanie GPU nie powiodło się ({e}). Przełączam na CPU.")
        
        self.logger.info("Używam ścieżki CPU (obraz zbyt mały lub błąd GPU).")
        return cpu.map_pixels_to_palette_cpu(image_array, palette, config, self.logger)

    def process_images(self, master_path: str, target_path: str, output_path: str, **kwargs) -> bool:
        """Pełny potok przetwarzania, od ekstrakcji palety po finalny zapis."""
        run_config = self.default_config.copy()
        run_config.update(kwargs)
        
        try:
            # Konwersja typów, aby uniknąć błędów
            for key in ['hue_weight', 'dithering_strength', 'edge_blur_radius', 'edge_blur_strength']:
                if key in run_config: run_config[key] = float(run_config[key])
            for key in ['num_colors', 'quality', 'extremes_threshold', 'edge_detection_threshold', 'gpu_batch_size']:
                 if key in run_config: run_config[key] = int(run_config[key])
            if 'edge_blur_device' in run_config:
                run_config['edge_blur_device'] = str(run_config['edge_blur_device']).lower()
        except (ValueError, TypeError) as e:
            self.logger.error(f"Błąd konwersji typów w konfiguracji: {e}", exc_info=True)
            return False

        try:
            # Używamy cpu.extract_palette z `algorithm_gpu_cpu_fallback.py`
            master_image = cpu.safe_image_load(master_path, self.logger)
            # Pamiętaj, że ta wersja `extract_palette` nie przyjmuje `method`
            palette = cpu.extract_palette(
                image=master_image, 
                num_colors=run_config['num_colors'],
                quality=run_config['quality'],
                inject_extremes=run_config['inject_extremes'],
                max_palette_size=self.default_config.get('_max_palette_size', 256),
                logger=self.logger
            )

            target_image_pil = Image.open(target_path).convert("RGB")
            target_array = np.array(target_image_pil)
            array_to_map = target_array

            if run_config.get("dithering_method") == "ordered":
                array_to_map = self._apply_ordered_dithering(array_to_map, run_config.get("dithering_strength", 8.0))

            mapped_array = self._map_pixels_to_palette(array_to_map, palette, run_config)
            
            if run_config.get("preserve_extremes"):
                mapped_array = self._preserve_extremes(mapped_array, target_array, run_config.get("extremes_threshold", 10))

            mapped_image = Image.fromarray(mapped_array, "RGB")

            if run_config.get("edge_blur_enabled"):
                mapped_image = self._apply_edge_blending(mapped_image, run_config)
            
            mapped_image.save(output_path, quality=95)
            self.logger.info(f"Obraz pomyślnie zapisany w: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Główny proces przetwarzania nie powiódł się: {e}", exc_info=True)
            return False

    def _log_blur_benchmark(self, device: str, radius: int, elapsed_ms: float, width: int, height: int):
        """Loguje wynik benchmarku do CSV w folderze logs."""
        try:
            logs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
            os.makedirs(logs_dir, exist_ok=True)
            csv_path = os.path.join(logs_dir, "edge_blur_benchmarks.csv")
            header_needed = not os.path.exists(csv_path)
            import csv, datetime
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if header_needed:
                    writer.writerow(["timestamp", "device", "radius", "elapsed_ms", "width", "height"])
                writer.writerow([
                    datetime.datetime.now().isoformat(), device, radius, f"{elapsed_ms:.2f}", width, height
                ])
        except Exception:
            # nie blokuj głównego procesu w razie problemów z logiem
            pass

def create_palette_mapping_algorithm_gpu():
    return PaletteMappingAlgorithmGPU()
