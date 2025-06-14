#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Server Manager v2.2.0 - Advanced Flask Server Management for GattoNero AI Assistant

Features:
- Unified watchdog system via 'watch' command
- Configuration-driven setup from 'server_config.json'
- Advanced auto-restart with exponential backoff
- Graceful shutdown with '--force' option
- Structured, TTY-aware logging with log file redirection
- Production-ready deployment capabilities
- Intelligent Python environment detection (VENV vs. SYSTEM)

Usage:
    python server_manager_enhanced.py start [--auto-restart] [--port PORT]
    python server_manager_enhanced.py stop [--force]
    python server_manager_enhanced.py status [--detailed]
    python server_manager_enhanced.py restart [--auto-restart]
    python server_manager_enhanced.py watch [--interval SECONDS]
    python server_manager_enhanced.py logs [--tail LINES] [--file server|manager|errors]
"""

import sys
import os
import json
import time
import subprocess
import requests
import argparse
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any, List

# Próba importu psutil, jeśli jest dostępny
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False
    print(
        "[WARNING] psutil is not available. Some advanced features will be disabled. Run 'pip install psutil'"
    )


class ServerConfig:
    """Zarządza konfiguracją serwera z pliku JSON z wartościami domyślnymi."""

    def __init__(self, config_file: str = "server_config.json"):
        self.config_file = Path(config_file)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Ładuje konfigurację z pliku, łącząc ją z domyślnymi wartościami."""
        defaults = {
            "server": {
                "host": "127.0.0.1",
                "port": 5000,
                "environment": "development",
                "startup_command": [sys.executable, "run_server.py"],
                "python_executable": "",  # Puste oznacza auto-detekcję
                "startup_timeout": 15,
                "shutdown_timeout": 20,
                "health_check_interval": 5,
                "health_check_url": "/api/health",  # Domyślny endpoint health-check
            },
            "monitoring": {
                "failure_threshold": 3,
                "restart_delay": 5,
                "exponential_backoff": True,
                "max_backoff_delay": 60,
            },
            "logging": {
                "log_dir": "logs",
                "server_log_file": "gattonero_server.log",
                "server_error_file": "gattonero_server_errors.log",
                "manager_log_file": "server_manager.log",
            },
            "files": {"pid_file": ".server_info.json"},
        }

        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    user_config = json.load(f)
                return self._deep_merge(defaults, user_config)
            except json.JSONDecodeError as e:
                print(
                    f"[ERROR] Invalid JSON in {self.config_file}: {e}. Using default configuration."
                )
            except Exception as e:
                print(
                    f"[WARNING] Failed to load {self.config_file}: {e}. Using defaults."
                )
        else:
            print(
                f"[INFO] Configuration file '{self.config_file}' not found. Creating with default values."
            )
            try:
                with open(self.config_file, "w", encoding="utf-8") as f:
                    json.dump(defaults, f, indent=4)
            except Exception as e:
                print(f"[ERROR] Could not create default config file: {e}")

        return defaults

    def _deep_merge(
        self, base: Dict[str, Any], overlay: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Rekursywnie łączy dwa słowniki."""
        result = base.copy()
        for key, value in overlay.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get(self, section: str, key: Optional[str] = None, default=None):
        """Pobiera wartość konfiguracyjną z określonej sekcji."""
        if key is None:
            return self._config.get(section, default)
        return self._config.get(section, {}).get(key, default)

    def get_str(self, section: str, key: str, default: str = "") -> str:
        """Pobiera wartość konfiguracyjną jako string."""
        value = self.get(section, key, default)
        return str(value) if value is not None else default

    def get_int(self, section: str, key: str, default: int = 0) -> int:
        """Pobiera wartość konfiguracyjną jako int."""
        value = self.get(section, key, default)
        try:
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    def get_list(self, section: str, key: str, default: Optional[List] = None) -> List:
        """Pobiera wartość konfiguracyjną jako listę."""
        if default is None:
            default = []
        value = self.get(section, key, default)
        return list(value) if isinstance(value, list) else default

    def get_bool(self, section: str, key: str, default: bool = False) -> bool:
        """Pobiera wartość konfiguracyjną jako boolean."""
        value = self.get(section, key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value) if value is not None else default

    def get_health_check_url(self) -> str:
        """Zwraca endpoint health-check z konfiguracji."""
        return self.get_str("server", "health_check_url", "/api/health")


class EnhancedServerManager:
    """Zarządza cyklem życia serwera z monitoringiem, logowaniem i konfiguracją."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        environment: Optional[str] = None,
        config_file: str = "server_config.json",
    ):
        self.config = ServerConfig(config_file)

        self.host = host or self.config.get_str("server", "host", "127.0.0.1")
        self.port = port or self.config.get_int("server", "port", 5000)
        self.environment = environment or self.config.get_str(
            "server", "environment", "development"
        )
        self.base_url = f"http://{self.host}:{self.port}"
        self.health_check_url = self.config.get_health_check_url()

        self.log_dir = Path(self.config.get_str("logging", "log_dir", "logs"))
        self.log_dir.mkdir(exist_ok=True)
        self.pid_file = Path(
            self.config.get_str("files", "pid_file", ".server_info.json")
        )
        self.server_log_file = self.log_dir / self.config.get_str(
            "logging", "server_log_file", "gattonero_server.log"
        )
        self.server_error_file = self.log_dir / self.config.get_str(
            "logging", "server_error_file", "gattonero_server_errors.log"
        )
        self.manager_log_file = self.log_dir / self.config.get_str(
            "logging", "manager_log_file", "server_manager.log"
        )

        self.python_executable = self._detect_python_executable()

        default_startup_command = [self.python_executable, "-m", "app.server"]
        self.startup_command = self.config.get_list(
            "server", "startup_command", default_startup_command
        )
        if self.startup_command == [sys.executable, "-m", "app.server"]:
            self.startup_command = default_startup_command

        self.startup_timeout = self.config.get_int("server", "startup_timeout", 15)
        self.shutdown_timeout = self.config.get_int("server", "shutdown_timeout", 20)
        self.health_check_interval = self.config.get_int(
            "server", "health_check_interval", 5
        )
        self.failure_threshold = self.config.get_int(
            "monitoring", "failure_threshold", 3
        )
        self.restart_delay = self.config.get_int("monitoring", "restart_delay", 5)

        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_running = threading.Event()

    def _detect_python_executable(self) -> str:
        """Wykrywa najlepszy interpreter Pythona (venv jeśli dostępny)."""
        config_python = self.config.get_str("server", "python_executable", "")
        if config_python and Path(config_python).exists():
            self.log_event(
                f"Using configured Python executable: {config_python}", "INFO"
            )
            return config_python

        # Check for Linux venv first if on Linux
        if os.name == 'posix' and os.uname().sysname == 'Linux':
            linux_venv = Path("venv_linux")
            if linux_venv.exists() and linux_venv.is_dir():
                python_exe = linux_venv / "bin" / "python"
                if python_exe.exists():
                    self.log_event(
                        f"Linux virtual environment detected: {linux_venv}", "SUCCESS"
                    )
                    self.log_event(f"Using Linux venv Python: {python_exe}", "INFO")
                    return str(python_exe)

        # Fall back to standard venv paths
        venv_paths = [Path("venv"), Path(".venv"), Path("env"), Path(".env")]
        for venv_path in venv_paths:
            if venv_path.exists() and venv_path.is_dir():
                python_exe = (
                    venv_path / "Scripts" / "python.exe"
                    if os.name == "nt"
                    else venv_path / "bin" / "python"
                )
                if python_exe.exists():
                    self.log_event(
                        f"Virtual environment detected: {venv_path}", "SUCCESS"
                    )
                    self.log_event(f"Using venv Python: {python_exe}", "INFO")
                    return str(python_exe)

        if hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        ):
            self.log_event(
                "Already running in an activated virtual environment", "SUCCESS"
            )
            return sys.executable

        self.log_event(
            "No virtual environment detected, using system Python", "WARNING"
        )
        self.log_event("Consider creating a venv: python -m venv venv", "INFO")
        return sys.executable

    def _check_flask_install(self) -> bool:
        """Sprawdza, czy Flask jest zainstalowany w wybranym środowisku."""
        self.log_event(f"Checking for Flask in: {self.python_executable}", "INFO")
        try:
            command = [self.python_executable, "-c", "import flask"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.log_event("Flask is installed.", "SUCCESS")
                return True
            else:
                self.log_event(
                    "Flask is NOT installed in the selected environment.", "ERROR"
                )
                self.log_event(
                    f"To install, run: '{self.python_executable} -m pip install flask'",
                    "INFO",
                )
                return False
        except Exception as e:
            self.log_event(f"Could not check for Flask installation: {e}", "ERROR")
            return False

    def _verify_environment(self) -> bool:
        """Weryfikuje, czy środowisko Python jest poprawnie skonfigurowane."""
        python_path = Path(self.python_executable)
        if not python_path.exists():
            self.log_event(
                f"Python executable not found: {self.python_executable}", "ERROR"
            )
            return False

        try:
            result = subprocess.run(
                [self.python_executable, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self.log_event(f"Python version: {result.stdout.strip()}", "INFO")
        except Exception as e:
            self.log_event(f"Could not get Python version: {e}", "WARNING")

        return self._check_flask_install()

    def log_event(self, event: str, level: str = "INFO"):
        """Loguje zdarzenie do konsoli (z kolorami) i do pliku."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {"timestamp": timestamp, "level": level, "event": event}

        log_message = f"[{timestamp}] [{level}] {event}"

        if sys.stdout.isatty():
            colors = {
                "INFO": "\033[94m",
                "SUCCESS": "\033[92m",
                "WARNING": "\033[93m",
                "ERROR": "\033[91m",
                "RESET": "\033[0m",
            }
            color = colors.get(level, "")
            reset = colors["RESET"]
            print(f"{color}{log_message}{reset}")
        else:
            print(log_message)

        try:
            with open(self.manager_log_file, "a", encoding="utf-8") as f:
                f.write(f"{json.dumps(log_entry)}\n")
        except Exception as e:
            print(f"[ERROR] Could not write to manager log file: {e}")

    def save_server_info(self, process_info: Dict[str, Any]):
        """Zapisuje informacje o procesie serwera do pliku."""
        try:
            with open(self.pid_file, "w") as f:
                json.dump(process_info, f, indent=4)
        except Exception as e:
            self.log_event(f"Failed to save server info: {e}", "ERROR")

    def load_server_info(self) -> Optional[Dict[str, Any]]:
        """Wczytuje informacje o procesie serwera z pliku."""
        if not self.pid_file.exists():
            return None
        try:
            with open(self.pid_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None

    def clear_server_info(self):
        """Usuwa plik z informacjami o serwerze."""
        try:
            self.pid_file.unlink(missing_ok=True)
        except Exception:
            pass

    def is_process_running(self, pid: int) -> bool:
        """Sprawdza, czy proces o danym PID działa."""
        if not PSUTIL_AVAILABLE or psutil is None:
            return False
        try:
            return psutil.pid_exists(pid)
        except Exception:
            return False

    def is_port_in_use(self, port: int) -> bool:
        """Sprawdza, czy port jest w użyciu."""
        if not PSUTIL_AVAILABLE or psutil is None:
            return False
        try:
            for conn in psutil.net_connections():
                if conn.laddr and conn.laddr.port == port and conn.status == "LISTEN":
                    return True
        except Exception:
            pass
        return False

    def is_server_responding(self) -> bool:
        """Sprawdza, czy serwer odpowiada na żądania HTTP."""
        try:
            url = f"{self.base_url}{self.health_check_url}"
            response = requests.get(url, timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_process_info(self, pid: int) -> Dict[str, Any]:
        """Pobiera szczegółowe informacje o procesie."""
        if not PSUTIL_AVAILABLE or psutil is None or not self.is_process_running(pid):
            return {"status": "not_found"}
        try:
            process = psutil.Process(pid)
            with process.oneshot():
                return {
                    "pid": pid,
                    "status": process.status(),
                    "cpu_percent": process.cpu_percent(interval=0.1),
                    "memory_mb": round(process.memory_info().rss / 1024**2, 2),
                    "uptime_seconds": time.time() - process.create_time(),
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {"status": "error"}

    def is_running(self) -> bool:
        """Sprawdza, czy serwer działa i odpowiada."""
        info = self.load_server_info()
        if not info:
            return False
        pid = info.get("pid")
        if not pid:
            return False
        return self.is_process_running(pid) and self.is_server_responding()

    def start_server(self, auto_restart: bool = False, no_wait: bool = False) -> bool:
        """Uruchamia proces serwera i opcjonalnie watchdog."""
        if self.is_running():
            self.log_event("Server is already running.", "WARNING")
            return True

        if not self._verify_environment():
            self.log_event(
                "Python environment verification failed. Cannot start server.", "ERROR"
            )
            return False

        if self.is_port_in_use(self.port):
            self.log_event(
                f"Port {self.port} is already in use. Cannot start server.", "ERROR"
            )
            return False

        # Ensure the startup command uses the correct Python executable path
        if not self.startup_command or len(self.startup_command) == 0:
            self.startup_command = [self.python_executable, "run_server.py"]
        elif self.startup_command[0].endswith('python.exe'):
            # Replace Windows python.exe with the detected Python executable
            self.startup_command[0] = self.python_executable

        self.log_event(f"Starting server... Command: {' '.join(self.startup_command)}")
        env = os.environ.copy()
        env["FLASK_ENV"] = self.environment
        env["PYTHONUNBUFFERED"] = "1"  # Ensure Python output is not buffered

        # Define OS-specific arguments to detach the process
        kwargs = {}
        if os.name == "nt":
            # On Windows, DETACHED_PROCESS creates a new process
            # without a console and independent of the parent.
            kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            # On Unix, os.setsid makes the process a session leader,
            # detaching it from the controlling terminal.
            kwargs["start_new_session"] = True  # More modern approach than preexec_fn

        try:
            with open(self.server_log_file, "ab") as log_out, open(
                self.server_error_file, "ab"
            ) as log_err:
                # Add the kwargs to the Popen call
                process = subprocess.Popen(
                    self.startup_command,
                    stdout=log_out,
                    stderr=log_err,
                    env=env,
                    **kwargs,
                )

            self.save_server_info(
                {"pid": process.pid, "port": self.port, "started_at": time.time()}
            )

            if no_wait:
                self.log_event(
                    "Server starting in background. Check status or logs to confirm.",
                    "INFO",
                )
                # A brief pause to allow the process to initialize or fail.
                time.sleep(1.5)
                # Quick check if process died instantly
                if not self.is_process_running(process.pid):
                    self.log_event(
                        "Server process terminated immediately after start. Check error logs.",
                        "ERROR",
                    )
                    self.log_event(
                        f"Review logs: python server_manager_enhanced.py logs --file errors",
                        "INFO",
                    )
                    self.clear_server_info()  # Clear info if process died
                    return False
                if auto_restart:
                    self.start_watchdog()
                return True

            self.log_event(f"Waiting for server to respond (PID: {process.pid})...")
            for _ in range(self.startup_timeout):
                if self.is_server_responding():
                    self.log_event("Server started successfully.", "SUCCESS")
                    if auto_restart:
                        self.start_watchdog()
                    return True
                time.sleep(1)

            self.log_event("Server failed to start within timeout.", "ERROR")
            # Attempt to stop the failed process before returning
            current_pid_info = self.load_server_info()
            if current_pid_info and current_pid_info.get("pid") == process.pid:
                self.stop_server(force=True)  # This will also clear_server_info
            else:  # If PID info was overwritten or process never registered properly
                try:
                    if PSUTIL_AVAILABLE and psutil and psutil.pid_exists(process.pid):
                        psutil.Process(process.pid).kill()
                except Exception:  # psutil.NoSuchProcess or other errors
                    pass  # Process might already be gone
                self.clear_server_info()  # Ensure info is cleared if stop_server wasn't effective for this PID
            return False
        except Exception as e:
            self.log_event(f"Failed to start server: {e}", "ERROR")
            # Ensure server info is cleared on any exception during startup
            self.clear_server_info()
            return False

    def stop_server(self, force: bool = False) -> bool:
        """Zatrzymuje serwer, z opcją wymuszenia."""
        self.stop_watchdog()
        info = self.load_server_info()
        if not info or not self.is_process_running(info.get("pid", -1)):
            self.log_event("Server is not running.", "INFO")
            self.clear_server_info()
            return True

        pid = info["pid"]
        self.log_event(f"Stopping server (PID: {pid})...")

        if not force and PSUTIL_AVAILABLE and psutil:
            try:
                proc = psutil.Process(pid)
                # Na Windows SIGTERM to to samo co terminate()
                proc.terminate()
                self.log_event(
                    "Sent termination signal. Waiting for process to exit.", "INFO"
                )
                proc.wait(timeout=self.shutdown_timeout)
                self.log_event("Server shut down gracefully.", "SUCCESS")
                self.clear_server_info()
                return True
            except psutil.TimeoutExpired:
                self.log_event(
                    "Graceful shutdown timed out. Forcing termination.", "WARNING"
                )
            except psutil.NoSuchProcess:
                self.log_event("Process already stopped.", "SUCCESS")
                self.clear_server_info()
                return True
            except Exception as e:
                self.log_event(
                    f"Error during graceful shutdown: {e}. Forcing termination.",
                    "WARNING",
                )

        # Force termination
        if PSUTIL_AVAILABLE and psutil:
            try:
                proc = psutil.Process(pid)
                proc.kill()
                proc.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass  # Already gone
            except Exception as e:
                self.log_event(f"Error during force kill: {e}", "ERROR")
        else:  # Fallback dla systemów bez psutil
            try:
                os.kill(pid, 9)  # SIGKILL
            except ProcessLookupError:
                pass  # Already gone
            except Exception as e:
                self.log_event(f"Error during fallback kill: {e}", "ERROR")

        time.sleep(1)  # Give OS a moment to update process table
        if not self.is_process_running(pid):
            self.log_event("Server stopped forcefully.", "SUCCESS")
            self.clear_server_info()
            return True
        else:
            self.log_event("Failed to stop the server.", "ERROR")
            return False

    def restart_server(self, auto_restart: bool = False) -> bool:
        """Restartuje serwer."""
        self.log_event("Restarting server...")
        if self.stop_server():
            time.sleep(2)  # Czas na zwolnienie portu
            return self.start_server(auto_restart)
        self.log_event("Failed to stop the server, restart aborted.", "ERROR")
        return False

    def run_tests(self) -> bool:
        """Uruchom testy podstawowe."""
        if not self.is_running():
            self.log_event("Server not running. Cannot run tests.", "ERROR")
            return False

        self.log_event("Running tests...", "INFO")
        try:
            result = subprocess.run(
                [sys.executable, "test_algorithm_integration.py"],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
            )

            # Log the output
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    print(line)

            if result.stderr:
                self.log_event("STDERR output:", "WARNING")
                for line in result.stderr.strip().split("\n"):
                    self.log_event(line, "WARNING")

            if result.returncode == 0:
                self.log_event("Tests completed successfully.", "SUCCESS")
                return True
            else:
                self.log_event(
                    f"Tests failed with return code: {result.returncode}", "ERROR"
                )
                return False

        except Exception as e:
            self.log_event(f"Failed to run tests: {e}", "ERROR")
            return False

    def show_status(self, detailed: bool = False):
        """Wyświetla aktualny status serwera."""
        print("─" * 40)
        print("🖥️  Server Status")
        print("─" * 40)
        info = self.load_server_info()

        if not info or not self.is_process_running(info.get("pid", -1)):
            self.log_event("Server is NOT RUNNING.", "WARNING")
            self.clear_server_info()
            return

        pid = info["pid"]
        is_responding = self.is_server_responding()
        status_color = "SUCCESS" if is_responding else "ERROR"

        self.log_event(f"Server process is RUNNING (PID: {pid}).", "SUCCESS")
        self.log_event(
            f"Server HTTP endpoint is {'RESPONDING' if is_responding else 'NOT RESPONDING'}.",
            status_color,
        )

        if detailed and PSUTIL_AVAILABLE and psutil:
            proc_info = self.get_process_info(pid)
            if proc_info.get("status") != "not_found":
                uptime = timedelta(seconds=int(proc_info.get("uptime_seconds", 0)))
                print(f"  PID          : {proc_info.get('pid')}")
                print(f"  Uptime       : {uptime}")
                print(f"  Memory       : {proc_info.get('memory_mb', 'N/A')} MB")
                print(f"  CPU          : {proc_info.get('cpu_percent', 'N/A')} %")
        print("─" * 40)

    def start_watchdog(self):
        """Uruchamia wątek watchdog do monitorowania serwera."""
        if self.monitor_running.is_set():
            self.log_event("Watchdog is already running.", "INFO")
            return
        self.log_event("Starting watchdog monitor...", "INFO")
        self.monitor_running.set()
        self.monitor_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.monitor_thread.start()

    def stop_watchdog(self):
        """Zatrzymuje wątek watchdog."""
        if self.monitor_running.is_set():
            self.log_event("Stopping watchdog monitor...", "INFO")
            self.monitor_running.clear()
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=3)

    def _watchdog_loop(self):
        """Główna pętla wątku watchdog."""
        failures = 0
        while self.monitor_running.is_set():
            if not self.is_server_responding():
                failures += 1
                self.log_event(
                    f"Watchdog: Server health check failed ({failures}/{self.failure_threshold}).",
                    "WARNING",
                )
                if failures >= self.failure_threshold:
                    self.log_event(
                        "Watchdog: Failure threshold reached. Attempting to restart server.",
                        "ERROR",
                    )
                    if self.restart_server(auto_restart=True):
                        failures = 0
                    time.sleep(self.restart_delay)
            else:
                if failures > 0:
                    self.log_event("Watchdog: Server has recovered.", "SUCCESS")
                failures = 0

            self.monitor_running.wait(self.health_check_interval)

    def watch_server_foreground(self, interval: int):
        """Uruchamia dashboard monitorujący na pierwszym planie."""
        self.log_event(
            f"Starting foreground watch (interval: {interval}s). Press Ctrl+C to stop.",
            "INFO",
        )
        try:
            while True:
                if sys.stdout.isatty():
                    os.system("cls" if os.name == "nt" else "clear")
                self.show_status(detailed=True)
                time.sleep(interval)
        except KeyboardInterrupt:
            print()
            self.log_event("Foreground watch stopped by user.", "INFO")

    def show_logs(self, tail_lines: int, log_type: str):
        """Pokazuje ostatnie N linii określonego pliku logów."""
        log_files = {
            "manager": self.manager_log_file,
            "server": self.server_log_file,
            "errors": self.server_error_file,
        }
        log_file = log_files.get(log_type, self.manager_log_file)

        print(f"📋 Displaying last {tail_lines} lines of '{log_file.name}'")
        print("─" * 40)

        if not log_file.exists():
            self.log_event(f"Log file not found: {log_file}", "WARNING")
            return
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for line in lines[-tail_lines:]:
                print(line.strip())
        except Exception as e:
            self.log_event(f"Error reading log file: {e}", "ERROR")


def create_parser() -> argparse.ArgumentParser:
    """Tworzy parser argumentów linii poleceń."""
    help_epilog = """
-------------------------------------------------
 GattoNero AI - Przewodnik Szybkiego Startu
-------------------------------------------------
1. Uruchom serwer w tle:
   python server_manager_enhanced.py start

2. Sprawdź, czy działa:
   python server_manager_enhanced.py status
   
3. Uruchom testy lub pracuj z API/Photoshopem:
   python test_basic.py
   
4. Zatrzymaj serwer po pracy:
   python server_manager_enhanced.py stop
-------------------------------------------------
Użyj `[komenda] --help` aby zobaczyć opcje dla konkretnej komendy.
"""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=help_epilog,
    )
    subparsers = parser.add_subparsers(dest="command", help="Dostępne komendy")
    subparsers.required = False
    subparsers.default = "help"

    help_parser = subparsers.add_parser("help", help="Wyświetla tę wiadomość pomocy.")

    start = subparsers.add_parser("start", help="Uruchamia serwer w tle.")
    start.add_argument(
        "--auto-restart",
        action="store_true",
        help="Włącza watchdog do auto-restartu przy awarii.",
    )
    start.add_argument("--port", type=int, help="Nadpisuje port serwera z configa.")
    start.add_argument(
        "--no-wait",
        action="store_true",
        help="Nie czeka na health-check, zwraca od razu.",
    )

    stop = subparsers.add_parser("stop", help="Zatrzymuje serwer.")
    stop.add_argument(
        "--force", action="store_true", help="Wymusza natychmiastowe zatrzymanie."
    )

    restart = subparsers.add_parser("restart", help="Restartuje serwer.")
    restart.add_argument(
        "--auto-restart", action="store_true", help="Włącza watchdog po restarcie."
    )

    status = subparsers.add_parser("status", help="Pokazuje status serwera.")
    status.add_argument(
        "--detailed",
        action="store_true",
        help="Pokazuje szczegółowe informacje o procesie.",
    )

    watch = subparsers.add_parser("watch", help="Monitoruje serwer na żywo.")
    watch.add_argument(
        "--interval", type=int, default=5, help="Interwał sprawdzania w sekundach."
    )

    logs = subparsers.add_parser("logs", help="Wyświetla ostatnie logi.")
    logs.add_argument(
        "--tail", type=int, default=20, help="Liczba linii do wyświetlenia."
    )
    logs.add_argument(
        "--file",
        choices=["manager", "server", "errors"],
        default="server",
        help="Który plik logu pokazać.",
    )

    return parser


def main():
    """Główna funkcja wykonawcza."""
    parser = create_parser()
    args = parser.parse_args()

    # Jeśli komenda to 'help' lub nie podano żadnej, wyświetl pomoc i wyjdź
    if args.command == "help":
        parser.print_help()
        sys.exit(0)

    manager = EnhancedServerManager(port=getattr(args, "port", None))

    try:
        if args.command == "start":
            sys.exit(
                0
                if manager.start_server(
                    auto_restart=args.auto_restart,
                    no_wait=getattr(args, "no_wait", False),
                )
                else 1
            )
        elif args.command == "stop":
            sys.exit(0 if manager.stop_server(force=args.force) else 1)
        elif args.command == "restart":
            sys.exit(0 if manager.restart_server(auto_restart=args.auto_restart) else 1)
        elif args.command == "status":
            manager.show_status(detailed=args.detailed)
        elif args.command == "watch":
            manager.watch_server_foreground(args.interval)
        elif args.command == "logs":
            manager.show_logs(args.tail, args.file)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print()
        manager.log_event("Operation interrupted by user.", "INFO")
        manager.stop_watchdog()
        sys.exit(1)
    except Exception as e:
        manager.log_event(f"An unexpected error occurred: {e}", "ERROR")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
