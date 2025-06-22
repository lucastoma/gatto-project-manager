#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Server Manager - Advanced Flask Server Management
"""

import sys
import os
import json
import time
import subprocess
import requests
import argparse
import threading
import re
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

class EnhancedServerManager:
    """Manages the lifecycle of the Flask server."""

    def __init__(self, port=None, server_script='run_server.py', pid_file='server.pid', log_file='server.log'):
        self.project_root = Path(__file__).parent.resolve()
        self.server_script = self.project_root / server_script
        self.pid_file = self.project_root / pid_file
        self.log_file = self.project_root / log_file
        self.port = port if port is not None else 5000
        self.log_event(f"Server manager initialized for port {self.port}")

    def log_event(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level.upper()}] {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')

    def _find_process_on_port_windows(self, port):
        pids = []
        try:
            cmd = f'netstat -aon | findstr ":{port}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    parts = line.strip().split()
                    if len(parts) > 4 and 'LISTENING' in parts[3].upper():
                        pids.append(parts[4])
        except Exception as e:
            self.log_event(f"Error finding process on port {port} on Windows: {e}", "ERROR")
        return list(set(pids))

    def _find_process_on_port_linux(self, port):
        pids = []
        try: # lsof
            result = subprocess.run(f"lsof -ti:{port} -sTCP:LISTEN", shell=True, capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout.strip():
                pids.extend(result.stdout.strip().split('\n'))
        except Exception as e:
            self.log_event(f"lsof failed: {e}", "WARNING")
        try: # ss
            result = subprocess.run(f"ss -Hlntp sport = :{port}", shell=True, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                matches = re.findall(r'pid=(\d+)', result.stdout)
                if matches:
                    pids.extend(matches)
        except Exception as e:
            self.log_event(f"ss failed: {e}", "WARNING")
        return list(set(p for p in pids if p.isdigit()))

    def _find_process_on_port(self, port):
        if sys.platform == "win32":
            return self._find_process_on_port_windows(port)
        else:
            return self._find_process_on_port_linux(port)

    def _get_pid_from_file(self):
        if not self.pid_file.exists():
            return None
        try:
            return int(self.pid_file.read_text().strip())
        except (IOError, ValueError):
            return None

    def _is_process_running(self, pid):
        if pid is None: return False
        if sys.platform == "win32":
            result = subprocess.run(f'tasklist /FI "PID eq {pid}"', shell=True, capture_output=True, text=True)
            return str(pid) in result.stdout
        else:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False

    def _terminate_process(self, pid, force=False):
        if not self._is_process_running(pid): return
        if sys.platform == "win32":
            cmd = f"taskkill /PID {pid} /T {'/F' if force else ''}"
            subprocess.run(cmd, shell=True, capture_output=True)
        else:
            sig = signal.SIGKILL if force else signal.SIGTERM
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass # Already gone

    def start_server(self, auto_restart=False, no_wait=False):
        if self._is_process_running(self._get_pid_from_file()):
            self.log_event("Server is already running.")
            return True
        lingering_pids = self._find_process_on_port(self.port)
        if lingering_pids:
            self.log_event(f"Port {self.port} is busy. Stopping lingering processes: {lingering_pids}", "WARNING")
            for pid in lingering_pids:
                self._terminate_process(int(pid), force=True)
            time.sleep(2)
        command = [sys.executable, str(self.server_script), "--port", str(self.port)]
        try:
            p = subprocess.Popen(command, stdout=open(self.log_file, 'a'), stderr=subprocess.STDOUT)
            self.pid_file.write_text(str(p.pid))
            self.log_event(f"Server started with PID {p.pid}.")
            return True
        except Exception as e:
            self.log_event(f"Failed to start server: {e}", "ERROR")
            return False

    def stop_server(self, force=False):
        pid = self._get_pid_from_file()
        if pid and self._is_process_running(pid):
            self.log_event(f"Stopping server with PID {pid}...")
            self._terminate_process(pid, force)
            time.sleep(1)
            if not self._is_process_running(pid):
                self.log_event("Server stopped successfully.")
                self.pid_file.unlink(missing_ok=True)
            else:
                self.log_event("Failed to stop server.", "ERROR")
        else:
            self.log_event("Server not running or PID file is stale.")
        # Final cleanup
        lingering_pids = self._find_process_on_port(self.port)
        if lingering_pids:
            for pid_str in lingering_pids:
                self._terminate_process(int(pid_str), force=True)
        return True

    def restart_server(self, auto_restart=False):
        self.log_event("Restarting server...")
        self.stop_server()
        return self.start_server(auto_restart=auto_restart)

    def get_server_status(self):
        pid = self._get_pid_from_file()
        running = self._is_process_running(pid)
        status = {"running": running, "pid": pid if running else None, "port": self.port}
        return status

    def show_status(self, detailed=False):
        status = self.get_server_status()
        if status['running']:
            print(f"Server is RUNNING on port {status['port']} with PID {status['pid']}.")
        else:
            print("Server is STOPPED.")

    def show_logs(self, tail=20, file_choice='server'):
        if self.log_file.exists():
            lines = self.log_file.read_text().splitlines()
            for line in lines[-tail:]:
                print(line)
        else:
            print("Log file not found.")

def create_parser():
    parser = argparse.ArgumentParser(description="Enhanced Server Manager")
    subparsers = parser.add_subparsers(dest="command", required=True, help='Available commands')
    subparsers.add_parser("start", help="Starts the server.")
    subparsers.add_parser("stop", help="Stops the server.")
    subparsers.add_parser("restart", help="Restarts the server.")
    subparsers.add_parser("status", help="Checks server status.")
    logs_parser = subparsers.add_parser("logs", help="Shows server logs.")
    logs_parser.add_argument("--tail", type=int, default=20, help="Number of lines to show.")
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    manager = EnhancedServerManager()

    if args.command == "start":
        manager.start_server()
    elif args.command == "stop":
        manager.stop_server()
    elif args.command == "restart":
        manager.restart_server()
    elif args.command == "status":
        manager.show_status()
    elif args.command == "logs":
        manager.show_logs(args.tail)

if __name__ == "__main__":
    main()
