#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GattoNero Server Runner - Cross-Platform Startup Script

Handles server startup, port conflict resolution, and provides helpful
IP address information for network access.
"""

import os
import sys
import subprocess
import signal
import time
import argparse
import socket
import re
from app.core.development_logger import get_logger

# Global logger for the script
logger = get_logger("run_server_script")

def get_local_non_loopback_ip():
    """Tries to get a non-loopback local IP address for display purposes."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        s.connect(('10.254.254.254', 1))  # Doesn't have to be reachable
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'  # Fallback
    finally:
        s.close()
    return ip

def _kill_processes_on_port_windows(port):
    """Finds and kills processes on a given port for Windows."""
    logger.info(f"Attempting to free port {port} on Windows...")
    pids_found = []
    try:
        cmd_find = f'netstat -aon | findstr ":{port}"'
        result_find = subprocess.run(cmd_find, shell=True, capture_output=True, text=True, check=False)
        if result_find.returncode == 0 and result_find.stdout:
            for line in result_find.stdout.strip().split('\n'):
                parts = line.strip().split()
                if len(parts) >= 5 and 'LISTENING' in parts[3].upper() and parts[4].isdigit():
                    pids_found.append(parts[4])
            pids_found = list(set(pids_found))
    except Exception as e:
        logger.error(f"Error finding processes on Windows: {e}", exc_info=True)
        return False

    if not pids_found:
        logger.info(f"No active processes found listening on port {port}.")
        return True

    logger.info(f"Terminating processes on port {port}: {pids_found}")
    all_killed = True
    for pid in pids_found:
        try:
            cmd_kill = f"taskkill /PID {pid} /F"
            subprocess.run(cmd_kill, shell=True, check=True, capture_output=True)
            logger.info(f"Successfully terminated PID {pid}.")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to kill PID {pid}. It may have already stopped. Stderr: {e.stderr.decode().strip()}")
        except Exception as e:
            logger.error(f"Unexpected error killing PID {pid}: {e}", exc_info=True)
            all_killed = False
    return all_killed

def _kill_processes_on_port_linux(port):
    """Finds and kills processes on a given port for Linux/WSL."""
    logger.info(f"Attempting to free port {port} on Linux/WSL...")
    pids = []
    try:
        cmd = f"ss -Hlntp sport = :{port}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout:
            matches = re.findall(r'pid=(\d+)', result.stdout)
            if matches:
                pids.extend(matches)
        pids = list(set(pids))
    except Exception as e:
        logger.error(f"Error finding processes with 'ss': {e}", exc_info=True)
        return False

    if not pids:
        logger.info(f"No active processes found on port {port}.")
        return True

    logger.info(f"Terminating processes on port {port}: {pids}")
    all_killed = True
    for pid_str in pids:
        try:
            pid = int(pid_str)
            os.kill(pid, signal.SIGTERM)
            logger.debug(f"Sent SIGTERM to PID {pid}. Waiting...")
            time.sleep(1)
            os.kill(pid, signal.SIGKILL) # Force kill if still alive
            logger.info(f"Sent SIGKILL to PID {pid}.")
        except ProcessLookupError:
            logger.info(f"Process {pid} already terminated.")
        except Exception as e:
            logger.error(f"Failed to kill process {pid}: {e}", exc_info=True)
            all_killed = False
    return all_killed

def kill_processes_on_port(port):
    """Dispatches process killing to the correct OS-specific function."""
    if sys.platform == "win32":
        return _kill_processes_on_port_windows(port)
    elif sys.platform.startswith("linux"):
        return _kill_processes_on_port_linux(port)
    else:
        logger.warning(f"Unsupported OS '{sys.platform}' for automatic port clearing.")
        return True

def safe_start_server(host, port):
    """Ensures the port is free before starting the server."""
    logger.info("Preparing to start server...")
    if not kill_processes_on_port(port):
        logger.error("Failed to clear target port. Aborting server start.")
        sys.exit(1)
    
    logger.info(f"Port {port} is clear. Starting server at http://{host}:{port}")
    
    local_ip = get_local_non_loopback_ip()
    logger.info("="*50)
    logger.info(f"  Server starting on:")
    logger.info(f"  - Local:   http://localhost:{port}")
    if local_ip != '127.0.0.1':
        logger.info(f"  - Network: http://{local_ip}:{port}")
    logger.info("="*50)

    from app.server import app
    try:
        app.run(host=host, port=port, debug=True)
    except Exception as e:
        logger.error(f"Failed to start Flask application: {e}", exc_info=True)
        sys.exit(1)

def main():
    """Main entry point for the server runner script."""
    parser = argparse.ArgumentParser(description="GattoNero Server Runner")
    parser.add_argument("--host", default=os.environ.get("GATTONERO_HOST", "0.0.0.0"), help="Host to bind the server to.")
    parser.add_argument("--port", type=int, default=os.environ.get("GATTONERO_PORT", 5000), help="Port to run the server on.")
    args = parser.parse_args()

    safe_start_server(args.host, args.port)

if __name__ == "__main__":
    main()
