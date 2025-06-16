import socket
import subprocess
import time
import sys
import os
from app.server import app

def check_port_free(port):
    """Sprawdza czy port jest wolny"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False

def kill_process_on_port(port):
    """Zabija proces na danym porcie (Linux/WSL)"""
    try:
        # Znajdź PID procesu na porcie - Linux version
        result = subprocess.run(
            f'lsof -ti:{port}',
            shell=True,
            capture_output=True,
            text=True
        )

        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    print(f"Zatrzymuje proces PID {pid} na porcie {port}...")
                    subprocess.run(f'kill -9 {pid}', shell=True, capture_output=True)
            return True
        return False
    except Exception as e:
        print(f"Blad podczas zabijania procesu: {e}")
        return False

def safe_start_server():
    """Bezpiecznie uruchamia serwer z kontrola portu"""
    port = 5000

    print("Sprawdzam port 5000...")

    if not check_port_free(port):
        print("Port 5000 jest zajety! Probuje zatrzymac istniejacy proces...")

        if kill_process_on_port(port):
            print("Czekam 2 sekundy na zwolnienie portu...")
            time.sleep(2)

            if check_port_free(port):
                print("Port zwolniony!")
            else:
                print("Nie udalo sie zwolnic portu. Sprawdz recznie procesy.")
                print("Uzyj: lsof -i:5000")
                sys.exit(1)
        else:
            print("Nie znaleziono procesu do zatrzymania, ale port jest zajety.")
            print("Sprobuj recznie: lsof -i:5000")
            sys.exit(1)
    else:
        print("Port 5000 jest wolny!")

    print("Uruchamiam serwer Flask...")
    print("Serwer bedzie dostepny na:")
    print("  - http://127.0.0.1:5000 (localhost)")
    print("  - http://172.21.30.59:5000 (z Windows)")
    print("Aby zatrzymac serwer, nacisnij Ctrl+C")
    print("-" * 50)

    # Uruchamiamy serwer Flask
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    safe_start_server()