import socket
import subprocess
import time
import sys
import os
import re
import argparse
import signal
from app.server import app
from app.core.development_logger import get_logger

# Global logger for the script
logger = get_logger("run_server_script")

def get_local_non_loopback_ip():
    """Tries to get a non-loopback local IP address for display purposes."""
        try:
        # -H for no header, makes parsing cleaner
        cmd = f"ss -Hlntp sport = :{port}"
        logger.debug(f"Wykonywanie komendy: {cmd}")
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=False # Do not raise exception on non-zero exit code
        )

        output_to_parse = result.stdout
        if result.stderr:
            logger.debug(f"Stderr z komendy ss dla portu {port}:\n{result.stderr.strip()}")
            # In some cases, ss might output to stderr even on success or for warnings.
            # If stdout is empty but stderr has info, we might consider parsing stderr, 
            # but typically PIDs are in stdout.

        if output_to_parse:
            logger.debug(f"Stdout z ss dla portu {port}:\n{output_to_parse.strip()}")
            # Example ss output (no header with -H): LISTEN 0 4096 *:5000 *:* users:(("python",pid=12345,fd=3))
            matches = re.findall(r'pid=(\d+)', output_to_parse)
            if matches:
                pids_found = list(set(matches))
                logger.info(f"Znaleziono procesy na porcie {port} z PIDami (przez ss): {pids_found}")
            else:
                logger.info(f"Nie znaleziono informacji o PID w wyniku ss dla portu {port} (stdout był: '{output_to_parse.strip()}').")
        else:
            logger.info(f"Brak stdout z komendy ss dla portu {port}. Sprawdź stderr log powyżej. Możliwe, że port jest wolny.")
            # If there's no output, we assume no process was found by ss on that port.
            # The check_port_free function is the authority on whether the port is truly busy.
            # This function (kill_processes_on_port) is about finding PIDs if the port is busy.
            # If no PIDs found by ss, and port is busy, it's an issue for check_port_free to have caught.
            # We return True here indicating ss command ran and found nothing, not that port is free.
            # The calling logic in safe_start_server handles the overall flow.
            # If pids_found remains empty, the loop for killing processes won't run.

    except FileNotFoundError:
        logger.error("Blad: Komenda 'ss' nie zostala znaleziona. Upewnij sie, ze pakiet 'iproute2' jest zainstalowany.")
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(('8.8.8.8', 1))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    
    try:
        hostname = socket.gethostname()
        addresses = socket.getaddrinfo(hostname, None)
        for addr_info in addresses:
            family, _, _, _, sockaddr = addr_info
            if family == socket.AF_INET:
                ip = sockaddr[0]
                if not ip.startswith("127.") and not ip.startswith("169.254"):
                    return ip
    except socket.gaierror:
        pass
    
    return None

def check_port_free(port, host='127.0.0.1'):
    """Sprawdza czy port jest wolny na danym hoście."""
    logger.debug(f"Sprawdzanie portu {port} na {host}...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            logger.debug(f"Port {port} na {host} jest wolny.")
            return True
        except OSError:
            logger.debug(f"Port {port} na {host} jest zajęty.")
            return False

def kill_processes_on_port(port):
    """Zabija procesy na danym porcie (Linux/WSL) - używając ss, z próbą SIGTERM, potem SIGKILL."""
    logger.info(f"Próba zwolnienia portu {port}...")
    pids_found = []
    try:
        cmd = f"ss -lntp sport = :{port}"
        logger.debug(f"Wykonywanie komendy: {cmd}")
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=False
        )

        if result.stdout:
            logger.debug(f"Output z ss dla portu {port}:\n{result.stdout.strip()}")
            matches = re.findall(r'pid=(\d+)', result.stdout)
            if matches:
                pids_found = list(set(matches))
                logger.info(f"Znaleziono procesy na porcie {port} z PIDami: {pids_found}")
            else:
                logger.info(f"Nie znaleziono informacji o PID w wyniku ss dla portu {port}.")
        else:
            logger.info(f"Brak outputu z komendy ss dla portu {port} lub port jest już wolny.")
            return True

    except FileNotFoundError:
        logger.error("Blad: Komenda 'ss' nie zostala znaleziona. Upewnij sie, ze pakiet 'iproute2' jest zainstalowany.")
        return False
    except Exception as e:
        logger.error(f"Blad podczas wyszukiwania procesow (metoda ss): {e}", exc_info=True)
        return False

    if not pids_found:
        logger.info(f"Nie znaleziono procesów do zatrzymania na porcie {port}.")
        return True

    killed_any_process = False
    for pid_str in pids_found:
        pid = int(pid_str)
        try:
            logger.info(f"Wysyłanie SIGTERM do procesu PID {pid}...")
            os.kill(pid, signal.SIGTERM)
            wait_time = 2
            logger.info(f"Czekanie {wait_time}s na zamknięcie procesu PID {pid}...")
            time.sleep(wait_time)
            os.kill(pid, 0)
            logger.warning(f"Proces PID {pid} wciąż aktywny po SIGTERM. Wysyłanie SIGKILL...")
            os.kill(pid, signal.SIGKILL)
            logger.info(f"SIGKILL wysłany do PID {pid}.")
            killed_any_process = True
            time.sleep(0.5)
        except ProcessLookupError:
            logger.info(f"Proces PID {pid} zakończył się po SIGTERM (lub już nie istniał).")
            killed_any_process = True
        except PermissionError:
            logger.error(f"Brak uprawnień do wysłania sygnału do PID {pid}. Spróbuj z sudo.")
        except Exception as e:
            logger.error(f"Nieoczekiwany błąd podczas próby zatrzymania PID {pid}: {e}", exc_info=True)
    return killed_any_process

def safe_start_server(host, port):
    """Bezpiecznie uruchamia serwer Flask z kontrolą portu i logowaniem."""
    logger.info(f"Próba uruchomienia serwera na {host}:{port}")

    if not check_port_free(port):
        logger.warning(f"Port {port} jest zajęty. Próba zatrzymania istniejącego procesu...")
        if kill_processes_on_port(port):
            logger.info(f"Zakończono próbę zwolnienia portu {port}. Czekanie 2s na ustabilizowanie...")
            time.sleep(2)
            if check_port_free(port):
                logger.info(f"Port {port} został pomyślnie zwolniony.")
            else:
                logger.error(f"Nie udało się zwolnić portu {port} mimo prób. Sprawdź ręcznie procesy.")
                logger.error(f"Możesz użyć: sudo ss -lntp sport = :{port}  lub  sudo lsof -i:{port}")
                sys.exit(1)
        else:
            logger.error("Nie udało się podjąć próby zwolnienia portu z powodu wcześniejszego błędu.")
            sys.exit(1)
    else:
        logger.info(f"Port {port} jest wolny.")

    logger.info("Uruchamianie serwera Flask...")
    
    server_urls = []
    if host == '0.0.0.0':
        server_urls.append(f"  - http://127.0.0.1:{port} (localhost)")
        external_ip = get_local_non_loopback_ip()
        if external_ip:
            server_urls.append(f"  - http://{external_ip}:{port} (dostępny w sieci lokalnej, np. z Windows)")
        else:
            server_urls.append(f"  - (Nie udało się automatycznie wykryć adresu IP dostępnego z Windows)")
        server_urls.append(f"  (Serwer nasłuchuje na wszystkich interfejsach: 0.0.0.0:{port})")
    else:
        server_urls.append(f"  - http://{host}:{port}")

    logger.info("Serwer będzie dostępny na:")
    for url_info in server_urls:
        logger.info(url_info)
    
    logger.info("Aby zatrzymać serwer, naciśnij Ctrl+C")
    logger.info("-" * 50)

    try:
        app.run(host=host, port=port, debug=False)
    except KeyboardInterrupt:
        logger.info("Serwer zatrzymany przez użytkownika (Ctrl+C).")
    except Exception as e:
        logger.error(f"Krytyczny błąd podczas uruchamiania serwera Flask: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Uruchamia serwer Flask GattoNero z kontrolą portów.")
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("GATTONERO_HOST", "0.0.0.0"),
        help="Host, na którym serwer będzie nasłuchiwał (domyślnie: 0.0.0.0)."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("GATTONERO_PORT", 5000)),
        help="Port, na którym serwer będzie nasłuchiwał (domyślnie: 5000)."
    )
    args = parser.parse_args()
    safe_start_server(args.host, args.port)
import subprocess
import time
import sys
import os
import re # Added for the new kill_process_on_port
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
    """Zabija proces na danym porcie (Linux/WSL) - uzywajac ss"""
    try:
        # Uzyj ss do znalezienia PID. Ta komenda jest czesto bardziej niezawodna na nowszych systemach.
        # Opcje: -l (listening), -n (numeric ports), -p (processes), -t (tcp)
        result = subprocess.run(
            f"ss -lntp sport = :{port} | grep ':{port}'", # szukamy konkretnego portu
            shell=True,
            capture_output=True,
            text=True,
            check=False # Nie rzucaj wyjatku jesli grep nic nie znajdzie
        )

        pids_killed = False
        if result.stdout:
            # Wyszukaj PID uzywajac wyrazenia regularnego
            # Przyklad linii z ss: users:(("python3",pid=12345,fd=3))
            matches = re.findall(r'pid=(\d+)', result.stdout)
            
            if matches:
                unique_pids = set(matches) # Unikalne PIDy
                for pid in unique_pids:
                    print(f"Zatrzymuje proces PID {pid} na porcie {port} (znaleziony przez ss)...")
                    kill_result = subprocess.run(f'kill -9 {pid}', shell=True, capture_output=True, text=True)
                    if kill_result.returncode == 0:
                        print(f"Proces PID {pid} zatrzymany.")
                        pids_killed = True
                    else:
                        print(f"Nie udalo sie zatrzymac PID {pid}. Blad: {kill_result.stderr.strip()}")
                return pids_killed # True jesli chociaz jeden PID zostal zabity
            else:
                print(f"Nie znaleziono informacji o PID w wyniku ss dla portu {port}.")
                print(f"Output ss: {result.stdout.strip()}") # Logujemy output dla diagnostyki
                return False
        else:
            # To moze oznaczac, ze port jest wolny lub ss nie zwrocil oczekiwanego outputu
            print(f"Brak outputu z komendy ss dla portu {port} lub port jest wolny.")
            return False # Zakladamy, ze nic nie znaleziono do zabicia
            
    except FileNotFoundError:
        print("Blad: Komenda 'ss' nie zostala znaleziona. Upewnij sie, ze pakiet 'iproute2' jest zainstalowany.")
        return False
    except Exception as e:
        print(f"Blad podczas zabijania procesu (metoda ss): {e}")
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