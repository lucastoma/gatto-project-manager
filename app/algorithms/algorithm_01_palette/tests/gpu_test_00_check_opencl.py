import numpy as np
import unittest

try:
    import pyopencl as cl
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False

@unittest.skipIf(not PYOPENCL_AVAILABLE, "Biblioteka PyOpenCL nie jest zainstalowana. Pomijam test.")
class TestOpenCLInstallation(unittest.TestCase):
    """
    Super-prosty test diagnostyczny.
    Cel: Sprawdzić, czy PyOpenCL jest w stanie wykryć jakiekolwiek urządzenie (GPU/CPU)
    i wykonać na nim najprostszą operację.
    """
    def test_find_device_and_add_vectors(self):
        print("\n" + "="*70)
        print("🚀 START: Test diagnostyczny PyOpenCL")
        print("="*70)

        # Krok 1: Znajdowanie platformy i urządzenia
        print("📢 Krok 1: Szukanie dostępnych platform i urządzeń OpenCL...")
        try:
            platforms = cl.get_platforms()
            self.assertGreater(len(platforms), 0, "Nie znaleziono żadnej platformy OpenCL. Sprawdź sterowniki.")
            print(f"✅ Znaleziono {len(platforms)} platform(ę/y). Wybieram pierwszą: {platforms[0].name}")

            # Spróbuj znaleźć urządzenie GPU, jeśli nie ma, weź CPU
            try:
                device = platforms[0].get_devices(device_type=cl.device_type.GPU)[0]
                print(f"✅ Znaleziono urządzenie GPU: {device.name}")
            except IndexError:
                print("⚠️ Nie znaleziono GPU. Szukam urządzenia CPU...")
                device = platforms[0].get_devices(device_type=cl.device_type.CPU)[0]
                print(f"✅ Znaleziono urządzenie CPU: {device.name}")

            self.assertIsNotNone(device, "Nie udało się znaleźć żadnego urządzenia OpenCL.")
        except Exception as e:
            self.fail(f"Krytyczny błąd podczas inicjalizacji OpenCL: {e}")

        # Krok 2: Tworzenie kontekstu i kernela
        print("\n📢 Krok 2: Tworzenie kontekstu i kompilacja prostego kernela...")
        try:
            ctx = cl.Context([device])
            queue = cl.CommandQueue(ctx)

            # Prosty kernel do dodawania dwóch wektorów
            prg = cl.Program(ctx, """
            __kernel void sum(__global const float *a_g, __global const float *b_g, __global float *res_g) {
              int gid = get_global_id(0);
              res_g[gid] = a_g[gid] + b_g[gid];
            }
            """).build()
            print("✅ Kontekst i kernel gotowe.")
        except Exception as e:
            self.fail(f"Błąd podczas tworzenia kontekstu lub kompilacji kernela: {e}")

        # Krok 3: Przygotowanie danych i wykonanie obliczeń
        print("\n📢 Krok 3: Przygotowanie danych i wykonanie dodawania na urządzeniu...")
        a_np = np.random.rand(50000).astype(np.float32)
        b_np = np.random.rand(50000).astype(np.float32)

        mf = cl.mem_flags
        a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
        b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
        res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)

        prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)
        print("✅ Kernel wykonany.")

        # Krok 4: Weryfikacja wyniku
        print("\n📢 Krok 4: Pobieranie wyników i weryfikacja poprawności...")
        res_np = np.empty_like(a_np)
        cl.enqueue_copy(queue, res_np, res_g)

        expected_result = a_np + b_np
        self.assertTrue(np.allclose(res_np, expected_result), "Wynik z GPU nie zgadza się z wynikiem z CPU.")
        print("✅ WYNIK POPRAWNY! OpenCL działa na Twoim sprzęcie.")
        print("="*70)

if __name__ == '__main__':
    # Upewnij się, że masz zainstalowane `pyopencl` i `numpy`
    # (powinno być z pliku requirements.txt)
    unittest.main(verbosity=2)
