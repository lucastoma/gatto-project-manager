from tests.base_test_case import BaseAlgorithmTestCase
import os
import unittest

class TestBaseCaseDemo(BaseAlgorithmTestCase):
    def test_create_image(self):
        path = self.create_test_image("demo.png", shape=(32, 32, 3), color=[255, 0, 0])
        self.assertTrue(os.path.exists(path))
        print(f"[TEST] Utworzono plik: {path}")

    def test_create_image_with_noise(self):
        path = self.create_test_image("demo_noise.png", shape=(32, 32, 3))
        self.assertTrue(os.path.exists(path))
        print(f"[TEST] Utworzono plik: {path}")

if __name__ == "__main__":
    unittest.main()
