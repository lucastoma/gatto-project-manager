---
version: "1.0"
last_updated: 2025-06-10
author: lucastoma
type: roadmap
priority_system: "1-3"
tags:
  - todo
  - roadmap
  - palette
aliases:
  - "[[Algorithm 01 - TODO]]"
---

# TODO - [[Algorithm 01: Palette Mapping]]

## Priorytet 1 (Critical) 🔴

- [ ] **[[Handle images with alpha channel correctly]]**
  - **Opis:** Obecnie kanał alfa jest ignorowany i zastępowany białym tłem. Należy dodać opcję zachowania przezroczystości tam, gdzie to możliwe.
  - **Effort:** 1 dzień
  - **Dependencies:** Brak

## Priorytet 2 (Important) 🟡

- [ ] **[[Optimize Edge Blending]]**
  - **Opis:** Obecna implementacja `Edge Blending` bazuje na `scipy`, co może być wolne. Należy przepisać ją z użyciem zoptymalizowanych funkcji OpenCV (np. `cv2.GaussianBlur`, `cv2.Sobel`).
  - **Value:** Znaczne przyspieszenie działania dla jednej z kluczowych funkcji post-processingu.
  - **Effort:** 2 dni
- [ ] **[[Add color space selection for analysis]]**
  - **Opis:** Pozwól użytkownikowi wybrać, czy analiza kolorów (ekstrakcja palety) ma odbywać się w przestrzeni RGB czy LAB. Analiza w LAB może dać lepsze wyniki percepcyjne.
  - **Value:** Zwiększenie kontroli i jakości wyników dla zaawansowanych użytkowników.
  - **Effort:** 1 dzień

## Priorytet 3 (Nice to have) 🟢

- [ ] **[[Implement color weighting]]**
  - **Opis:** Dodaj możliwość ważenia kolorów, np. aby ignorować kolory z krawędzi obrazu lub skupić się na jego centrum podczas ekstrakcji palety.
  - **Value:** Lepsze dopasowanie palety do głównego motywu obrazu.
- [ ] **[[Export palette to Adobe Swatch Exchange]]**
  - **Opis:** Dodaj metodę `export_palette_to_ase(palette, output_path)`, która zapisze wygenerowaną paletę do pliku `.ase`.
  - **Value:** Ułatwienie integracji z innymi narzędziami Adobe.

## Backlog 📋

- [[Palette sorting]] - Dodanie opcji sortowania palety wynikowej (np. wg jasności, odcienia).
- [[Batch apply_mapping]] - Możliwość zaaplikowania jednej palety do całego folderu obrazów.
- [[Support for CMYK]] - Wstępna obsługa obrazów w trybie CMYK.

## Done ✅

- [x] **[[K-Means implementation]]** (2025-06-08) - Podstawowa, deterministyczna implementacja.
- [x] **[[Median Cut implementation]]** (2025-06-10) - Dodanie alternatywnej, szybszej metody ekstrakcji.
- [x] **[[Dithering and Extremes Preservation]]** (2025-06-09) - Zaimplementowano podstawowe opcje post-processingu.
- [x] **[[Initial documentation suite]]** (2025-06-10)

## Blocked 🚫

- [ ] Brak zablokowanych zadań.
