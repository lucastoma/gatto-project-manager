
# 🎨 GattoNero AI Assistant - Color Matching System
**Wersja dokumentu:** 1.1
**Data:** 8 czerwca 2025

## 1. Przegląd i Cele Systemu
Color Matching System to zaawansowany moduł dla Adobe Photoshop, zaprojektowany w architekturze Klient-Serwer. Jego celem jest dostarczenie profesjonalnym fotografom i retuszerom zestawu narzędzi do precyzyjnego i wydajnego dopasowywania kolorystyki oraz tonacji między obrazami. System ma na celu drastyczne przyspieszenie pracy przy obróbce serii zdjęć (np. reportaże, fotografia produktowa, portretowa) poprzez automatyzację powtarzalnych i złożonych zadań korekcji kolorów.

## 2. Architektura i Przepływ Danych
System składa się z dwóch głównych komponentów:
- **Frontend (Klient):** Zestaw skryptów `.jsx` dla Adobe Photoshop, odpowiedzialnych za interfejs użytkownika, przygotowanie danych (eksport warstw, odczyt próbników kolorów) i wizualizację wyników.
- **Backend (Serwer):** Aplikacja w Pythonie (Flask), która stanowi "mózg" operacji. Wykorzystuje biblioteki **OpenCV** i **scikit-learn** do wykonania wszystkich ciężkich obliczeń i analizy obrazu.

#### Przepływ Danych (Data Flow)
Przepływ danych jest kluczowy dla zrozumienia działania systemu.
```
1. Użytkownik w Photoshopie wybiera obrazy/warstwy i parametry w oknie skryptu JSX.
   ↓
2. Skrypt JSX eksportuje obrazy MASTER i TARGET jako bezstratne pliki TIFF do folderu tymczasowego.
   ↓
3. Skrypt JSX wysyła zapytanie HTTP POST do serwera Python, zawierające obrazy oraz wybraną metodę i jej parametry.
   ↓
4. Serwer Python wykonuje analizę i transformację kolorystyczną zgodnie z wybraną metodologią.
   ↓
5. Serwer zapisuje wynikowy obraz jako plik TIFF i odsyła do klienta odpowiedź (w formacie CSV lub JSON) zawierającą status operacji i nazwę pliku wynikowego.
   ↓
6. Skrypt JSX odbiera odpowiedź, otwiera plik wynikowy w Photoshopie i usuwa pliki tymczasowe.
```

#### Komunikacja: CSV vs JSON
- **Faza 1-2:** Dla prostych metod zwracających tylko status i nazwę pliku, użyjemy prostego formatu **CSV** (`"success,method1,result.tif"`), który jest trywialny do parsowania w JSX.
- **Faza 3-5:** Dla bardziej złożonych metod, które mogą wymagać przekazywania struktur danych (np. listy par kolorów), system przejdzie na format **JSON**, który jest standardem w komunikacji API.

## 3. Metodologia Przestrzeni Kolorów (Fundament Systemu)
Wszystkie kluczowe operacje porównywania i transformacji kolorów będą przeprowadzane w percypcyjnie jednorodnej przestrzeni barw **CIE L\*a\*b\***. Jest to absolutnie fundamentalne dla uzyskania wyników, które są spójne z ludzkim postrzeganiem kolorów.

- **Dlaczego LAB?** W przeciwieństwie do RGB, w przestrzeni LAB odległość euklidesowa między dwoma punktami odpowiada faktycznej, postrzeganej przez człowieka różnicy między kolorami. Dzięki temu algorytmy "wiedzą", które kolory są do siebie naprawdę podobne.
- **Proces Konwersji:** Standardowy proces konwersji w tle to `sRGB -> XYZ -> L*a*b*`. Biblioteka OpenCV obsługuje to za pomocą jednej funkcji: `cv2.cvtColor(image, cv2.COLOR_BGR2LAB)`.
- **Miara Różnicy Kolorów:** Jako metryka do oceny jakości i znajdowania najbliższych kolorów używana jest odległość **Delta E (ΔE)**. `sqrt((L1-L2)² + (A1-A2)² + (B1-B2)²)`.

## 4. Szczegółowa Metodologia Metod Transferu
Każda metoda jest zaprojektowana do rozwiązania innego problemu i używa innej podstawy teoretycznej.

---

### **METODA 1: Palette Mapping (AUTO) - Basic Stylization**

* **Cel:** Nadanie obrazowi `target` spójnej stylistyki kolorystycznej (tzw. "look") bazującej na palecie barw z obrazu `master`. Idealne do szybkiej stylizacji serii zdjęć.
* **Ulepszona Metodologia:** Zamiast mapować dwie oddzielne palety, co jest nieprecyzyjne, stosujemy podejście jednostronne. Tworzymy paletę wzorcową tylko z obrazu `master` i używamy jej jako jedynego dozwolonego zestawu kolorów dla obrazu `target`.
* **Kluczowe Kroki:**
    1.  Wczytaj obraz `master` i zmniejsz go do szerokości ~500px dla drastycznego przyspieszenia analizy.
    2.  Użyj algorytmu **K-Means** z `scikit-learn` na spłaszczonej liście pikseli obrazu `master`, aby znaleźć `k` środków klastrów. Te środki to nasza `master_palette`.
    3.  Wczytaj obraz `target` (w pełnej rozdzielczości).
    4.  Dla każdego piksela w obrazie `target` znajdź najbliższy mu kolor z `master_palette` (używając odległości w przestrzeni LAB).
    5.  Stwórz nowy obraz, zastępując każdy piksel z `target` jego znalezionym odpowiednikiem z `master_palette`.
* **Zalety:** Szybkie, proste w zrozumieniu, daje mocne, stylizowane efekty.
* **Wady:** Może niszczyć subtelne przejścia tonalne; nie nadaje się do precyzyjnej korekcji, a raczej do artystycznej interpretacji.

---

### **METODA 2: Statistical Color Transfer (AUTO) - Professional Grade**

* **Cel:** Jak najwierniejsze i najbardziej naturalne dopasowanie kolorystyki i tonacji obrazu `target` do `master`, z zachowaniem oryginalnych tekstur i detali. Metoda "studyjna".
* **Fundamenty Teoretyczne:** Implementacja bazuje na przełomowej pracy Reinharda i współpracowników z 2001 roku, która polega na dopasowaniu dwóch pierwszych momentów statystycznych (średniej i odchylenia standardowego) rozkładu pikseli w przestrzeni LAB.
* **Szczegółowa Metodologia:**
    1.  Oba obrazy (`master` i `target`) są konwertowane do przestrzeni **LAB** i typu danych `float64`, aby umożliwić precyzyjne operacje matematyczne.
    2.  Dla każdego z trzech kanałów (L\*, a\*, b\*) osobno obliczana jest średnia (`μ`) i odchylenie standardowe (`σ`).
    3.  Każdy piksel obrazu `target` jest normalizowany przez odjęcie jego średniej i podzielenie przez jego odchylenie standardowe.
    4.  Znormalizowany wynik jest następnie skalowany przez odchylenie standardowe obrazu `master` i przesuwany o jego średnią.
    5.  **Formuła dla każdego kanału:** `wynik = (piksel_target - μ_target) * (σ_master / (σ_target + 1e-6)) + μ_master`. Dodanie małej wartości `epsilon` (1e-6) chroni przed dzieleniem przez zero.
    6.  Wynikowe wartości w kanałach LAB są przycinane do ich prawidłowych zakresów (np. L\* do 0-100).
    7.  Obraz jest z powrotem konwertowany do formatu `uint8` i przestrzeni BGR do zapisu.
* **Zalety:** Zazwyczaj daje najbardziej naturalne i fotograficznie poprawne rezultaty, doskonale zachowuje szczegóły.
* **Wady:** Może zawieść (wprowadzić artefakty), jeśli obrazy `master` i `target` mają skrajnie różną zawartość (np. dopasowywanie zdjęcia pustyni do zdjęcia lasu).

---

### **METODA 3: Histogram Matching (AUTO) - Exposure & Contrast**

* **Cel:** Dopasowanie ogólnej jasności, kontrastu i rozkładu tonalnego. Szczególnie użyteczne, gdy zdjęcia w serii były robione w różnym oświetleniu.
* **Fundamenty Teoretyczne:** Metoda polega na modyfikacji histogramu obrazu `target` tak, aby jego skumulowana funkcja dystrybucji (CDF) pasowała do CDF obrazu `master`. W uproszczeniu, jeśli w obrazie `master` 20% pikseli jest bardzo ciemnych, to po transformacji w obrazie `target` również 20% pikseli będzie bardzo ciemnych.
* **Szczegółowa Metodologia:**
    1.  Dla najlepszych, najbardziej naturalnych rezultatów, operacja jest wykonywana **tylko na kanale L\* (jasność)** w przestrzeni LAB. Pozwala to na dopasowanie ekspozycji i kontrastu bez nienaturalnej zmiany kolorów.
    2.  Obliczany jest histogram i CDF dla kanału L\* obu obrazów.
    3.  Tworzona jest tablica przyporządkowania (Lookup Table, LUT), która mapuje każdą wartość jasności z `target` na nową wartość, tak aby finalny rozkład pasował do `master`.
    4.  Tablica LUT jest aplikowana do kanału L\* obrazu `target`.
    5.  Kanały a\* i b\* pozostają nietknięte, po czym obraz jest składany z powrotem.
* **Zalety:** Znakomita do wyrównywania ekspozycji w serii.
* **Wady:** Może być zbyt "agresywna" i prowadzić do utraty oryginalnego klimatu zdjęcia, jeśli jest używana nieostrożnie.

---

### **METODA 4: Manual Color Picker Pairs (RĘCZNY) - Precision Control**

* **Cel:** Zapewnienie użytkownikowi maksymalnej, chirurgicznej precyzji w dopasowywaniu kluczowych punktów kolorystycznych (np. odcień skóry, kolor produktu, błękit nieba).
* **Fundamenty Teoretyczne:** Metoda opiera się na **interpolacji przestrzennej**. Wpływ każdej pary próbników jest najsilniejszy w jej otoczeniu i maleje wraz z odległością, zgodnie z funkcją wagową (np. odwróconej odległości lub dzwonowej Gaussa).
* **Szczegółowa Metodologia:**
    1.  Użytkownik umieszcza w Photoshopie pary próbników kolorów (`ColorSampler`) na kluczowych, odpowiadających sobie elementach na obrazach `master` i `target`.
    2.  Skrypt `.jsx` odczytuje pozycje `(x, y)` i kolory `(R, G, B)` wszystkich próbników i przesyła te dane do serwera.
    3.  Serwer konwertuje wszystkie kolory do przestrzeni LAB.
    4.  Dla każdej pary próbników obliczany jest **wektor różnicy (delta)**: `Δ = kolor_master_LAB - kolor_target_LAB`.
    5.  Następnie, dla każdego piksela w obrazie `target`:
        a. Obliczana jest jego odległość od każdego z umieszczonych na nim próbników.
        b. Na podstawie tych odległości obliczane są wagi, określające jak mocno dany próbnik powinien wpływać na bieżący piksel.
        c. Finalny wektor korekty dla piksela jest obliczany jako **ważona suma** wszystkich wektorów `Δ`.
        d. Nowy kolor piksela to `oryginalny_kolor_LAB + finalny_wektor_korekty`.
    6.  Obraz wynikowy jest przycinany do prawidłowych zakresów i konwertowany z powrotem do RGB.
* **Zalety:** Niezrównana kontrola i precyzja. Użytkownik decyduje, co jest ważne.
* **Wady:** Najbardziej pracochłonna metoda, wymaga świadomego działania od użytkownika.

---

### **METODA 5: Hybrid Mix (COMBO) - Ultimate Control**

* **Cel:** Udostępnienie "stołu mikserskiego", który pozwala na łączenie i ważenie efektów metod automatycznych (2 i 3) z precyzją metody ręcznej (4), aby osiągnąć unikalne, w pełni kontrolowane rezultaty.
* **Szczegółowa Metodologia:**
    1.  System w tle oblicza wyniki dla Metody 2, 3 i 4.
    2.  Użytkownik za pomocą suwaków w interfejsie (lub parametrów w oknie dialogowym) określa wagi procentowe dla każdej z metod (np. 50% Statystycznej, 20% Histogramu, 30% Ręcznej).
    3.  Obraz wynikowy jest tworzony jako **ważona suma (blending)** obrazów z poszczególnych metod. Obliczenia muszą być wykonywane na obrazach w formacie `float`.
    4.  Dodatkowy suwak "Moc całkowita" (`Overall Strength`) pozwala na finalne zmiksowanie wyniku z oryginalnym obrazem `target`, kontrolując intensywność całego efektu.
* **Zalety:** Maksymalna kreatywna elastyczność, możliwość tworzenia autorskich "looków".
* **Wady:** Najbardziej złożona w implementacji i potencjalnie przytłaczająca dla nowych użytkowników.

---

### **METODA 6: ACES Color Space Transfer (CINEMATIC) - Professional Grade**

* **Cel:** Wykorzystanie przestrzeni kolorów ACES (Academy Color Encoding System) do uzyskania kinematograficznej jakości dopasowania kolorów z lepszym zachowaniem wysokich świateł i głębokich cieni.
* **Fundamenty Teoretyczne:** ACES to standardowa przestrzeń kolorów używana w przemyśle filmowym, która oferuje szerszy gamut kolorów i lepsze zachowanie tonów niż tradycyjne przestrzenie. Dzięki logarytmicznej charakterystyce lepiej radzi sobie z wysokim zakresem dynamicznym (HDR).
* **Szczegółowa Metodologia:**
    1.  Konwersja obrazów `master` i `target` z sRGB do przestrzeni ACES2065-1 przez pośrednią konwersję XYZ.
    2.  Zastosowanie transformacji statystycznej (podobnie jak w Metodzie 2) w przestrzeni ACES.
    3.  Dodatkowe przetwarzanie w przestrzeni ACEScct (logarytmicznej) dla lepszego zachowania szczegółów w cieniach i światłach.
    4.  Konwersja wyniku z powrotem do sRGB z zastosowaniem tone mappingu ACES.
* **Zalety:** Profesjonalna jakość kinematograficzna, doskonałe zachowanie HDR, standardowy workflow w branży filmowej.
* **Wady:** Wyższa złożoność obliczeniowa, wymaga precyzyjnej implementacji transformacji kolorów.

---

### **METODA 7: Perceptual Color Matching (CIEDE2000) - Scientific Precision**

* **Cel:** Wykorzystanie najnowocześniejszej metryki różnicy kolorów CIEDE2000 zamiast prostej odległości euklidesowej dla bardziej precyzyjnego dopasowania percepcyjnego.
* **Fundamenty Teoretyczne:** CIEDE2000 to najdokładniejsza obecnie dostępna formuła do obliczania różnic kolorów postrzeganych przez człowieka. Uwzględnia nieliniowości ludzkiego postrzegania, szczególnie w obszarach niebieskich i neutralnych.
* **Szczegółowa Metodologia:**
    1.  Konwersja obrazów do przestrzeni LAB z wysoką precyzją (float64).
    2.  Dla każdego piksela w obrazie `target` obliczenie odległości CIEDE2000 do wszystkich kolorów w palecie `master`.
    3.  Wybór najbliższego koloru na podstawie minimum CIEDE2000 zamiast odległości euklidesowej.
    4.  Opcjonalne zastosowanie interpolacji ważonej dla płynniejszych przejść.
* **Zalety:** Najwyższa precyzja percepcyjna, zgodność ze standardami CIE, idealne dla aplikacji wymagających dokładności kolorów.
* **Wady:** Znacznie wyższa złożoność obliczeniowa, wymaga implementacji skomplikowanej formuły CIEDE2000.

---

### **METODA 8: Adaptive Region-Based Matching (AI-ENHANCED) - Intelligent Segmentation**

* **Cel:** Inteligentne dopasowanie kolorów z uwzględnieniem semantyki obrazu - różne podejście dla skóry, nieba, roślinności, obiektów itp.
* **Fundamenty Teoretyczne:** Metoda wykorzystuje segmentację semantyczną do identyfikacji różnych regionów obrazu i stosuje dedykowane algorytmy dopasowania dla każdego typu obszaru.
* **Szczegółowa Metodologia:**
    1.  **Segmentacja semantyczna:** Użycie prostego klasyfikatora kolorów lub zaawansowanego modelu AI do identyfikacji regionów (skóra, niebo, roślinność, neutralne).
    2.  **Adaptacyjne parametry:** Dla każdego typu regionu zastosowanie optymalnych parametrów:
        - Skóra: priorytet dla odcieni i nasycenia z zachowaniem jasności
        - Niebo: focus na odcieniu z możliwością większych zmian nasycenia
        - Roślinność: zachowanie naturalnych odcieni zieleni
        - Neutralne: standardowe dopasowanie statystyczne
    3.  **Inteligentne blendowanie:** Płynne łączenie granic między regionami z użyciem masek gradientowych.
    4.  **Fallback:** W przypadku niepewności klasyfikacji, powrót do standardowej metody statystycznej.
* **Zalety:** Najbardziej naturalne rezultaty, zachowanie semantyki obrazu, profesjonalna jakość dla portretów.
* **Wady:** Najwyższa złożoność, wymaga dodatkowych modeli AI lub zaawansowanych heurystyk.

---

### **METODA 9: Temporal Consistency (VIDEO) - Motion Picture Workflow**

* **Cel:** Zapewnienie spójności kolorystycznej w sekwencjach wideo poprzez analizę czasową i minimalizację flickeringu.
* **Fundamenty Teoretyczne:** Rozszerzenie metod statycznych o wymiar czasowy, z uwzględnieniem koherencji między klatkami i minimalizacją artefaktów temporalnych.
* **Szczegółowa Metodologia:**
    1.  **Analiza sekwencji:** Wczytanie sekwencji klatek i analiza zmian kolorystycznych w czasie.
    2.  **Temporal smoothing:** Zastosowanie filtrów temporalnych do wygładzenia nagłych zmian kolorów.
    3.  **Keyframe-based matching:** Dopasowanie kluczowych klatek z propagacją zmian na sąsiednie klatki.
    4.  **Flicker reduction:** Algorytmy minimalizujące migotanie kolorów między klatkami.
* **Zalety:** Profesjonalny workflow dla video, eliminacja flickeringu, spójność czasowa.
* **Wady:** Wymaga przetwarzania sekwencji, znacznie wyższe wymagania pamięciowe i obliczeniowe.