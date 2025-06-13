# **Koncepcja Semantyczna i Matematyczna Algorytmu Transferu Kolorów LAB**

## **Wprowadzenie**

Celem algorytmu jest przeniesienie nastroju, oświetlenia i palety kolorów z jednego obrazu (zwanego **docelowym** lub _target_) na drugi (zwany **źródłowym** lub _source_). Aby operacja ta była zgodna z ludzką percepcją, cały proces odbywa się w percepcyjnej przestrzeni barw **CIELAB**, a nie w standardowej przestrzeni RGB.

## **Filar 1: Percepcyjna Przestrzeń Barw CIELAB**

#### **Koncepcja Semantyczna**

Standardowa przestrzeń RGB (Red, Green, Blue) jest zorientowana na sposób wyświetlania kolorów przez urządzenia, a nie na to, jak postrzega je ludzkie oko. Przestrzeń CIELAB została zaprojektowana tak, aby odległości geometryczne między punktami (kolorami) w tej przestrzeni jak najlepiej odpowiadały różnicom w percepcji tych kolorów.

Kluczową zaletą jest rozdzielenie informacji o **jasności** od informacji o **kolorze**:

- **L**\* (Lightness): Kanał luminancji, reprezentujący jasność (od 0=czarny do 100=biały).
- **a**\*: Kanał chrominancji, reprezentujący oś od zielonego (-128) do czerwonego (+127).
- **b**\*: Kanał chrominancji, reprezentujący oś od niebieskiego (-128) do żółtego (+127).

Dzięki temu możemy modyfikować kolorystykę obrazu (a\*, b\*) niezależnie od jego struktury jasności (L\*), co jest fundamentem tego algorytmu.

#### **Koncepcja Matematyczna**

Konwersja z przestrzeni RGB do CIELAB jest procesem dwuetapowym: **RGB → XYZ → CIELAB**.

1. **RGB do XYZ**: Obraz RGB jest najpierw linearyzowany (przez usunięcie korekcji gamma), a następnie transformowany do przestrzeni XYZ za pomocą stałej macierzy transformacji. Przestrzeń XYZ to pośredni model, który opisuje kolory w sposób niezależny od urządzenia.
2. **XYZ do CIELAB**: Wartości XYZ są normalizowane względem punktu bieli (np. D65), a następnie poddawane nieliniowej transformacji, która oblicza ostateczne wartości L\*, a\* i b\*. Ta nieliniowość jest kluczowa dla percepcyjnej jednolitości przestrzeni.

## **Filar 2: Rdzeń Algorytmu – Metody Transferu**

Po przekonwertowaniu obu obrazów (źródłowego i docelowego) do przestrzeni LAB, stosowana jest jedna z poniższych metod w celu modyfikacji obrazu źródłowego.

### **Metoda 1: Transfer Statystyczny**

#### **Koncepcja Semantyczna**

Główna idea polega na znormalizowaniu statystyk każdego kanału (L\*, a\*, b\*) obrazu źródłowego tak, aby pasowały do statystyk obrazu docelowego. W praktyce oznacza to "przesunięcie" i "rozciągnięcie" rozkładu wartości kolorów w obrazie źródłowym, aby jego średnia i odchylenie standardowe stały się takie same jak w obrazie docelowym.

#### **Koncepcja Matematyczna**

Dla każdego piksela w danym kanale (np. L\*) obrazu źródłowego, nowa wartość jest obliczana według wzoru:

Lnew​=(Lold−μLsource​)×σLsource​​σLtarget​​​+μLtarget​​

Gdzie:

- μsource​ – średnia wartość pikseli w kanale obrazu źródłowego.
- σsource​ – odchylenie standardowe w kanale obrazu źródłowego.
- μtarget​ – średnia wartość pikseli w kanale obrazu docelowego.
- σtarget​ – odchylenie standardowe w kanale obrazu docelowego.

Operacja jest powtarzana dla wszystkich trzech kanałów (L\*, a\*, b\*).

### **Metoda 2: Dopasowanie Histogramu (Histogram Matching)**

#### **Koncepcja Semantyczna**

Jest to metoda bardziej precyzyjna niż transfer statystyczny. Zamiast dopasowywać tylko dwa parametry (średnią i odchylenie standardowe), jej celem jest całkowite przekształcenie rozkładu wartości (histogramu) kanału źródłowego, aby idealnie naśladował kształt histogramu kanału docelowego. Można to sobie wyobrazić jako "przelanie" wartości z jednego pojemnika (histogramu) do drugiego, tak aby przyjął jego kształt.

#### **Koncepcja Matematyczna**

Proces opiera się na **dystrybuantach skumulowanych (CDF)**, które opisują prawdopodobieństwo, że wartość piksela jest mniejsza lub równa danej wartości.

1. Oblicz dystrybuantę (CDF) dla kanału źródłowego (CDFsource​).
2. Oblicz dystrybuantę (CDF) dla kanału docelowego (CDFtarget​).
3. Dla każdej unikalnej wartości v w kanale źródłowym, znajdź jej pozycję na dystrybuancie, np. p=CDFsource​(v).
4. Znajdź nową wartość vnew​ w kanale docelowym, która odpowiada tej samej pozycji na dystrybuancie docelowej, tj. CDFtarget​(vnew​)=p.
5. Zastąp wszystkie wystąpienia wartości v w obrazie źródłowym nową wartością vnew​. W praktyce odbywa się to za pomocą interpolacji liniowej (np.interp) między wartościami kwantyli obu rozkładów.

### **Metoda 3: Transfer Selektywny i Ważony**

#### **Koncepcja Semantyczna**

Metody te pozwalają na precyzyjną kontrolę nad efektem końcowym:

- **Transfer Selektywny**: Umożliwia zastosowanie powyższych technik tylko na wybranych kanałach. Najczęstszy przypadek to transfer tylko chrominancji (kanały a\* i b\*), aby zmienić paletę kolorów bez wpływu na oryginalną jasność i kontrast obrazu (L\*).
- **Transfer Ważony**: Jest to mechanizm do kontrolowania "siły" efektu. Po wykonaniu pełnego transferu, obraz wynikowy jest mieszany (blendowany) z oryginalnym obrazem źródłowym. Waga (od 0 do 1\) określa, czy efekt ma być subtelny, czy dominujący.

#### **Koncepcja Matematyczna**

- **Selektywny**: Formuły z Metody 1 lub 2 są aplikowane tylko do wybranych osi danych (np. drugiej i trzeciej dla a\* i b\*).
- Ważony: Obliczenie końcowej wartości piksela jest prostą interpolacją liniową:  
  Pfinal​=(1−w)×Psource​+w×Ptransferred​  
  Gdzie w to waga (siła) efektu.

## **Filar 3: Pomiar Jakości (Metryka CIEDE2000)**

#### **Koncepcja Semantyczna**

Aby obiektywnie ocenić, jak bardzo transfer zmienił obraz lub jak bardzo wynik różni się od zamierzonego celu, potrzebujemy miary, która odzwierciedla ludzką percepcję różnicy kolorów. Tą miarą jest **Delta E (ΔE)**.

#### **Koncepcja Matematyczna**

Delta E to pojedyncza liczba reprezentująca "odległość" między dwoma kolorami w przestrzeni LAB. Algorytm wykorzystuje najnowszą i najdokładniejszą standardową formułę **CIEDE2000**. Jest to zaawansowana wersja odległości euklidesowej, która wprowadza korekty wag dla luminancji, chrominancji i nasycenia w zależności od ich położenia w przestrzeni barw, aby lepiej naśladować nieliniowość ludzkiego wzroku.

Obliczenie średniej wartości ΔE dla całego obrazu daje ogólną miarę jakości i siły przeprowadzonego transferu.
