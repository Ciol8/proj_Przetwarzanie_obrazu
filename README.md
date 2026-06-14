# Aplikacja do Przetwarzania Obrazu i Detekcji Obiektów

**Projekt zaliczeniowy** – Metody Detekcji i Interpretacji Obiektów 2025/2026  
Artur Nieżurawski, nr albumu 21469

---

## Opis projektu

Zaawansowana aplikacja desktopowa do interaktywnego przetwarzania i interpretacji obrazów cyfrowych. Interfejs graficzny utrzymany jest w estetyce Frutiger Aero i łączy klasyczne algorytmy filtracji oraz binaryzacji z nowoczesnymi modelami głębokiego uczenia do detekcji obiektów i estymacji póz człowieka. Aplikacja oferuje ponadto autorski efekt wizualny, system przetwarzania wsadowego oraz ukrytą symulację automatu komórkowego Conwaya.

---

## Technologie

| Biblioteka | Zastosowanie |
|---|---|
| Python 3.13 | Język programowania |
| PyQt6 | Interfejs graficzny |
| pywinstyles | Styl Frutiger Aero z efektem Aero Glass |
| OpenCV | Silnik przetwarzania obrazu |
| NumPy | Operacje macierzowe na pikselach |
| Matplotlib | Dynamiczne wykresy histogramu RGB |
| PyTorch i Torchvision | Model Faster R-CNN do detekcji ras kotów |
| Ultralytics YOLOv8 | Detekcja póz i ras kotów |
| PyQt6.QtMultimedia | Asynchroniczne odtwarzanie dźwięku |

---

## Wymagania i instalacja

### Instalacja zależności

```bash
pip install PyQt6 opencv-python numpy matplotlib torch torchvision ultralytics pywinstyles Pillow
```

### Wymagane pliki w folderze projektu

| Plik | Opis |
|---|---|
| `fasterrcnn_catbreeds_epoch_15.pth` | Wagi modelu Faster R-CNN wytrenowanego na rasach kotów(nie ma w repozytorium ponieważ plik waży za dużo) |
| `best_n.pt` | Niestandardowy model YOLOv8n do detekcji ras kotów |
| `jp2.jpg` lub `jp2.png` | Obraz docelowy dla efektu papajifikacji |
| `jp2twarz.jpg` lub `jp2twarz.png` | Twarz docelowa dla podmiany twarzy |
| `oil_sfx.mp3` | Efekt dźwiękowy dla funkcji smażenia obrazu |
| `btn_blue.png`, `btn_green.png`, `btn_yellow.png`, `btn_purple.png` | Tekstury przycisków interfejsu |
| `glider.png` | Tekstura przycisku automatu komórkowego |

Model `yolov8n-pose.pt` pobierany jest automatycznie przez bibliotekę Ultralytics przy pierwszym wywołaniu funkcji detekcji póz.

---

## Uruchomienie

```bash
python main.py
```

---

## Struktura projektu

```
projekt/
├── main.py               # Interfejs graficzny PyQt6
├── processor.py          # Logika przetwarzania obrazu i modele AI
├── game_of_life.py       # Silnik automatu komórkowego Conwaya
├── fasterrcnn_...pth     # Wagi modelu Faster R-CNN
├── best_n.pt             # Wagi modelu YOLOv8n dla kotów
├── jp2.jpg               # Obraz docelowy papajifikacji
└── oil_sfx.mp3           # Efekt dźwiękowy
```

---

## Funkcjonalności

### Interfejs graficzny i zarządzanie plikami

Interfejs aplikacji jest podzielony na 4 części. Pasek boczny zawiera wszystkie narzędzia do przetwarzania i detekcji obiektów, podgląd obrazu znajduje się w oknie po prawej strone, na dole aplikacji znajduje się histogram obrazu odświeżający się z każdą zmianą na obrazie, a na górnym pasku są przyciski odpowiedzialne za wczytanie, zapis i cofanie zmian przeprowadzonych na obrazie.  Wczytywanie obrazów realizowane jest metodą przeciągnij-i-upuść lub przez okno dialogowe z pełną obsługą polskich znaków diakrytycznych w ścieżkach systemowych. Obszar roboczy obsługuje przybliżanie kółkiem myszy, przesuwanie lewym przyciskiem i natychmiastowy podgląd oryginału po przytrzymaniu prawego przycisku. Każda operacja zapisywana jest na stosie historii umożliwiającym cofanie zmian.

---

### Wykrywanie krawędzi

Dostępne są trzy metody detekcji konturów. Każda operuje na wersji obrazu w skali szarości.

**Canny** – wielostopniowy detektor realizujący wygładzanie Gaussa, obliczanie gradientów Sobela, nielokalne tłumienie maksimów i śledzenie krawędzi przez histerezę z progami T_low=100 i T_high=200. Wynikiem są cienkie, jednopikselowe kontury.

**Sobel** – obliczane są pochodne cząstkowe pierwszego rzędu obrazu w kierunkach poziomym i pionowym, a wynik łączony jest jako moduł wektora gradientu według wzoru `|∇I| = √(Gx² + Gy²)`.

**Laplace** – obliczana jest druga pochodna przestrzenna obrazu według operatora `∇²I = ∂²I/∂x² + ∂²I/∂y²`. Przed detekcją stosowane jest wygładzanie Gaussa redukujące czułość operatora na szumy wysokoczęstotliwościowe.

---

### Progowanie obrazu

Dostępne są trzy metody binaryzacji zamieniające obraz w skali szarości na czarno-biały.

**Binarne** – stosowany jest stały próg T=127; piksel przyjmuje wartość 255 gdy jego jasność przekracza próg, lub 0 w przeciwnym wypadku.

**Otsu** – próg wyznaczany jest automatycznie przez minimalizację ważonej wariancji wewnątrzklasowej `σ²_w = w₁·σ₁² + w₂·σ₂²` obliczanej na podstawie histogramu obrazu.

**Adaptacyjne** – próg wyznaczany jest lokalnie dla każdego fragmentu obrazu jako ważona średnia Gaussa wartości pikseli w oknie sąsiedztwa o rozmiarze 11x11. Metoda zachowuje szczegóły przy nierównomiernym oświetleniu.

---

### Przetwarzanie wsadowe

Dedykowane okno dialogowe umożliwia przetworzenie wszystkich obrazów z wybranego folderu w jednej operacji. Obsługiwana jest zmiana rozdzielczości z interpolacją INTER_AREA oraz nałożenie znaku wodnego z pełną obsługą przezroczystości PNG. Kanał Alfa pliku znaku wodnego odczytywany jest piksel-po-pikselu i stosowany jako maska blendingu według wzoru `wynik = tło·(1−α) + logo·α`, a widoczność ograniczona jest do maksymalnie 50 procent.

---

### Dynamiczny histogram RGB

Wykres rozkładu jasności dla kanałów niebieskiego, zielonego i czerwonego generowany i aktualizowany jest automatycznie po każdej operacji na obrazie. Renderowanie wykresu wektorowego realizowane jest przez bibliotekę Matplotlib osadzoną bezpośrednio w interfejsie PyQt6 przez klasę `FigureCanvasQTAgg`.

---

### Detekcja ras kotów – Faster R-CNN

Model Faster R-CNN z kręgosłupem ResNet-50 został dotrenowany do klasyfikacji pięciu ras kotów: Bengal, Persian, Ragdoll, Russian Blue i Siamese. Na obrazie rysowane są czerwone ramki ograniczające z podpisem zawierającym nazwę rasy i wartość pewności predykcji. Próg akceptacji wyników regulowany jest suwakiem interfejsu w zakresie od 10 do 100 procent.

---

### Detekcja ras kotów – YOLOv8

Drugi, niezależny moduł detekcji ras kotów oparty jest na modelu YOLOv8n wytrenowanym na zbiorze danych przygotowanym na platformie Roboflow. Architektura YOLOv8n przetwarza obraz jednym przejściem sieci, co przekłada się na niższe czasy inferencji względem Faster R-CNN. Ramki detekcji rysowane są kolorem zielonym, co pozwala odróżnić wyniki obu modeli. Próg akceptacji regulowany jest tym samym suwakiem co w Faster R-CNN.

---

### Detekcja póz – YOLOv8 Pose

Model YOLOv8n-Pose wykrywa sylwetki ludzkie na zdjęciu i nakłada na nie szkielet złożony z 17 punktów kluczowych standardu COCO, obejmujących oczy, nos, tułów i kończyny. Wizualizacja szkieletu generowana jest metodą `results[0].plot()` bezpośrednio na macierzy obrazu.

---

### Efekt głębokiego smażenia obrazu

Autorski efekt wizualny przesterowuje obraz przez potok czterech operacji: wyostrzanie splotem z jądrem `[[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]`, czterokrotne podniesienie saturacji w przestrzeni HSV, liniowe przekształcenie kontrastu i jasności `I' = 2·I + 30` oraz dodanie szumu addytywnego z rozkładu jednostajnego `U[-60, 60]`. Przejście między obrazem oryginalnym a usmażonym animowane jest przez 60 klatek z interpolacją liniową alfa, a procesowi towarzyszy efekt dźwiękowy.

---

### Podmiana twarzy i sortowanie pikseli według jasności

Algorytm realizuje przetwarzanie w dwóch niezależnych fazach.

**Faza I – podmiana twarzy:** model YOLOv8-Pose wykrywa punkty kluczowe twarzy na zdjęciu i naciąga na nie twarz docelową przez obliczenie macierzy przekształcenia geometrycznego z trzech par punktów korespondencyjnych. Nakładka obracana jest o 5 stopni CCW wokół punktu nosa.

**Faza II – transport pikseli według jasności:** gdy na zdjęciu nie wykryto twarzy, piksele obrazu źródłowego sortowane są według rangi luminancji i przypisywane do pozycji docelowych o tej samej randze. Operacja realizowana jest metodą Coarse-to-Fine w blokach malejącej wielkości, a maska aktywnych pikseli wyklucza z sortowania obszary białego tła, ograniczając operację do obszaru twarzy docelowej.

---

### Automat komórkowy Conwaya

Ukryta funkcja uruchamiana po zastosowaniu filtru krawędziowego Canny. Białe kontury obrazu stają się żywymi komórkami automatu Conwaya, które ewoluują w nieskończoność zgodnie z regułami S23 i B3. Implementacja nie korzysta z pętli po pojedynczych pikselach – liczba żywych sąsiadów Moorea wyznaczana jest przez macierzowy splot 2D z jądrem pierścieniowym 3x3, a nowa generacja obliczana jest przez wektorowe maskowanie tablicy NumPy. Pozwala to na ewolucję pełnej siatki pikseli w ułamku sekundy niezależnie od rozdzielczości obrazu.

---

## Autor

Artur Nieżurawski – projekt na zaliczenie przedmiotu Metody Detekcji i Interpretacji Obiektów, ANS 2026

> Efekt dźwiękowy smażenia: Sound Effect by Abrar Hussain, Pixabay
