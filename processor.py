import time

import cv2
import numpy as np
from PIL import Image
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as TF
import ultralytics

class ImageProcessor:
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.history = []

        # Zmienne dla Sieci Neuronowej
        self.nn_model = None
        self.device = torch.device('cpu')
        self.cat_classes = {
            1: "Bengal",
            2: "Persian",
            3: "Ragdoll",
            4: "Russian Blue",
            5: "Siamese"
        }

    def _save_to_history(self):
        """Zapisuje kopię aktualnego obrazu do historii przed zmianą."""
        if self.processed_image is not None:
            if len(self.history) > 10:
                self.history.pop(0)
            self.history.append(self.processed_image.copy())

    def undo(self):
        """Przywraca ostatni zapisany stan z historii."""
        if self.history:
            self.processed_image = self.history.pop()
            return True
        return False

    def load_image(self, file_path):
        try:
            file_bytes = np.fromfile(file_path, dtype=np.uint8)
            self.original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            self.processed_image = self.original_image.copy()
            self.history = [] # Czyści historię przy nowym obrazie
            return True
        except Exception as e:
            print(f"Błąd wczytywania: {e}")
            return False

    def save_image(self, file_path):
        if self.processed_image is not None:
            try:
                extension = file_path.split('.')[-1]
                is_success, im_buf_arr = cv2.imencode(f'.{extension}', self.processed_image)
                if is_success:
                    im_buf_arr.tofile(file_path)
                    return True
            except Exception as e:
                print(f"Błąd zapisu: {e}")
        return False

    def get_image_for_gui(self):
        if self.processed_image is not None:
            rgb_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_image)
        return None

    def reset_image(self):
        if self.original_image is not None:
            self._save_to_history() # Reset to też zmiana, którą można cofnąć
            self.processed_image = self.original_image.copy()

    # ==========================================
    # WYKRYWANIE KRAWĘDZI
    # ==========================================
    def detect_edges_canny(self):
        if self.processed_image is not None:
            self._save_to_history()
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            self.processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def detect_edges_sobel(self):
        if self.processed_image is not None:
            self._save_to_history()
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = cv2.magnitude(sobelx, sobely)
            sobel_8u = cv2.convertScaleAbs(sobel_combined)
            self.processed_image = cv2.cvtColor(sobel_8u, cv2.COLOR_GRAY2BGR)

    def detect_edges_laplacian(self):
        if self.processed_image is not None:
            self._save_to_history()
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
            laplacian_8u = cv2.convertScaleAbs(laplacian)
            self.processed_image = cv2.cvtColor(laplacian_8u, cv2.COLOR_GRAY2BGR)

    # ==========================================
    # PROGOWANIE (3 Algorytmy)
    # ==========================================
    def threshold_binary(self, thresh=127):
        """Progowanie binarne - wszystko powyżej progu staje się białe, poniżej czarne."""
        if self.processed_image is not None:
            self._save_to_history()
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            _, thresh_img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
            self.processed_image = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)

    # ==========================================
    # SZTUCZNA SIEĆ NEURONOWA (Faster R-CNN)
    # ==========================================
    def load_nn_model(self, model_path="fasterrcnn_catbreeds_epoch_15.pth"):
        """Wczytuje wytrenowany model z dysku (uruchamiane tylko raz)."""
        if self.nn_model is not None:
            return True

        try:
            print("Wczytywanie modelu sieci neuronowej (to może chwilę potrwać)...")
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
            in_features = model.roi_heads.box_predictor.cls_score.in_features

            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 6)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()

            self.nn_model = model
            print("Model załadowany pomyślnie!")
            return True
        except Exception as e:
            print(f"Błąd ładowania modelu: {e}")
            return False

    def detect_cats_with_nn(self, threshold=0.6):
        """Przepuszcza aktualny obraz przez sieć i rysuje ramki. Zwraca wyniki jako listę do GUI."""
        if self.processed_image is None:
            return []  # Zwraca pustą listę zamiast None

        if not self.load_nn_model():
            return []

        self._save_to_history()

        img_rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
        img_tensor = TF.to_tensor(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.nn_model(img_tensor)[0]

        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()

        detected_info = []  # Lista przechowująca informacje tekstowe dla GUI

        for box, label, score in zip(boxes, labels, scores):
            if score >= threshold:
                x_min, y_min, x_max, y_max = map(int, box)
                class_name = self.cat_classes.get(label, "Nieznany")
                confidence = int(score * 100)

                # Zapisujemy info do wyświetlenia na panelu w GUI
                detected_info.append(f"{class_name} ({confidence}%)")

                # Czerwona ramka
                cv2.rectangle(self.processed_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)

                # Podpis w LEWYM DOLNYM rogu (wewnątrz ramki)
                text = f"{class_name}: {confidence}%"
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

                # Tło pod tekstem zaczyna się tuż przy dolnej krawędzi (y_max)
                cv2.rectangle(self.processed_image, (x_min, y_max - text_h - 10), (x_min + text_w, y_max),
                              (0, 0, 255), -1)

                # Rysowanie tekstu
                cv2.putText(self.processed_image, text, (x_min, y_max - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)

        return detected_info  # Zwracamy listę wykrytych kotów do aplikacji
    def threshold_otsu(self):
        """Metoda Otsu - automatycznie dobiera próg na podstawie histogramu."""
        if self.processed_image is not None:
            self._save_to_history()
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            _, thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.processed_image = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)

    def threshold_adaptive(self):
        """Progowanie adaptacyjne - oblicza próg dla małych obszarów obrazu."""
        if self.processed_image is not None:
            self._save_to_history()
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            thresh_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
            self.processed_image = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)

    # ==========================================
    # PRZETWARZANIE WSADOWE (Batch Processing)
    # ==========================================
    def process_batch(self, input_folder, output_folder, target_size=None, watermark_path=None):
        """Przetwarza wszystkie obrazy w folderze: zmienia rozmiar i/lub dodaje znak wodny."""
        if not os.path.exists(input_folder) or not os.path.exists(output_folder):
            return False

        # Wczytanie znaku wodnego (jeśli podano)
        watermark = None
        if watermark_path and os.path.exists(watermark_path):
            wm_bytes = np.fromfile(watermark_path, dtype=np.uint8)
            watermark = cv2.imdecode(wm_bytes, cv2.IMREAD_COLOR)

        success_count = 0
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

        for filename in os.listdir(input_folder):
            if not filename.lower().endswith(valid_extensions):
                continue

            filepath = os.path.join(input_folder, filename)

            # Wczytywanie obrazu
            file_bytes = np.fromfile(filepath, dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                continue

            # 1. Zmiana rozdzielczości
            if target_size is not None:
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

            # 2. Znak wodny (prawy dolny róg)
            if watermark is not None:
                h_img, w_img = img.shape[:2]

                # Skalowanie znaku wodnego (np. do 20% szerokości obrazka)
                wm_w = int(w_img * 0.2)
                scale = wm_w / watermark.shape[1]
                wm_h = int(watermark.shape[0] * scale)

                if wm_w > 0 and wm_h > 0:
                    wm_resized = cv2.resize(watermark, (wm_w, wm_h), interpolation=cv2.INTER_AREA)

                    # Definiowanie obszaru (ROI) w prawym dolnym rogu (z małym marginesem)
                    margin = 10
                    y1, y2 = h_img - wm_h - margin, h_img - margin
                    x1, x2 = w_img - wm_w - margin, w_img - margin

                    # Nakładanie znaku wodnego (blending z przezroczystością 50%)
                    if y1 > 0 and x1 > 0:
                        roi = img[y1:y2, x1:x2]
                        blended = cv2.addWeighted(roi, 0.7, wm_resized, 0.5, 0)
                        img[y1:y2, x1:x2] = blended

            # Zapisywanie
            out_filepath = os.path.join(output_folder, filename)
            extension = filename.split('.')[-1]
            is_success, im_buf_arr = cv2.imencode(f'.{extension}', img)
            if is_success:
                im_buf_arr.tofile(out_filepath)
                success_count += 1

        return success_count

    # ==========================================
    # FUNKCJE AUTORSKIE
    # ==========================================
    def deep_fry_image(self):
        """Efekt smażenia (Deep Fry): wyostrzenie, saturacja, kontrast, jasność i szum."""
        if self.processed_image is None:
            return

        self._save_to_history()

        # Pracujemy na kopii w ramach tej funkcji
        img = self.processed_image.copy()

        # 1. Agresywne wyostrzanie (Sharpening)
        # Taka macierz mocno wyciąga krawędzie, tworząc ten specyficzny, "chrupiący" wygląd
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel_sharpen)

        # 2. Ekstremalna Saturacja (Przestrzeń barw HSV)
        # Konwertujemy na HSV, wyciągamy kanał S (nasycenie), podbijamy go i łączymy z powrotem
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 4.0)  # Zwiększamy nasycenie 3-krotnie
        s = np.clip(s, 0, 255).astype(np.uint8)  # Ucinamy wartości powyżej 255
        hsv = cv2.merge([h, s, v])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 3. Kontrast i Jasność (Przepalenie)
        # alpha to kontrast (np. 2.0 to 200%), beta to jasność (dodawana stała wartość)
        img = cv2.convertScaleAbs(img, alpha=2.0, beta=30)

        # 4. Dodanie szumu (Noise/Grain)
        # Generujemy losowe wartości i dodajemy do obrazu. Musimy użyć int16, żeby nie przekręcić licznika.
        noise = np.random.randint(-60, 60, img.shape, dtype='int16')
        img = np.clip(img.astype('int16') + noise, 0, 255).astype('uint8')

        # Zapisanie usmażonego obrazka
        self.processed_image = img

    # ==========================================
    # ANALIZA OBRAZU (Histogram)
    # ==========================================
    def get_histogram_data(self):
        """Oblicza rozkład pikseli dla kanałów B, G, R i zwraca jako listę."""
        if self.processed_image is None:
            return None

        hist_data = []
        # Kanały w OpenCV to kolejno B (0), G (1), R (2)
        for i in range(3):
            hist = cv2.calcHist([self.processed_image], [i], None, [256], [0, 256])
            hist_data.append(hist)

        return hist_data

    # ==========================================
    # DETEKCJA POZ (YOLOv8 Pose)
    # ==========================================
    def detect_pose(self):
        """Wykrywa sylwetki ludzi i rysuje na nich szkielet używając nowoczesnego YOLOv8."""
        if self.processed_image is None:
            return

        self._save_to_history()

        try:
            # Importujemy bibliotekę (wewnątrz funkcji, żeby nie spowalniać startu aplikacji)
            from ultralytics import YOLO

            # Pobranie i załadowanie małego modelu YOLOv8-pose (dzieje się automatycznie przy 1 uruchomieniu)
            if not hasattr(self, 'pose_model'):
                print("Ładowanie modelu YOLOv8 Pose... (to potrwa sekundę)")
                # 'yolov8n-pose.pt' to wersja Nano - superszybka i lekka
                self.pose_model = YOLO('yolov8n-pose.pt')

                # Uruchomienie wykrywania na naszym obrazie (wyłączamy wypisywanie logów w terminalu)
            results = self.pose_model(self.processed_image, verbose=False)

            # YOLOv8 ma wbudowaną, piękną funkcję rysującą wyniki od razu na obrazku!
            # result[0].plot() zwraca macierz BGR, gotową do użycia w OpenCV
            annotated_image = results[0].plot()

            self.processed_image = annotated_image
            print("Pomyślnie nałożono szkielety YOLOv8!")

        except Exception as e:
            print(f"Błąd podczas wykrywania poz YOLO: {e}")

    # ==========================================
    # OPTYMALNY TRANSPORT (PAPAJIFIKACJA 3.0 - Szum bez Blura)
    # ==========================================
    def papajify_image(self, callback=None, max_distance=20, iterations=15):
        """Miesza piksele malejącymi blokami, zachowując 100% oryginalnego histogramu."""
        if self.processed_image is None:
            return

        self._save_to_history()

        target_path = "jp2.jpg"
        if not os.path.exists(target_path):
            target_path = "jp2.png"
            if not os.path.exists(target_path):
                print("Błąd: Brak pliku jp2.jpg lub jp2.png w głównym folderze!")
                return

        src = self.processed_image.copy()
        tgt = cv2.imread(target_path)
        if tgt is None:
            return

        h, w = src.shape[:2]
        tgt = cv2.resize(tgt, (w, h))
        S = src.copy()

        T_luma = 0.114 * tgt[:, :, 0] + 0.587 * tgt[:, :, 1] + 0.299 * tgt[:, :, 2]

        # Tworzymy listę rozmiarów bloków, np. od 20 w dół aż do małego szumu (2x2 piksele)
        distances = np.linspace(max_distance, 4, iterations).astype(int)

        for i in range(iterations):
            curr_dist = distances[i]
            if curr_dist < 2: curr_dist = 2  # Zabezpieczenie rozmiaru

            # Losowe przesunięcie siatki, by unikać powtarzalnych wzorów
            offset_y = np.random.randint(0, curr_dist)
            offset_x = np.random.randint(0, curr_dist)

            for y in range(offset_y, h - curr_dist, curr_dist):
                for x in range(offset_x, w - curr_dist, curr_dist):

                    t_luma_block = T_luma[y:y + curr_dist, x:x + curr_dist]

                    # Ignorowanie czysto białego tła
                    if np.mean(t_luma_block) > 240:
                        continue

                    s_block = S[y:y + curr_dist, x:x + curr_dist]

                    s_flat = s_block.reshape(-1, 3)
                    t_luma_flat = t_luma_block.reshape(-1)

                    s_luma_flat = 0.114 * s_flat[:, 0] + 0.587 * s_flat[:, 1] + 0.299 * s_flat[:, 2]

                    s_sort_idx = np.argsort(s_luma_flat)
                    t_sort_idx = np.argsort(t_luma_flat)

                    new_s_flat = np.zeros_like(s_flat)
                    new_s_flat[t_sort_idx] = s_flat[s_sort_idx]

                    S[y:y + curr_dist, x:x + curr_dist] = new_s_flat.reshape(curr_dist, curr_dist, 3)

            # BRAK ROZMYCIA! Histogram pozostaje matematycznie nietknięty.
            self.processed_image = S.copy()

            if callback:
                callback(i + 1, iterations)