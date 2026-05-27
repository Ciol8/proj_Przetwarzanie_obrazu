import time

import cv2
import numpy as np
from PIL import Image
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # <--- DODAJ TĘ LINIJKĘ
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
    # OPTYMALNY TRANSPORT & PERFEKCYJNE DOPASOWANIE
    # ==========================================
    def papajify_image(self, callback=None, iterations=15):
        """Używa AI do idealnego dopasowania twarzy ze specjalnego pliku (z obrotem o 15° CCW), a następnie w tym miejscu wykonuje sortowanie pikseli."""
        if self.processed_image is None:
            return

        self._save_to_history()

        # 1. Wczytanie obrazu GŁÓWNEGO dla Optymalnego Transportu (brak twarzy)
        target_path = "jp2.jpg"
        if not os.path.exists(target_path):
            target_path = "jp2.png"
            if not os.path.exists(target_path):
                print("Błąd: Brak pliku jp2.jpg lub jp2.png w głównym folderze!")
                return
        tgt = cv2.imread(target_path)
        if tgt is None: 
            return

        # 2. Wczytanie DEDYKOWANEJ twarzy dla Face Swapu (jeśli AI kogoś znajdzie)
        target_face_path = "jp2twarz.jpg"
        if not os.path.exists(target_face_path):
            target_face_path = "jp2twarz.png"
        
        # Jeśli masz plik z twarzą to go użyje, jak nie - awaryjnie użyje zwykłego jp2.jpg
        if os.path.exists(target_face_path):
            tgt_face = cv2.imread(target_face_path)
        else:
            tgt_face = tgt.copy()

        src = self.processed_image.copy()
        h_img, w_img = src.shape[:2]

        # --- DODANE: Dynamiczne obliczanie rozmiaru bloków ---
        max_dim = max(h_img, w_img)
        dyn_max_dist = int(max_dim * 0.10)  # 10% najdłuższego boku
        # 1% boku, ale zabezpieczamy funkcją max(), by nigdy nie spadło poniżej 2 pikseli
        dyn_min_dist = max(int(max_dim * 0.01), 2)
        # -----------------------------------------------------

        try:
            from ultralytics import YOLO
            if not hasattr(self, 'pose_model'):
                print("Ładowanie modelu YOLOv8 Pose do detekcji twarzy...")
                self.pose_model = YOLO('yolov8n-pose.pt')

            # Szukamy twarzy na zdjęciu, które chcemy przerobić
            results_src = self.pose_model(src, verbose=False)
            kpts_src = results_src[0].keypoints

            twarze_znalezione = False
            
            # Tworzymy puste, CZYSTO BIAŁE płótno. Posłuży jako wzór dla pikseli.
            T_canvas = np.full((h_img, w_img, 3), 255, dtype=np.uint8)

            # Bezpieczne sprawdzenie czy wykryto człowieka
            if kpts_src is not None and len(kpts_src) > 0 and hasattr(kpts_src, 'xy') and kpts_src.xy.shape[1] >= 3:
                
                # Szukamy oczu i nosa na DEDYKOWANYM zdjęciu twarzy
                results_tgt = self.pose_model(tgt_face, verbose=False)
                kpts_tgt = results_tgt[0].keypoints
                
                h_tgt, w_tgt = tgt_face.shape[:2]
                pts_jp2 = np.float32([[w_tgt*0.5, h_tgt*0.6], [w_tgt*0.65, h_tgt*0.4], [w_tgt*0.35, h_tgt*0.4]])
                
                if kpts_tgt is not None and len(kpts_tgt) > 0 and kpts_tgt.xy.shape[1] >= 3:
                    jp2_pts = kpts_tgt.xy[0].cpu().numpy()
                    if jp2_pts[0][0] > 0 and jp2_pts[1][0] > 0 and jp2_pts[2][0] > 0:
                        pts_jp2 = np.float32([jp2_pts[0], jp2_pts[1], jp2_pts[2]])

                # Tworzymy miękką, eliptyczną maskę dla wyciętej twarzy
                jp2_mask = np.zeros((h_tgt, w_tgt), dtype=np.uint8)
                cv2.ellipse(jp2_mask, (int(w_tgt*0.5), int(h_tgt*0.5)), (int(w_tgt*0.4), int(h_tgt*0.5)), 0, 0, 360, 255, -1)
                jp2_mask = cv2.GaussianBlur(jp2_mask, (21, 21), 0)

                for i in range(len(kpts_src.xy)):
                    pts = kpts_src.xy[i].cpu().numpy()
                    if len(pts) >= 3:
                        nose = pts[0]; le = pts[1]; re = pts[2]
                        if nose[0] > 0 and le[0] > 0 and re[0] > 0:
                            twarze_znalezione = True
                            
                            # ===============================================
                            # MAGIA: OBRÓT O 15 STOPNI CCW (W lewo) WOKÓŁ NOSA
                            # ===============================================
                            angle = 5  # Wartość dodatnia w OpenCV to obrót Counter-Clockwise (CCW)
                            R = cv2.getRotationMatrix2D((float(nose[0]), float(nose[1])), angle, 1.0)
                            
                            # Przekształcamy oryginalne punkty oczu za pomocą macierzy obrotu
                            pts_eyes = np.array([le, re])
                            pts_ones = np.hstack([pts_eyes, np.ones((2, 1))])
                            rotated_eyes = R.dot(pts_ones.T).T
                            
                            # Budujemy nowy trójkąt: Oryginalny Nos + Obrócone Oczy
                            pts_img = np.float32([nose, rotated_eyes[0], rotated_eyes[1]])
                            # ===============================================

                            M = cv2.getAffineTransform(pts_jp2, pts_img)
                            
                            # Nakładanie zniekształconej i obróconej twarzy na białe płótno
                            warped_jp2 = cv2.warpAffine(tgt_face, M, (w_img, h_img), borderValue=(255,255,255))
                            warped_mask = cv2.warpAffine(jp2_mask, M, (w_img, h_img))
                            
                            mask_3ch = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR) / 255.0
                            T_canvas = (T_canvas * (1.0 - mask_3ch) + warped_jp2 * mask_3ch).astype(np.uint8)

            if not twarze_znalezione:
                print("Brak twarzy ludzkich. Zastosowanie filtra na całym oryginalnym jp2.")
                T_canvas = cv2.resize(tgt, (w_img, h_img))
                
        except Exception as e:
            print(f"Błąd AI: {e}. Wracam do trybu pełnoekranowego.")
            T_canvas = cv2.resize(tgt, (w_img, h_img))

        # ========================================================
        # WŁAŚCIWY ALGORYTM MIESZANIA PIKSELI Z PASKIEM POSTĘPU
        # ========================================================
        S = src.copy()
        T_luma = 0.114 * T_canvas[:,:,0] + 0.587 * T_canvas[:,:,1] + 0.299 * T_canvas[:,:,2]
        # Używamy naszych dynamicznych zmiennych zamiast wpisanych na sztywno liczb
        distances = np.linspace(dyn_max_dist, dyn_min_dist, iterations).astype(int)

        for i in range(iterations):
            curr_dist = distances[i]
            if curr_dist < dyn_min_dist: curr_dist = dyn_min_dist

            offset_y = np.random.randint(0, curr_dist)
            offset_x = np.random.randint(0, curr_dist)

            for y in range(offset_y, h_img - curr_dist, curr_dist):
                for x in range(offset_x, w_img - curr_dist, curr_dist):
                    
                    t_luma_block = T_luma[y:y+curr_dist, x:x+curr_dist]
                    
                    if np.mean(t_luma_block) > 240:
                        continue
                        
                    s_block = S[y:y+curr_dist, x:x+curr_dist]
                    s_flat = s_block.reshape(-1, 3)
                    t_luma_flat = t_luma_block.reshape(-1)

                    s_luma_flat = 0.114 * s_flat[:,0] + 0.587 * s_flat[:,1] + 0.299 * s_flat[:,2]

                    s_sort_idx = np.argsort(s_luma_flat)
                    t_sort_idx = np.argsort(t_luma_flat)

                    new_s_flat = np.zeros_like(s_flat)
                    new_s_flat[t_sort_idx] = s_flat[s_sort_idx]

                    S[y:y+curr_dist, x:x+curr_dist] = new_s_flat.reshape(curr_dist, curr_dist, 3)

            self.processed_image = S.copy()

            if callback:
                callback(i + 1, iterations)