import cv2
import numpy as np
from PIL import Image
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as TF

class ImageProcessor:
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.history = []

        # Zmienne dla Sieci Neuronowej
        self.nn_model = None
        self.device = torch.device('cpu')  # W aplikacji okienkowej bezpieczniej użyć CPU, działa na każdym komputerze
        # Zaktualizowane rasy (ID zazwyczaj są przypisywane alfabetycznie przez Roboflow)
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
            # Ograniczamy historię np. do 10 kroków, żeby nie zabrać całej pamięci RAM
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
            self.history = [] # Czyścimy historię przy nowym obrazie
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
            return True  # Model już jest wczytany

        try:
            print("Wczytywanie modelu sieci neuronowej (to może chwilę potrwać)...")
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
            in_features = model.roi_heads.box_predictor.cls_score.in_features

            # Pamiętaj: 6 klas (5 ras + 1 tło)
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 6)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()  # Tryb inferencji (wykrywania)

            self.nn_model = model
            print("Model załadowany pomyślnie!")
            return True
        except Exception as e:
            print(f"Błąd ładowania modelu: {e}")
            return False

    def detect_cats_with_nn(self, threshold=0.6):
        """Przepuszcza aktualny obraz przez sieć i rysuje ramki (Bounding Boxes)."""
        if self.processed_image is None:
            return

        # Upewniamy się, że model jest wczytany
        if not self.load_nn_model():
            return

        self._save_to_history()

        # Przygotowanie obrazu dla PyTorch (OpenCV ma BGR, PyTorch lubi RGB)
        img_rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
        img_tensor = TF.to_tensor(img_rgb).unsqueeze(0).to(self.device)

        # Magia sieci neuronowej (Inferencja)
        with torch.no_grad():
            predictions = self.nn_model(img_tensor)[0]

        # Pobieranie wyników
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()

        # Rysowanie ramek na obrazie
        for box, label, score in zip(boxes, labels, scores):
            if score >= threshold:
                x_min, y_min, x_max, y_max = map(int, box)
                class_name = self.cat_classes.get(label, "Nieznany")

                # Czerwona ramka
                cv2.rectangle(self.processed_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)

                # Etykieta z rasą i pewnością (np. "Maine Coon: 95%")
                text = f"{class_name}: {int(score * 100)}%"

                # Tło dla tekstu (żeby był czytelny)
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(self.processed_image, (x_min, y_min - text_h - 10), (x_min + text_w, y_min),
                              (0, 0, 255), -1)
                cv2.putText(self.processed_image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
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

