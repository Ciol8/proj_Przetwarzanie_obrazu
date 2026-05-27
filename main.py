import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QProgressBar, QSlider,
                             QPushButton, QLabel, QFileDialog, QFrame, QSizePolicy, QDialog, QLineEdit, QMessageBox,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem) # <- DODANE
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt, QUrl
import time
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
import cv2
import pywinstyles
# Sound Effect by Abrar Hussain from Pixabay
# Integracja z Matplotlib w PyQt6
import matplotlib

matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from processor import ImageProcessor
from game_of_life import game_of_life_step


class InteractiveViewer(QGraphicsView):
    def __init__(self, main_app):
        super().__init__()
        self.app = main_app  # Zapisuje referencję do głównej aplikacji
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.photo = QGraphicsPixmapItem()
        self.scene.addItem(self.photo)

        # Ukrywamy paski przewijania
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Płynne przesuwanie obrazu (Pan) łapiąc lewym przyciskiem myszy
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # Tło przezroczyste, żeby widzieć matowy panel pod spodem
        self.setBackgroundBrush(Qt.GlobalColor.transparent)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def set_image(self, pixmap, is_new_image=False):
        self.photo.setPixmap(pixmap)
        # Przy nowym obrazku dopasowujemy go do okna, w przeciwnym razie zostawiamy obecny zoom
        if is_new_image or self.transform().m11() == 1.0:
            self.fitInView(self.photo.boundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        """Obsługa przybliżania (Zoom) kółkiem myszy."""
        if self.photo.pixmap().isNull():
            return

        # Obliczanie kierunku kręcenia rolką
        if event.angleDelta().y() > 0:
            zoom_factor = 1.15  # Przybliż
        else:
            zoom_factor = 1.0 / 1.15  # Oddal

        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        """Obsługa naciśnięcia prawego przycisku myszy - pokazuje oryginalny obraz."""
        if event.button() == Qt.MouseButton.RightButton:
            self.app.show_original_preview(True)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Obsługa zwolnienia prawego przycisku myszy - wraca do przetwarzanego obrazu."""
        if event.button() == Qt.MouseButton.RightButton:
            self.app.show_original_preview(False)
        super().mouseReleaseEvent(event)

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = ImageProcessor()
        self.last_dir = ""  # Zapamiętywanie ostatniego folderu
        self.gol_running = False  # Kontrola działania Easter Egga

        # Inicjalizacja odtwarzacza dźwięku dla efektu smażenia
        self.audio_output = QAudioOutput()
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio_output)
        self.player.setSource(QUrl.fromLocalFile("oil_sfx.mp3"))

        self.setWindowTitle("Projekt - Detekcja i Interpretacja Obiektów")
        self.resize(1200, 900)
        self.setAcceptDrops(True)  # Włączenie obsługi Drag & Drop

        aero_font = QFont("Trebuchet MS", 10, QFont.Weight.Bold)
        QApplication.setFont(aero_font)

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        pywinstyles.apply_style(self, "aero")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Tło całej aplikacji - lekko przezroczysta, morska zieleń
        central_widget.setStyleSheet("background-color: rgba(80, 143, 91, 50);")

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # ==========================================
        # FUNKCJE TWORZENIA PRZYCISKÓW Z TEKSTURAMI
        # ==========================================
        def create_btn(text, command, texture_file):
            btn = QPushButton(text)
            btn.clicked.connect(command)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(f"Kliknij, aby {text.lower()}")

            # Włączamy antialiasing dla tekstu
            font = btn.font()
            font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
            btn.setFont(font)

            if os.path.exists(texture_file):
                # Aby naprawić problem poszarpanych rogów przy użyciu border-image,
                # dodajemy mały trik: ustawiamy tło na 'transparent' i wymuszamy obcięcie zawartości
                btn.setStyleSheet(f"""
                    QPushButton {{
                        color: black;
                        font-weight: bold;
                        border-image: url({texture_file}) 0 0 0 0 stretch stretch;
                        background-color: transparent; 
                        border-radius: 15px;
                        padding: 8px 15px;
                        min-height: 20px;
                        outline: none; /* Usuwa poszarpaną ramkę focusa */
                    }}
                    QPushButton:hover {{
                        color: #E0F7FA;
                    }}
                    QPushButton:pressed {{
                        padding-top: 10px;
                        padding-left: 17px;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: rgba(41, 128, 185, 180);
                        color: black;
                        border: 1px solid rgba(255, 255, 255, 80);
                        border-radius: 15px;
                        padding: 8px 15px;
                        font-weight: bold;
                    }}
                """)
            return btn

        # ==========================================
        # GÓRNY PASEK
        # ==========================================
        self.top_frame = QFrame()
        self.top_frame.setFixedHeight(65)
        self.top_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(34, 166, 209, 100); 
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 40);
            }
        """)
        top_layout = QHBoxLayout(self.top_frame)
        top_layout.setContentsMargins(15, 5, 15, 5)
        top_layout.setSpacing(15)

        # Używamy niebieskiego dla I/O
        self.btn_load = create_btn("Wczytaj Obraz", self.open_file, "btn_blue.png")
        self.btn_save = create_btn("Zapisz Obraz", self.save_file, "btn_blue.png")
        top_layout.addWidget(self.btn_load)
        top_layout.addWidget(self.btn_save)

        # Zamiast samego top_layout.addStretch() wstaw to:
        top_layout.addStretch()

        # Tworzymy pasek postępu Frutiger Aero
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(180)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
                    QProgressBar {
                        border: 2px solid rgba(255, 255, 255, 50);
                        border-radius: 8px;
                        background-color: rgba(0, 0, 0, 50);
                    }
                    QProgressBar::chunk {
                        background-color: #2ECC71; /* Zielony pasek */
                        border-radius: 6px;
                    }
                """)
        self.progress_bar.hide()  # Ukryty, dopóki nic się nie przetwarza
        top_layout.addWidget(self.progress_bar)

        top_layout.addStretch()

        # Używamy żółtego/pomarańczowego dla ostrzegawczych operacji
        self.btn_undo = create_btn("Cofnij (Undo)", self.undo, "btn_yellow.png")
        self.btn_reset = create_btn("Resetuj", self.reset_image, "btn_yellow.png")
        top_layout.addWidget(self.btn_undo)
        top_layout.addWidget(self.btn_reset)

        main_layout.addWidget(self.top_frame)

        # ==========================================
        # OBSZAR ROBOCZY (Panel boczny + Obraz)
        # ==========================================
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)

        # --- PANEL BOCZNY ---
        sidebar_container = QFrame()
        sidebar_container.setFixedWidth(280)
        sidebar_container.setStyleSheet("""
            QFrame {
                background-color: rgba(34, 166, 209, 80); 
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 30);
            }
        """)

        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(15, 15, 15, 15)
        sidebar_layout.setSpacing(8)

        logo = QLabel("NARZĘDZIA")
        logo.setStyleSheet(
            "color: white; font-weight: bold; font-size: 16px; background: transparent; border: none; letter-spacing: 2px;")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(logo)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background: rgba(255,255,255,50); border: none; height: 1px;")
        sidebar_layout.addWidget(line)
        sidebar_layout.addSpacing(5)

        # --- ROZWIJANE MENU KRAWĘDZIE (Zielone) ---
        self.edges_visible = False
        self.btn_edges_toggle = create_btn("Krawędzie ▼", self.toggle_edges, "btn_green.png")
        sidebar_layout.addWidget(self.btn_edges_toggle)

        self.frame_edges = QFrame()
        self.frame_edges.setStyleSheet("background: transparent; border: none;")
        edges_layout = QVBoxLayout(self.frame_edges)
        edges_layout.setContentsMargins(15, 0, 0, 0)  # Lekkie wcięcie dla sub-przycisków
        edges_layout.setSpacing(4)

        # Zmodyfikowane pod run_with_progress:
        edges_layout.addWidget(
            create_btn("Canny", lambda: self.run_with_progress(lambda: self.apply_op("canny")), "btn_green.png"))
        edges_layout.addWidget(
            create_btn("Sobel", lambda: self.run_with_progress(lambda: self.apply_op("sobel")), "btn_green.png"))
        edges_layout.addWidget(
            create_btn("Laplacian", lambda: self.run_with_progress(lambda: self.apply_op("laplace")), "btn_green.png"))
        self.btn_gol = create_btn("Uruchom Game of Life ❖", self.start_game_of_life, "glider.png")
        edges_layout.addWidget(self.btn_gol)
        self.btn_gol.hide()  # Domyślnie przycisk jest niewidoczny

        sidebar_layout.addWidget(self.frame_edges)
        self.frame_edges.hide()  # Domyślnie schowane

        # --- ROZWIJANE MENU PROGOWANIE (Zielone) ---
        self.thresh_visible = False
        self.btn_thresh_toggle = create_btn("Progowanie ▼", self.toggle_thresh, "btn_green.png")
        sidebar_layout.addWidget(self.btn_thresh_toggle)

        self.frame_thresh = QFrame()
        self.frame_thresh.setStyleSheet("background: transparent; border: none;")
        thresh_layout = QVBoxLayout(self.frame_thresh)
        thresh_layout.setContentsMargins(15, 0, 0, 0)
        thresh_layout.setSpacing(4)

        # Zmodyfikowane pod run_with_progress:
        thresh_layout.addWidget(
            create_btn("Binarne", lambda: self.run_with_progress(lambda: self.apply_op("thresh_bin")), "btn_green.png"))
        thresh_layout.addWidget(
            create_btn("Otsu", lambda: self.run_with_progress(lambda: self.apply_op("thresh_otsu")), "btn_green.png"))
        thresh_layout.addWidget(
            create_btn("Adaptacyjne", lambda: self.run_with_progress(lambda: self.apply_op("thresh_adapt")),
                       "btn_green.png"))

        sidebar_layout.addWidget(self.frame_thresh)
        self.frame_thresh.hide()  # Domyślnie schowane

        sidebar_layout.addSpacing(15)

        # AI i Autorskie (Fioletowe)
        sidebar_layout.addWidget(QLabel("SZTUCZNA INTELIGENCJA",
                                        styleSheet="color: #E0F7FA; font-weight: bold; font-size: 11px; background: transparent; border: none;"))

        # --- NOWY ELEMENT: Suwak Threshold ---
        slider_layout = QHBoxLayout()
        self.lbl_thresh = QLabel("Próg AI: 60%")
        self.lbl_thresh.setStyleSheet("color: white; background: transparent; border: none; font-size: 11px;")

        self.slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh.setRange(10, 100)  # Skala od 10% do 100%
        self.slider_thresh.setValue(60)  # Domyślnie 60%
        self.slider_thresh.setStyleSheet("QSlider { background: transparent; }")
        self.slider_thresh.valueChanged.connect(lambda val: self.lbl_thresh.setText(f"Próg AI: {val}%"))

        slider_layout.addWidget(self.lbl_thresh)
        slider_layout.addWidget(self.slider_thresh)
        sidebar_layout.addLayout(slider_layout)
        # -------------------------------------

        sidebar_layout.addWidget(
            create_btn("Wykryj Koty", lambda: self.run_with_progress(self.run_nn_detection), "btn_purple.png"))

        # --- NOWY ELEMENT: Tekst z wynikiem klasyfikacji w GUI ---
        self.lbl_ai_result = QLabel("")
        self.lbl_ai_result.setStyleSheet(
            "color: #F1C40F; font-weight: bold; font-size: 11px; background: transparent; border: none;")
        self.lbl_ai_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_ai_result.setWordWrap(True)
        sidebar_layout.addWidget(self.lbl_ai_result)
        # -------------------------------------

        sidebar_layout.addWidget(
            create_btn("Wykryj Pozy", lambda: self.run_with_progress(self.run_pose_detection), "btn_purple.png"))

        # Narzędzia (Żółte)
        sidebar_layout.addWidget(QLabel("INNE NARZĘDZIA",
                                        styleSheet="color: #E0F7FA; font-weight: bold; font-size: 11px; background: transparent; border: none;"))# Usunięto run_with_progress, bo run_deep_fry ma teraz własną animację
        sidebar_layout.addWidget(create_btn("Usmaż Obraz 🍟", self.run_deep_fry, "btn_yellow.png"))
        sidebar_layout.addWidget(create_btn("Przetwarzanie Wsadowe", self.open_batch_window, "btn_yellow.png"))
        # Wklej to w sekcji żółtych przycisków w __init__
        sidebar_layout.addWidget(create_btn("JanPaweł-ifikacja 💛", self.start_papajify, "btn_yellow.png"))

        sidebar_layout.addStretch()
        content_layout.addWidget(sidebar_container)

        # --- PANEL PRAWY (Obraz + Histogram) ---
        right_panel = QWidget()
        right_panel.setStyleSheet("background: transparent;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(15)

        # 1. KONTENER NA OBRAZ (Jasny, matowy, z zaokrąglonymi rogami)
        self.image_container = QFrame()
        self.image_container.setStyleSheet("""
                    QFrame {
                        background-color: rgba(240, 240, 240, 255); 
                        border-radius: 15px;
                        border: 2px solid rgba(255, 255, 255, 100);
                    }
                """)
        image_layout = QVBoxLayout(self.image_container)
        image_layout.setContentsMargins(10, 10, 10, 10)

        # ---- NOWY WIDOK ZAMIAST QLabel ----
        self.viewer = InteractiveViewer(self)  # Przekazujemy referencję do głównej aplikacji
        image_layout.addWidget(self.viewer)
        # -----------------------------------

        right_layout.addWidget(self.image_container, stretch=4)

        # 2. Kontener na Histogram
        self.hist_container = QFrame()
        self.hist_container.setFixedHeight(180)
        self.hist_container.setStyleSheet("""
            QFrame {
                background-color: rgba(34, 166, 209, 100); 
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 30);
            }
        """)
        hist_layout = QVBoxLayout(self.hist_container)
        hist_layout.setContentsMargins(5, 5, 5, 5)

        self.figure = Figure(figsize=(5, 1.5), dpi=100)
        self.figure.patch.set_alpha(0.0)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('none')
        self.ax.tick_params(colors='white', labelsize=8)

        self.ax.grid(True, linestyle='-', alpha=0.1, color='white')
        for spine in self.ax.spines.values():
            spine.set_edgecolor((1.0, 1.0, 1.0, 0.2))

        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setStyleSheet("background: transparent;")
        hist_layout.addWidget(self.canvas)

        right_layout.addWidget(self.hist_container, stretch=1)

        content_layout.addWidget(right_panel)
        main_layout.addLayout(content_layout)

    # ==========================================
    # LOGIKA ROZWIJANYCH MENU
    # ==========================================
    def toggle_edges(self):
        self.edges_visible = not self.edges_visible
        if self.edges_visible:
            self.frame_edges.show()
            self.btn_edges_toggle.setText("Krawędzie ▲")
        else:
            self.frame_edges.hide()
            self.btn_edges_toggle.setText("Krawędzie ▼")

    def toggle_thresh(self):
        self.thresh_visible = not self.thresh_visible
        if self.thresh_visible:
            self.frame_thresh.show()
            self.btn_thresh_toggle.setText("Progowanie ▲")
        else:
            self.frame_thresh.hide()
            self.btn_thresh_toggle.setText("Progowanie ▼")

    # ==========================================
    # LOGIKA APLIKACJI
    # ==========================================
    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Wybierz obraz", self.last_dir, "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path and self.processor.load_image(file_path):
            self.last_dir = os.path.dirname(file_path)  # Zapisujemy nowy folder
            self.update_display(is_new_image=True)

    def save_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Zapisz obraz", self.last_dir, "PNG (*.png);;JPEG (*.jpg)")
        if file_path:
            self.last_dir = os.path.dirname(file_path)  # Zapisujemy nowy folder
            self.processor.save_image(file_path)

    def run_with_progress(self, task_function):
        self.gol_running = False  # Zatrzymuje symulację, jeśli działała
        self.btn_gol.hide()       # Ukrywa przycisk Easter Egga
        """Pokazuje pasek, wymusza odświeżenie ekranu, puszcza AI/Filtr i chowa pasek."""
        # setRange(0, 0) sprawia, że pasek "lata" w lewo i prawo (tzw. indeterminate state)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()

        # WYMUSZENIE ODŚWIEŻENIA UI (bez tego aplikacja zamarznie zanim pokaże pasek!)
        QApplication.processEvents()

        # Odpalenie docelowej funkcji (np. sieci neuronowej)
        task_function()

        # Ukrycie paska po zakończeniu
        self.progress_bar.hide()

    # ==========================================
    # PRZETWARZANIE WSADOWE (DODANE)
    # ==========================================
    def open_batch_window(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Przetwarzanie Wsadowe")
        dialog.resize(400, 350)
        # Nadajemy oknu dialogowemu podobny styl
        dialog.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        pywinstyles.apply_style(dialog, "aero")
        dialog.setStyleSheet("QDialog { background-color: rgba(34, 166, 209, 200); border-radius: 15px; }")

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)

        # Style dla małych elementów
        label_style = "color: white; font-weight: bold; font-size: 14px;"
        input_style = "background-color: rgba(255,255,255,50); color: white; border: 1px solid white; border-radius: 5px; padding: 5px;"

        # 1. Folder wejściowy
        layout.addWidget(QLabel("1. Folder źródłowy:", styleSheet=label_style))
        lbl_in = QLabel("Brak")
        lbl_in.setStyleSheet("color: #E0F7FA;")
        layout.addWidget(lbl_in)

        self.batch_input = ""

        def set_in():
            folder = QFileDialog.getExistingDirectory(dialog, "Wybierz folder")
            if folder:
                self.batch_input = folder
                lbl_in.setText(os.path.basename(folder))

        btn_in = QPushButton("Wybierz folder")
        btn_in.setToolTip("Wybierz folder źródłowy do przetwarzania wsadowego")
        btn_in.setStyleSheet("background-color: rgba(255,255,255,30); color: white; border-radius: 5px; padding: 5px;")
        btn_in.clicked.connect(set_in)
        layout.addWidget(btn_in)
        layout.addSpacing(10)

        # 2. Znak wodny
        layout.addWidget(QLabel("2. Znak wodny (opcjonalnie):", styleSheet=label_style))
        lbl_wm = QLabel("Brak")
        lbl_wm.setStyleSheet("color: #E0F7FA;")
        layout.addWidget(lbl_wm)

        self.batch_watermark = ""

        def set_wm():
            file, _ = QFileDialog.getOpenFileName(dialog, "Wybierz plik", "", "Images (*.png *.jpg)")
            if file:
                self.batch_watermark = file
                lbl_wm.setText(os.path.basename(file))

        btn_wm = QPushButton("Wybierz plik")
        btn_wm.setToolTip("Wybierz plik z znakiem wodnym do dodania")
        btn_wm.setStyleSheet("background-color: rgba(255,255,255,30); color: white; border-radius: 5px; padding: 5px;")
        btn_wm.clicked.connect(set_wm)
        layout.addWidget(btn_wm)
        layout.addSpacing(10)

        # 3. Rozdzielczość
        layout.addWidget(QLabel("3. Zmiana rozdzielczości (opcjonalnie):", styleSheet=label_style))
        res_layout = QHBoxLayout()
        entry_w = QLineEdit()
        entry_w.setPlaceholderText("Szerokość")
        entry_w.setStyleSheet(input_style)
        entry_h = QLineEdit()
        entry_h.setPlaceholderText("Wysokość")
        entry_h.setStyleSheet(input_style)
        res_layout.addWidget(entry_w)
        res_layout.addWidget(entry_h)
        layout.addLayout(res_layout)

        layout.addStretch()

        # START
        def start_batch():
            if not self.batch_input:
                QMessageBox.warning(dialog, "Błąd", "Wybierz folder źródłowy!")
                return

            out_dir = QFileDialog.getExistingDirectory(dialog, "Wybierz folder docelowy")
            if out_dir:
                try:
                    w = int(entry_w.text()) if entry_w.text() else None
                    h = int(entry_h.text()) if entry_h.text() else None
                    size = (w, h) if w and h else None
                except ValueError:
                    size = None
                wm = self.batch_watermark if self.batch_watermark else None

                count = self.processor.process_batch(self.batch_input, out_dir, size, wm)
                QMessageBox.information(dialog, "Sukces", f"Zapisano {count} plików w folderze docelowym.")
                dialog.accept()

        btn_start = QPushButton("ROZPOCZNIJ PRZETWARZANIE")
        btn_start.setToolTip("Rozpocznij przetwarzanie wsadowe zaznaczonego folderu")
        btn_start.setStyleSheet(
            "background-color: rgba(46, 204, 113, 200); color: white; border-radius: 8px; padding: 10px; font-weight: bold;")
        btn_start.clicked.connect(start_batch)
        layout.addWidget(btn_start)

        dialog.exec()

    def undo(self):
        self.gol_running = False  # Zatrzymuje symulację
        self.btn_gol.hide()
        if self.processor.undo():
            self.update_display()

    def reset_image(self):
        self.gol_running = False  # Zatrzymuje symulację
        self.btn_gol.hide()
        self.processor.reset_image()
        self.update_display()

    def apply_op(self, op_type):
        if self.processor.original_image is None: return

        if op_type == "canny":
            self.processor.detect_edges_canny()
            self.btn_gol.show()  # Pokazujemy przycisk po kliknięciu Canny
        else:
            self.btn_gol.hide()  # Chowamy dla Sobela / Laplaciana
            self.gol_running = False

        if op_type == "sobel":
            self.processor.detect_edges_sobel()
        elif op_type == "laplace":
            self.processor.detect_edges_laplacian()
        elif op_type == "thresh_bin":
            self.processor.threshold_binary()
        elif op_type == "thresh_otsu":
            self.processor.threshold_otsu()
        elif op_type == "thresh_adapt":
            self.processor.threshold_adaptive()
        self.update_display()

    def run_nn_detection(self):
        if self.processor.original_image is not None:
            # 1. Pobieramy próg z suwaka z GUI (np. 60 z 100 zamieniamy na 0.6)
            current_threshold = self.slider_thresh.value() / 100.0

            # 2. Uruchamiamy AI przekazując nasz próg i odbieramy listę wyników
            wyniki = self.processor.detect_cats_with_nn(threshold=current_threshold)

            # 3. Wyświetlamy tekst w GUI pod przyciskiem
            if wyniki:
                tekst = "\n".join(wyniki)  # Jeśli wykryto kilka kotów, każdy w nowej linii
                self.lbl_ai_result.setText(f"WYKRYTO:\n{tekst}")
            else:
                self.lbl_ai_result.setText("Nie wykryto obiektów")

            self.update_display()

    def run_pose_detection(self):
        if self.processor.original_image is not None:
            self.processor.detect_pose()
            self.update_display()

    def run_deep_fry(self):
        """Smaży obraz natychmiast, ale animuje przejście (fade-in) i odtwarza dźwięk przez 3 sekundy."""
        if self.processor.processed_image is None:
            return

        # 1. Kopiujemy oryginalny obraz do pamięci przed modyfikacją
        orig_img = self.processor.processed_image.copy()

        # 2. Smażymy obraz OD RAZU w procesorze i zapisujemy wynik do zmiennej
        self.processor.deep_fry_image()
        fried_img = self.processor.processed_image.copy()

        # 3. Przygotowujemy pasek postępu u góry na 60 klatek animacji
        steps = 60
        delay = 0.05  # 50 milisekund na klatkę (60 klatek * 0.05s = dokładnie 3 sekundy)

        self.progress_bar.setRange(0, steps)
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        # 4. Odpalamy dźwięk skwierczenia oleju
        self.player.setPosition(0)  # Przewiń do początku pliku
        self.player.play()

        # 5. Pętla animacji: płynnie zmieniamy alfa od 1.0 (oryginał) do 0.0 (usmażony)
        for step in range(steps + 1):
            alpha = 1.0 - (step / steps)

            # Blendowanie: oryginalny_obraz * alpha + usmażony_obraz * (1 - alpha)
            blended = cv2.addWeighted(orig_img, alpha, fried_img, 1.0 - alpha, 0)

            # Podmieniamy obraz w procesorze tylko "na chwilę" do wyświetlenia klatki
            self.processor.processed_image = blended
            self.update_display()

            # Aktualizacja paska postępu
            self.progress_bar.setValue(step)

            # Wymuszenie na PyQt6 natychmiastowego przerysowania ekranu i sen
            QApplication.processEvents()
            time.sleep(delay)

        # 6. Po 3 sekundach upewniamy się, że zostaje czysty, usmażony obraz, i sprzątamy GUI
        self.processor.processed_image = fried_img
        self.update_display()
        self.progress_bar.hide()
        self.player.stop()  # Zatrzymujemy dźwięk po 3 sekundach

    def update_display(self, is_new_image=False):  # <- Dodany argument
        img_bgr = self.processor.processed_image
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            # Przekazujemy CZYSTY, pełny obraz do naszego interaktywnego podglądu!
            self.viewer.set_image(pixmap, is_new_image)

        hist_data = self.processor.get_histogram_data()

        # ... reszta update_display zostaje bez zmian (histogram) ...
        if hist_data is not None:
            self.ax.clear()
            self.ax.grid(True, linestyle='-', alpha=0.1, color='white')

            for spine in self.ax.spines.values():
                spine.set_edgecolor((1.0, 1.0, 1.0, 0.2))
            self.ax.tick_params(colors='white')

            colors = ('#3498db', '#2ecc71', '#e74c3c')
            for i, col in enumerate(colors):
                self.ax.plot(hist_data[i], color=col, linewidth=2.0)
                self.ax.fill_between(range(256), hist_data[i].flatten(), color=col, alpha=0.15)
                self.ax.set_xlim([0, 256])

            self.figure.tight_layout()
            self.canvas.draw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.processor.processed_image is not None:
            self.update_display()

    # ==========================================
    # LOGIKA PĘTLI PAPAJIFIKACJI W LOCIE
    # ==========================================
    def start_papajify(self):
        """Przygotowuje okno i odpala procesor."""
        if self.processor.processed_image is None:
            return

        ilosc_krokow = 20  # Możesz zmienić na ile chcesz

        # Przygotowujemy pasek na konkretny zakres (0 do 12) zamiast latać w lewo/prawo
        self.progress_bar.setRange(0, ilosc_krokow)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        QApplication.processEvents()

        # Uruchamiamy algorytm i przekazujemy naszą metodę jako callback
        self.processor.papajify_image(callback=self.papajify_step_callback, iterations=ilosc_krokow)

        # Po zakończeniu pętli chowamy pasek
        self.progress_bar.hide()

    def papajify_step_callback(self, aktualna_iteracja, wszystkie_iteracje):
        """Ta funkcja wykonuje się automatycznie po KAŻDEJ iteracji w procesorze."""
        # 1. Aktualizujemy wartość paska postępu (np. wskakuje na 3 z 12)
        self.progress_bar.setValue(aktualna_iteracja)

        # 2. Odświeżamy etykietę ze zdjęciem oraz histogram na ekranie
        self.update_display()

        # 3. KLUCZ: Wymuszamy na systemie Windows natychmiastowe przerysowanie okna
        QApplication.processEvents()

    # ==========================================
    # PODGLĄD PRZED/PO (PRAWY PRZYCISK MYSZY)
    # ==========================================
    def show_original_preview(self, show_original):
        """Tymczasowo wyświetla oryginalny obraz, dopóki trzymany jest prawy przycisk myszy."""
        if self.processor.original_image is None:
            return

        # Zależnie od tego, czy przycisk jest wciśnięty, pobieramy oryginał lub przetworzony
        img_bgr = self.processor.original_image if show_original else self.processor.processed_image
        
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # Aktualizujemy obrazek na podglądzie (is_new_image=False, żeby nie resetował zooma!)
            self.viewer.set_image(pixmap, is_new_image=False)

    # ==========================================
    # OBSŁUGA DRAG & DROP
    # ==========================================
    def dragEnterEvent(self, event):
        """Sprawdza, czy przeciągany plik to obrazek, zanim pozwolimy go upuścić."""
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile() and url.toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event):
        """Wczytuje obraz po upuszczeniu go na okno programu."""
        file_path = event.mimeData().urls()[0].toLocalFile()
        if self.processor.load_image(file_path):
            self.last_dir = os.path.dirname(file_path)  # Zapamiętujemy folder!
            self.update_display(is_new_image=True)

    # ==========================================
    # EASTER EGG: CONWAY'S GAME OF LIFE
    # ==========================================
    def start_game_of_life(self):
        """Uruchamia automatyczną symulację gry w życie na bazie krawędzi Canny."""
        if self.processor.processed_image is None:
            return

        # Zapisujemy aktualny stan czystego Canny do historii.
        # Dzięki temu kliknięcie "Undo" idealnie przywróci czarno-biały obraz!
        self.processor._save_to_history()
        
        self.gol_running = True
        
        while self.gol_running:
            img = self.processor.processed_image
            if img is None:
                break
                
            # Konwertujemy 3-kanałowy obraz BGR z procesora do 1 kanału (Grayscale)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Obliczamy następny krok algorytmu Conwaya
            next_gray = game_of_life_step(gray)
            
            # Konwertujemy z powrotem do BGR i nadpisujemy obraz w procesorze
            self.processor.processed_image = cv2.cvtColor(next_gray, cv2.COLOR_GRAY2BGR)
            
            # Odświeżamy widok ekranu oraz histogram
            self.update_display()
            
            # Wymuszamy na PyQt6 przetworzenie zdarzeń systemowych (aby kliknięcie "Undo" mogło przerwać pętlę!)
            QApplication.processEvents()
            
            # Prędkość ewolucji - 0.1 sekundy na każdą generację komórek
            time.sleep(0.1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())