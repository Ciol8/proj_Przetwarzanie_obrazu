import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
from processor import ImageProcessor


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.processor = ImageProcessor()

        self.title("Projekt - Detekcja i Interpretacja Obiektów")
        self.geometry("1100x850")  # Zwiększyłem nieco wysokość na nowe przyciski

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- PANEL BOCZNY ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="OBRÓBKA OBRAZU",
                                       font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.btn_load = ctk.CTkButton(self.sidebar_frame, text="Wczytaj Obraz", command=self.open_file)
        self.btn_load.grid(row=1, column=0, padx=20, pady=10)

        self.btn_save = ctk.CTkButton(self.sidebar_frame, text="Zapisz Obraz", command=self.save_file, fg_color="green")
        self.btn_save.grid(row=2, column=0, padx=20, pady=10)

        # Krawędzie
        self.edge_label = ctk.CTkLabel(self.sidebar_frame, text="Krawędzie:", font=ctk.CTkFont(weight="bold"))
        self.edge_label.grid(row=3, column=0, padx=20, pady=(15, 5))
        ctk.CTkButton(self.sidebar_frame, text="Canny", command=lambda: self.apply_op("canny")).grid(row=4, column=0,
                                                                                                     padx=20, pady=2)
        ctk.CTkButton(self.sidebar_frame, text="Sobel", command=lambda: self.apply_op("sobel")).grid(row=5, column=0,
                                                                                                     padx=20, pady=2)
        ctk.CTkButton(self.sidebar_frame, text="Laplacian", command=lambda: self.apply_op("laplace")).grid(row=6,
                                                                                                           column=0,
                                                                                                           padx=20,
                                                                                                           pady=2)

        # PROGOWANIE (Nowa sekcja)
        self.thresh_label = ctk.CTkLabel(self.sidebar_frame, text="Progowanie:", font=ctk.CTkFont(weight="bold"))
        self.thresh_label.grid(row=7, column=0, padx=20, pady=(15, 5))
        ctk.CTkButton(self.sidebar_frame, text="Binarne", command=lambda: self.apply_op("thresh_bin")).grid(row=8,
                                                                                                            column=0,
                                                                                                            padx=20,
                                                                                                            pady=2)
        ctk.CTkButton(self.sidebar_frame, text="Otsu", command=lambda: self.apply_op("thresh_otsu")).grid(row=9,
                                                                                                          column=0,
                                                                                                          padx=20,
                                                                                                          pady=2)
        ctk.CTkButton(self.sidebar_frame, text="Adaptacyjne", command=lambda: self.apply_op("thresh_adapt")).grid(
            row=10, column=0, padx=20, pady=2)

        # SIECI NEURONOWE (Nowa sekcja)
        self.nn_label = ctk.CTkLabel(self.sidebar_frame, text="Sztuczna Inteligencja:", font=ctk.CTkFont(weight="bold"))
        self.nn_label.grid(row=11, column=0, padx=20, pady=(15, 5))

        self.btn_detect = ctk.CTkButton(self.sidebar_frame, text="Wykryj Koty (Sieć NN)",
                                        command=self.run_nn_detection, fg_color="#B22222", hover_color="#8B0000")
        self.btn_detect.grid(row=12, column=0, padx=20, pady=5)

        # Zmień index wierszy (row) dla starych przycisków, żeby się nie nałożyły!
        self.btn_undo = ctk.CTkButton(self.sidebar_frame, text="Cofnij (Undo)", command=self.undo, fg_color="#A0522D")
        self.btn_undo.grid(row=13, column=0, padx=20, pady=(30, 5))

        self.btn_reset = ctk.CTkButton(self.sidebar_frame, text="Resetuj wszystko", command=self.reset_image,
                                       fg_color="gray")
        self.btn_reset.grid(row=14, column=0, padx=20, pady=5)
        # --- PRZETWARZANIE WSADOWE (Nowa sekcja) ---
        self.batch_label = ctk.CTkLabel(self.sidebar_frame, text="Operacje na folderze:",
                                        font=ctk.CTkFont(weight="bold"))
        self.batch_label.grid(row=15, column=0, padx=20, pady=(15, 5))

        self.btn_batch = ctk.CTkButton(self.sidebar_frame, text="Przetwórz Folder", command=self.open_batch_window,
                                       fg_color="#483D8B")
        self.btn_batch.grid(row=16, column=0, padx=20, pady=5)

        self.btn_reset.grid(row=17, column=0, padx=20, pady=5)

        # --- PANEL GŁÓWNY ---
        self.image_label = ctk.CTkLabel(self, text="Nie wczytano obrazu")
        self.image_label.grid(row=0, column=1, padx=20, pady=20)

    def open_file(self):
        file_path = filedialog.askopenfilename()
        if file_path and self.processor.load_image(file_path):
            self.update_display()

    # ==========================================
    # LOGIKA PRZETWARZANIA WSADOWEGO
    # ==========================================
    def open_batch_window(self):
        """Otwiera nowe, małe okienko do ustawień operacji na folderach."""
        batch_win = ctk.CTkToplevel(self)
        batch_win.title("Przetwarzanie wsadowe")
        batch_win.geometry("400x450")
        batch_win.attributes('-topmost', True)  # Zawsze na wierzchu

        self.batch_input = ""
        self.batch_watermark = ""

        # Wybór folderu
        ctk.CTkLabel(batch_win, text="1. Folder źródłowy:").pack(pady=(20, 5))
        lbl_in = ctk.CTkLabel(batch_win, text="Brak", text_color="gray")
        lbl_in.pack()

        def set_in():
            self.batch_input = filedialog.askdirectory()
            lbl_in.configure(text=self.batch_input.split('/')[-1] if self.batch_input else "Brak")

        ctk.CTkButton(batch_win, text="Wybierz folder", command=set_in).pack(pady=5)

        # Wybór znaku wodnego
        ctk.CTkLabel(batch_win, text="2. Znak wodny (opcjonalnie):").pack(pady=(15, 5))
        lbl_wm = ctk.CTkLabel(batch_win, text="Brak", text_color="gray")
        lbl_wm.pack()

        def set_wm():
            self.batch_watermark = filedialog.askopenfilename()
            lbl_wm.configure(text=self.batch_watermark.split('/')[-1] if self.batch_watermark else "Brak")

        ctk.CTkButton(batch_win, text="Wybierz plik", command=set_wm).pack(pady=5)

        # Ustawienia rozmiaru
        ctk.CTkLabel(batch_win, text="3. Nowa rozdzielczość (opcjonalnie):").pack(pady=(15, 5))
        frame_res = ctk.CTkFrame(batch_win, fg_color="transparent")
        frame_res.pack()
        entry_w = ctk.CTkEntry(frame_res, placeholder_text="Szerokość", width=100)
        entry_w.pack(side="left", padx=5)
        entry_h = ctk.CTkEntry(frame_res, placeholder_text="Wysokość", width=100)
        entry_h.pack(side="left", padx=5)

        # Przycisk startu
        def start_batch():
            if not self.batch_input:
                return

            out_dir = filedialog.askdirectory(title="Wybierz folder docelowy")
            if out_dir:
                try:
                    w = int(entry_w.get()) if entry_w.get() else None
                    h = int(entry_h.get()) if entry_h.get() else None
                    size = (w, h) if w and h else None
                except ValueError:
                    size = None  # Jeśli wpisano bzdury, nie zmieniaj rozmiaru

                wm = self.batch_watermark if self.batch_watermark else None

                count = self.processor.process_batch(self.batch_input, out_dir, size, wm)
                lbl_in.configure(text=f"Gotowe! Zapisano: {count} plików", text_color="green")

        ctk.CTkButton(batch_win, text="START", command=start_batch, fg_color="green").pack(pady=30)

    def save_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png")
        if file_path:
            self.processor.save_image(file_path)

    def undo(self):
        if self.processor.undo():
            self.update_display()

    def reset_image(self):
        self.processor.reset_image()
        self.update_display()

    def apply_op(self, op_type):
        if self.processor.original_image is None: return

        if op_type == "canny":
            self.processor.detect_edges_canny()
        elif op_type == "sobel":
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

    def update_display(self):
        pil_img = self.processor.get_image_for_gui()
        if pil_img:
            max_size = (800, 600)
            pil_img.thumbnail(max_size, Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            self.image_label.configure(image=ctk_img, text="")

    def run_nn_detection(self):
        """Uruchamia detekcję kotów z małym opóźnieniem, żeby UI nie 'zamarzło' na sekundę."""
        if self.processor.original_image is not None:
            self.processor.detect_cats_with_nn(threshold=0.6)  # Możesz zmienić czułość (0.0 do 1.0)
            self.update_display()

if __name__ == "__main__":
    app = App()
    app.mainloop()