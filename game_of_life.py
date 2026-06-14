import cv2
import numpy as np


def game_of_life_step(binary_img):
    """
    binary_img: jedno-kanałowy obraz grayscale (wartości 0 lub 255)
    """
    # Zamiana wartości 0/255 na 0 i 1 dla łatwiejszego liczenia
    cells = (binary_img > 127).astype(np.uint8)

    # Kernel 3x3 z zerem w środku - posłuży do zsumowania żywych sąsiadów wokół komórki
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0

    # Szybkie filtrowanie 2D zliczające sąsiadów
    neighbors = cv2.filter2D(cells, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    # Tworzymy puste płótno na nową generację
    next_generation = np.zeros_like(cells)

    # Zasada 1 i 2: Żywa komórka przeżywa, jeśli ma 2 lub 3 sąsiadów
    next_generation[(cells == 1) & ((neighbors == 2) | (neighbors == 3))] = 1

    # Zasada 3: Martwa komórka ożywa, jeśli ma dokładnie 3 sąsiadów
    next_generation[(cells == 0) & (neighbors == 3)] = 1

    # Konwersja z powrotem do formatu wyświetlania 0/255
    return (next_generation * 255).astype(np.uint8)