# -*- coding: utf-8 -*-
"""
Увеличение громкости WAV-файла на заданный коэффициент.
"""

import sys
import os
import numpy as np
import soundfile as sf

# ===== Настройки =====
INPUT_FILE  = "test4.wav"       # исходный файл
OUTPUT_FILE = "test_louder.wav"  # куда сохранить
GAIN = 32.0   # коэффициент (1.0 = без изменений, 2.0 ≈ +6 dB, 4.0 ≈ +12 dB)


def amplify_wav(input_path: str, output_path: str, gain: float):
    # читаем wav (soundfile вернёт float32 в [-1,1], если так попросить)
    data, sr = sf.read(input_path, dtype="float32", always_2d=False)

    # применяем коэффициент
    data = data * gain

    # клип, чтобы не было перегрузки
    np.clip(data, -1.0, 1.0, out=data)

    # сохраняем обратно в wav (float32 — безопасно)
    sf.write(output_path, data, sr, subtype="PCM_16")
    print(f"Готово: файл сохранён в {output_path} (коэффициент {gain})")


if __name__ == "__main__":
    if not os.path.isfile(INPUT_FILE):
        print(f"Файл {INPUT_FILE} не найден")
        sys.exit(1)

    amplify_wav(INPUT_FILE, OUTPUT_FILE, GAIN)
