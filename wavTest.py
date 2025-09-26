# -*- coding: utf-8 -*-
"""
Запись в WAV с выбором устройства (Cubieboard2-friendly).
- 16 kHz, mono, int16
- сначала запись в память (sd.rec + sd.wait), затем сохранение .wav
"""

import os
import sys
import time
import wave
import numpy as np
import sounddevice as sd

# ===== Параметры по умолчанию =====
RATE = 16000
CHANNELS = 1
DEFAULT_DURATION = 5.0  # сек
DEFAULT_FILENAME = "record.wav"

def list_devices():
    print("Доступные устройства для записи:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            print(f"{i}: {dev['name']}")

def choose_device():
    list_devices()
    try:
        val = input("Введите номер микрофона (или Enter для умолчания): ").strip()
        if not val:
            return None
        return int(val)
    except Exception:
        print("Некорректный ввод, использую устройство по умолчанию.")
        return None

def ask_duration_and_filename():
    try:
        d = input(f"Длительность записи, сек (по умолчанию {DEFAULT_DURATION}): ").strip()
        duration = float(d) if d else DEFAULT_DURATION
        if duration <= 0:
            duration = DEFAULT_DURATION
    except Exception:
        duration = DEFAULT_DURATION

    fn = input(f"Имя файла (по умолчанию {DEFAULT_FILENAME}): ").strip()
    filename = fn if fn else DEFAULT_FILENAME
    if not filename.lower().endswith(".wav"):
        filename += ".wav"
    return duration, filename

def save_wav_int16(filename: str, pcm: np.ndarray, rate: int, channels: int):
    """Сохранение 1D/2D int16 массива в WAV (без внешних библиотек)."""
    if pcm.ndim == 1 and channels == 1:
        data_bytes = pcm.tobytes()
    elif pcm.ndim == 2:
        # (frames, channels)
        data_bytes = pcm.astype(np.int16, copy=False).tobytes()
    else:
        # приведём аккуратно к нужной форме
        pcm = np.asarray(pcm, dtype=np.int16).reshape(-1, channels)
        data_bytes = pcm.tobytes()

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # int16 = 2 байта
        wf.setframerate(rate)
        wf.writeframes(data_bytes)

def main():
    # Чуть «подушим» требования к реальному времени: больше буферы, высокая задержка
    try:
        os.nice(5)
    except Exception:
        pass
    sd.default.blocksize = int(RATE * 0.20)  # ~200 мс
    sd.default.latency = ("high", "high")

    device = choose_device()
    duration, filename = ask_duration_and_filename()

    # Проверяем, что устройство примет наши параметры
    try:
        sd.check_input_settings(device=device, samplerate=RATE, channels=CHANNELS, dtype="int16")
    except Exception as e:
        print(f"[audio] Недопустимые настройки: {e}", file=sys.stderr)
        return

    frames = int(RATE * duration)
    print(f"\nЗапись {duration:.1f} сек @ {RATE} Гц, mono → {filename}")
    try:
        # sd.rec создаёт поток, пишет в массив; sd.wait блокирует до завершения записи
        buf = sd.rec(frames, samplerate=RATE, channels=CHANNELS, dtype="int16", device=device)
        sd.wait()
    except Exception as e:
        print(f"[audio] Ошибка записи: {e}", file=sys.stderr)
        return

    # buf.shape = (frames, 1) -> сделаем 1D для mono
    if buf.ndim == 2 and buf.shape[1] == 1:
        pcm = buf[:, 0]
    else:
        pcm = buf

    try:
        save_wav_int16(filename, pcm, RATE, CHANNELS)
    except Exception as e:
        print(f"[io] Не удалось сохранить WAV: {e}", file=sys.stderr)
        return

    size_sec = len(pcm) / RATE
    print(f"Готово. Записано {size_sec:.2f} сек в '{filename}'.")

if __name__ == "__main__":
    main()
