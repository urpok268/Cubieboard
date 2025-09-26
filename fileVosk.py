# -*- coding: utf-8 -*-
"""
Распознавание из файла WAV Vosk с улучшениями качества:
- Автоприведение к 16 кГц mono
- Удаление DC-сместа
- Опциональный ВЧ-фильтр (HPF) для среза гула/поп-взрывов: переменная HPF_HZ
- Опциональный предусилитель PREAMP_GAIN (линейный коэффициент)
"""

import os
import sys
import json
import time
import math
import numpy as np
import soundfile as sf

# ----- настройки (через окружение можно переопределить) -----
MODEL_PATH     = os.getenv("VOSK_MODEL_PATH", "../models/vosk-model-small-ru-0.22")
WAV_PATH       = os.getenv("WAV_PATH", "test3n.wav")
TARGET_RATE    = 16000
PREAMP_GAIN    = float(os.getenv("PREAMP_GAIN", "1.0"))     # 1.0=без изменений; 2.0≈+6 dB
HPF_HZ         = float(os.getenv("HPF_HZ", "0"))            # 0=выкл; 80..150 Гц обычно хорошо
CHUNK_SAMPLES  = 8000                                       # ~0.5s при 16 кГц
VOSK_WORDS     = os.getenv("VOSK_WORDS", "0") == "1"        # SetWords(True)

def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    if x.shape[1] == 1:
        return x[:, 0]
    # усреднение каналов (это не шумодав)
    return np.mean(x, axis=1)

def highpass_if_needed(x: np.ndarray, sr: int, hpf_hz: float) -> np.ndarray:
    """1-го порядка Баттерворт HPF, если hpf_hz>0. Возвращает float32 [-1..1]."""
    if hpf_hz is None or hpf_hz <= 0:
        return x
    try:
        from scipy.signal import butter, lfilter
    except Exception as e:
        raise RuntimeError("Нужна scipy для HPF. Установите: pip install scipy") from e
    # Нормированная частота среза
    wc = max(1.0, min(hpf_hz, sr * 0.45)) / (sr * 0.5)
    b, a = butter(1, wc, btype='highpass')
    y = lfilter(b, a, x).astype(np.float32, copy=False)
    return y

def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_RATE:
        return x
    try:
        from scipy.signal import resample_poly
    except Exception as e:
        raise RuntimeError("Нужна scipy для ресемплинга. Установите: pip install scipy") from e
    g = math.gcd(sr, TARGET_RATE)
    up, down = TARGET_RATE // g, sr // g
    return resample_poly(x, up, down).astype(np.float32, copy=False)

def apply_gain_and_clip_to_int16(x: np.ndarray, gain: float) -> np.ndarray:
    """Применяем предусиление в float [-1..1], затем клипуем и переводим в int16."""
    if gain != 1.0:
        x = x * gain
    np.clip(x, -1.0, 1.0, out=x)  # защита от перегруза
    return (x * 32767.0).astype(np.int16, copy=False)

def main():
    # 1) Загрузка WAV (float32 в диапазоне [-1,1] от soundfile)
    try:
        audio, sr = sf.read(WAV_PATH, dtype="float32", always_2d=False)
    except Exception as e:
        print(f"[io] Не удалось прочитать {WAV_PATH}: {e}", file=sys.stderr)
        sys.exit(1)

    audio = to_mono(audio).astype(np.float32, copy=False)
    if audio.size == 0:
        print("[io] Пустой аудиофайл.", file=sys.stderr)
        sys.exit(1)

    # 1.1) Удаление DC-сместа (часто спасает «глухоту» и накачку НЧ)
    audio -= float(np.mean(audio, dtype=np.float64))

    # 1.2) Опциональный ВЧ-фильтр (например, HPF_HZ=120 хорошо режет гул/поп)
    if HPF_HZ > 0:
        audio = highpass_if_needed(audio, sr, HPF_HZ)

    # 2) Ресемплинг до 16 кГц при необходимости
    if sr != TARGET_RATE:
        audio = resample_to_16k(audio, sr)
        sr = TARGET_RATE

    # 3) Предусиление + преобразование к int16
    pcm16 = apply_gain_and_clip_to_int16(audio, PREAMP_GAIN)

    # 4) Vosk
    try:
        from vosk import Model, KaldiRecognizer, SetLogLevel
    except Exception as e:
        print("Пакет 'vosk' не найден. Установите: pip install vosk", file=sys.stderr)
        sys.exit(1)

    try:
        SetLogLevel(-1)
    except Exception:
        pass

    if not os.path.isdir(MODEL_PATH):
        print(
            f"[vosk] Не найдена папка модели: {MODEL_PATH}\n"
            f"Скачайте и распакуйте (напр. vosk-model-small-ru-0.22) и проверьте путь.",
            file=sys.stderr,
        )
        sys.exit(1)

    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, TARGET_RATE)
    if VOSK_WORDS:
        try:
            rec.SetWords(True)
        except Exception:
            pass

    # 5) Стримим в распознаватель чанками
    print(
        f"Файл: {WAV_PATH} | sr→{TARGET_RATE} Hz | длительность ~{len(pcm16)/TARGET_RATE:.2f}s | "
        f"PREAMP_GAIN={PREAMP_GAIN:g} | HPF_HZ={HPF_HZ:g}"
    )
    t0 = time.perf_counter()
    pos = 0
    n = len(pcm16)
    while pos < n:
        end = min(pos + CHUNK_SAMPLES, n)
        raw = pcm16[pos:end].tobytes()
        rec.AcceptWaveform(raw)
        pos = end

    out = rec.FinalResult()
    t1 = time.perf_counter()

    try:
        j = json.loads(out)
    except Exception:
        j = {"text": ""}

    text = (j.get("text") or "").strip()
    rtf = (t1 - t0) / max(1e-9, (len(pcm16) / TARGET_RATE))
    print(f"\nRTF={rtf:.2f} | t_proc={(t1 - t0):.3f}s")
    print("\n▶ Распознано:")
    print(text if text else "(пусто)")

if __name__ == "__main__":
    main()
