# -*- coding: utf-8 -*-
"""
Запись первых 5 секунд и распознавание (Vosk).
Добавлено ПОУРОВНЕВОЕ ПРЕДУСИЛЕНИЕ: переменная PREAMP_GAIN.
"""

import os
import sys
import json
import time

import numpy as np
import sounddevice as sd

# ===== Настройки =====
RATE = 16000
CHANNELS = 1
DURATION_SEC = 5.0

# Путь к модели Vosk (папка с model.conf внутри)
MODEL_PATH = "../models/vosk-model-small-ru-0.22"  # как вы просили

# Коэффициент предусиления (линейный). 1.0 = без изменений, 2.0 ≈ +6 dB
PREAMP_GAIN = float(os.getenv("PREAMP_GAIN", "1.0"))


def list_devices():
    print("Доступные устройства для записи:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            print(f"{i}: {dev['name']}")


def choose_device():
    list_devices()
    try:
        device = int(input("Введите номер микрофона (или Enter для умолчания): ") or "-1")
        if device < 0:
            return None
        return device
    except Exception:
        print("Некорректный ввод, использую устройство по умолчанию.")
        return None


def apply_gain_int16(mono_int16: np.ndarray, gain: float) -> np.ndarray:
    """
    Сатурирующее (с насыщением) предусиление для int16.
    Используем int32 как промежуточный формат, потом клип до диапазона int16.
    """
    if gain == 1.0:
        return mono_int16  # быстрый путь
    # переводим в int32, умножаем, клипуем, возвращаем в int16
    x = mono_int16.astype(np.int32)
    # округление к нулю достаточно; если хочешь — можно добавить np.rint
    x = (x * gain).astype(np.int32)
    np.clip(x, -32768, 32767, out=x)
    return x.astype(np.int16)


def record_5_seconds(device=None):
    """Блокирующая запись 5 секунд в int16 + предусиление PREAMP_GAIN."""
    print(f"\nЗапись {DURATION_SEC:.1f} сек @ {RATE} Гц, mono...")
    print(f"Предусиление: PREAMP_GAIN = {PREAMP_GAIN:g}")

    # Проверим, что устройство примет наши параметры
    try:
        sd.check_input_settings(device=device, samplerate=RATE, channels=CHANNELS, dtype="int16")
    except Exception as e:
        print(f"[audio] Недопустимые настройки для устройства: {e}", file=sys.stderr)
        return None

    # Блокирующая запись: без callback и потоков
    try:
        frames = int(RATE * DURATION_SEC)
        buf = sd.rec(frames, samplerate=RATE, channels=CHANNELS, dtype="int16", device=device)
        sd.wait()  # дождаться конца записи

        # buf.shape == (frames, CHANNELS); приведём к 1D int16
        if buf.ndim == 2 and buf.shape[1] > 1:
            # если вдруг стерео, берём среднее каналов (это не шумодав)
            mono = ((buf[:, 0].astype(np.int32) + buf[:, 1].astype(np.int32)) // 2).astype(np.int16)
        else:
            mono = buf[:, 0] if buf.ndim == 2 else buf.astype(np.int16)

        # ПРИМЕНЯЕМ ПРЕДУСИЛЕНИЕ
        mono = apply_gain_int16(mono, PREAMP_GAIN)

        print("Запись завершена.")
        return mono
    except Exception as e:
        print(f"[audio] Ошибка записи: {e}", file=sys.stderr)
        return None


def recognize_vosk(pcm16_mono: np.ndarray):
    print("Загрузка модели Vosk...")
    try:
        from vosk import Model, KaldiRecognizer, SetLogLevel
    except Exception as e:
        print("Пакет 'vosk' не найден. Установите: pip install vosk", file=sys.stderr)
        return None

    # Тише логирование
    try:
        SetLogLevel(-1)
    except Exception:
        pass

    if not os.path.isdir(MODEL_PATH):
        print(
            f"Не найдена папка модели: {MODEL_PATH}\n"
            f"Скачайте и распакуйте модель (например vosk-model-small-ru-0.22) и проверьте путь.",
            file=sys.stderr,
        )
        return None

    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, RATE)
    # try: rec.SetWords(True); except: pass  # можно включить слова

    print("Распознаю 5 секунд аудио...")
    t0 = time.perf_counter()
    raw = pcm16_mono.tobytes()
    try:
        rec.AcceptWaveform(raw)
        out = rec.FinalResult()
    except Exception:
        out = rec.Result()
    t1 = time.perf_counter()

    try:
        j = json.loads(out)
    except Exception:
        j = {"text": ""}

    text = (j.get("text") or "").strip()
    rtf = (t1 - t0) / DURATION_SEC  # real-time factor
    print(f"Готово. Время инференса: {(t1 - t0):.3f}s (RTF={rtf:.2f})\n")
    return text


def main():
    device = choose_device()
    audio = record_5_seconds(device=device)
    if audio is None:
        return

    text = recognize_vosk(audio)
    if text is None:
        return

    if text:
        print("▶ Распознано:")
        print(text)
    else:
        print("Ничего не распознано (пусто). Попробуйте говорить ближе к микрофону или громче.")


if __name__ == "__main__":
    main()
