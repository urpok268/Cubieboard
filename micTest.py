# -*- coding: utf-8 -*-
"""
Тестер микрофона: выбор устройства + VU-метр (RMS/Peak) в консоли.
Оптимизировано под ARM (Cubieboard2):
- RawInputStream (байты int16)
- редкая работа в callback (только расчёт и отправка в очередь)
- печать в главном цикле ~10 Гц
"""

import sys
import time
import math
import queue
import threading

import numpy as np
import sounddevice as sd

# ===== Параметры =====
RATE = 16000
CHANNELS = 1
BLOCK_SEC = 0.10                          # 100 мс блок — компромисс между латентностью и стабильностью
BLOCKSIZE = int(RATE * BLOCK_SEC)
PRINT_HZ = 10                              # как часто обновлять шкалу (≈10 Гц)
BAR_LEN = 40                               # длина шкалы
DBFS_FLOOR = -60                           # низ шкалы (дБFS)
DBFS_CEIL  = 0                             # верх шкалы (дБFS)

# очередь для передачи рассчитанных уровней из callback в главный поток
level_q: "queue.Queue[tuple]" = queue.Queue(maxsize=1)
overflow_count = 0
lock = threading.Lock()


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


def dbfs_and_peak(pcm16: np.ndarray):
    """Возвращает (rms_dbfs, peak_dbfs)."""
    if pcm16.size == 0:
        return float("-inf"), float("-inf")
    # RMS
    rms = math.sqrt(np.mean(pcm16.astype(np.float32) ** 2))
    # Нормируем к full-scale int16
    rms /= 32768.0
    # Пик
    peak = np.max(np.abs(pcm16)) / 32768.0
    # dBFS
    eps = 1e-12
    rms_db = 20.0 * math.log10(max(rms, eps))
    peak_db = 20.0 * math.log10(max(peak, eps))
    return rms_db, peak_db


def make_bar(db: float, length: int = BAR_LEN, lo: float = DBFS_FLOOR, hi: float = DBFS_CEIL):
    """Строит ASCII-шкалу для данного уровня в дБFS."""
    if db <= lo:
        fill = 0
    elif db >= hi:
        fill = length
    else:
        frac = (db - lo) / (hi - lo)
        fill = int(round(frac * length))
    return "[" + "#" * fill + "-" * (length - fill) + "]"


def audio_callback(indata_bytes, frames, t, status):
    """RawInputStream: минимальная работа в callback — расчёт уровней и отправка в очередь."""
    global overflow_count
    if status:
        # засечём переполнения
        with lock:
            overflow_count += 1

    # bytes -> int16 (копия — безопаснее, т.к. буфер живёт недолго)
    pcm = np.frombuffer(indata_bytes, dtype=np.int16).copy()

    # если вдруг пришло стерео, схлопнем в моно
    if CHANNELS == 1 and pcm.size == frames * 2:
        pcm = ((pcm[0::2].astype(np.int32) + pcm[1::2].astype(np.int32)) // 2).astype(np.int16)

    rms_db, peak_db = dbfs_and_peak(pcm)

    # положим в очередь последнюю метрику, не блокируясь
    try:
        while True:
            # держим только последнее значение
            level_q.get_nowait()
    except queue.Empty:
        pass
    try:
        level_q.put_nowait((rms_db, peak_db))
    except queue.Full:
        pass


def main():
    # (опционально) снизим приоритет вывода
    try:
        import os
        os.nice(5)
    except Exception:
        pass

    device = choose_device()

    # Проверка входных настроек
    try:
        sd.check_input_settings(device=device, samplerate=RATE, channels=CHANNELS, dtype="int16")
    except Exception as e:
        print(f"[audio] Недопустимые настройки: {e}", file=sys.stderr)
        return

    print("\nСтарт VU-метра. Нажмите Ctrl+C для выхода.")
    last_print = 0.0

    try:
        with sd.RawInputStream(
            samplerate=RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=BLOCKSIZE,
            latency="high",              # дайте бóльше буферов ALSA/PortAudio
            callback=audio_callback,
            device=device,
        ):
            while True:
                now = time.time()
                # обновляем не чаще PRINT_HZ
                timeout = max(0.0, (1.0 / PRINT_HZ) - (now - last_print))
                try:
                    rms_db, peak_db = level_q.get(timeout=timeout)
                except queue.Empty:
                    continue

                bar = make_bar(rms_db)
                with lock:
                    ovf = overflow_count
                    overflow_count = 0  # обнулим счётчик на период

                # печать одной строкой с возвратом каретки
                sys.stdout.write(
                    f"\r{bar}  RMS: {rms_db:6.1f} dBFS  Peak: {peak_db:6.1f} dBFS"
                    f"   overflows:{ovf:3d}"
                )
                sys.stdout.flush()
                last_print = time.time()
    except KeyboardInterrupt:
        print("\nЗавершение.")
    except Exception as e:
        print(f"\n[audio] Ошибка: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
