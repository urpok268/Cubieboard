# -*- coding: utf-8 -*-
"""
Live ASR (Vosk) — Cubieboard2-optimized:
- int16 весь путь (меньше CPU)
- deque-буфер (без постоянных np.concatenate)
- редкие принты в callback
- увеличенные буферы для избежания overflow
"""

import os
os.environ["TQDM_DISABLE"] = "1"

import sys
import time
import json
import queue
import threading
from typing import Optional, List
from collections import deque

import numpy as np
import sounddevice as sd

# ===================== ПАРАМЕТРЫ =====================
RATE = 16000
CHANNELS = 1

CHUNK_DURATION = 3.0     # строго 3 секунды
OVERLAP        = 0.5     # небольшое перекрытие

# Cubieboard2: дайте системе простор (120–160 мс). Начните с 0.12s:
BLOCKSIZE = int(RATE * 0.12)   # 120 мс

# Энерго-VAD
VAD_ENERGY_THRESHOLD = 0.015   # подстрой в шуме 0.007–0.015
VAD_MIN_SPEECH_RATIO = 0.22

# Очередь/вывод
PROC_QUEUE_MAX = 2
PRINT_COOLDOWN = 0.6

# Путь к модели Vosk (папка с model.conf внутри)
MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "../models/vosk-model-small-ru-0.22")

# троттлинг печати из callback
_mic_cnt = 0

# ===================== УТИЛИТЫ =====================
def energy_vad_mask_int16(x: np.ndarray, thr: float):
    # масштабируем порог под int16
    thr_i16 = int(32767 * thr)
    return np.abs(x.astype(np.int32)) > thr_i16

def is_speech_int16(x: np.ndarray) -> bool:
    if x.size == 0:
        return False
    return np.mean(energy_vad_mask_int16(x, VAD_ENERGY_THRESHOLD)) >= VAD_MIN_SPEECH_RATIO

def similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    a1, b1 = a.strip().lower(), b.strip().lower()
    if a1 == b1:
        return True
    if a1 in b1 or b1 in a1:
        la, lb = len(a1), len(b1)
        return abs(la - lb) / max(1, max(la, lb)) < 0.25
    return False

def squash_repeats(text: str, max_word_repeat: int = 3, max_phrase_repeat: int = 2) -> str:
    if not text:
        return text
    words = text.split()
    if not words:
        return text
    comp: List[str] = []
    rep = 0
    last = None
    for w in words:
        wl = w.lower()
        if last is not None and wl == last:
            rep += 1
            if rep < max_word_repeat:
                comp.append(w)
        else:
            rep = 0
            comp.append(w)
            last = wl
    text = " ".join(comp)

    def collapse_phrase(seq: List[str], n: int, max_rep: int) -> List[str]:
        if len(seq) < n:
            return seq
        out = []
        i = 0
        while i < len(seq):
            out.extend(seq[i:i+n])
            reps = 0
            while i + (reps+1)*n < len(seq) and seq[i:i+n] == seq[i+(reps+1)*n:i+(reps+2)*n]:
                reps += 1
            if reps > 0:
                reps_keep = min(reps, max_rep-1)
                for _ in range(reps_keep):
                    out.extend(seq[i:i+n])
            i += (reps + 1) * n
        return out

    words = text.split()
    words = collapse_phrase(words, 2, max_phrase_repeat)
    words = collapse_phrase(words, 3, max_phrase_repeat)
    return " ".join(words)

# ===================== АУДИО БУФЕР (deque, int16) =====================
class Int16Ring:
    """
    Буфер блоков (int16) на deque.
    - add(): добавляет блоки без копий
    - peek_chunk(): собирает первые N с минимальным количеством копий
    - consume(n): съедает n семплов слева
    """
    def __init__(self, rate: int, max_seconds: int = 10):
        self.rate = rate
        self.blocks = deque()          # deque[np.ndarray[int16]]
        self.total = 0                 # всего семплов в очереди
        self.lock = threading.Lock()
        self.data_event = threading.Event()
        self.max_len = rate * max_seconds

    def add(self, mono_int16: np.ndarray):
        with self.lock:
            self.blocks.append(mono_int16)
            self.total += mono_int16.size
            # удерживаем не более max_len
            while self.total > self.max_len and self.blocks:
                drop = self.blocks.popleft()
                self.total -= drop.size
            self.data_event.set()

    def wait_for(self, required_samples: int):
        while True:
            with self.lock:
                if self.total >= required_samples:
                    return
            self.data_event.wait(timeout=0.1)
            self.data_event.clear()

    def _gather(self, n: int) -> np.ndarray:
        """Собирает первые n семплов (без изменения очереди)."""
        out = np.empty(n, dtype=np.int16)
        copied = 0
        for blk in self.blocks:
            need = n - copied
            if need <= 0:
                break
            take = min(blk.size, need)
            out[copied:copied+take] = blk[:take]
            copied += take
        return out

    def consume(self, n: int):
        """Съедает n семплов слева."""
        with self.lock:
            remain = n
            while remain > 0 and self.blocks:
                blk = self.blocks[0]
                if blk.size <= remain:
                    self.total -= blk.size
                    remain -= blk.size
                    self.blocks.popleft()
                else:
                    # отрезаем слева
                    self.blocks[0] = blk[remain:].copy()
                    self.total -= remain
                    remain = 0

    def pop_chunk(self, duration_s: float, overlap_s: float) -> Optional[np.ndarray]:
        chunk_size = int(duration_s * self.rate)
        overlap_size = int(overlap_s * self.rate)
        with self.lock:
            if self.total < chunk_size:
                return None
            chunk = self._gather(chunk_size)
            # потребляем только (chunk_size - overlap), чтобы оставить перекрытие
            consume_n = max(0, chunk_size - overlap_size)
            # выходим из локов перед копиями
        if consume_n:
            self.consume(consume_n)
        return chunk

# ===================== ПОТОКИ =====================
def audio_callback(indata, frames, t, status):
    """Пишем крайне умеренно, чтобы не тормозить ARM-CPU."""
    if status:
        print(f"[audio] {status}", file=sys.stderr)

    # indata: int16, shape (frames, CHANNELS)
    if indata.ndim == 2 and indata.shape[1] > 1:
        # downmix в моно
        mono = indata.mean(axis=1).astype(np.int16, copy=False)
    else:
        mono = indata[:, 0] if indata.ndim == 2 else indata

    global audio_buffer, _mic_cnt
    audio_buffer.add(mono)

    _mic_cnt = (_mic_cnt + 1) % 30   # печатаем редко
    if _mic_cnt == 0:
        print(f"[mic] ok ({frames} frames)")

class LatestQueue:
    """Очередь, хранящая только последние элементы (без накопления задержки)."""
    def __init__(self, maxsize=1):
        self.q = queue.Queue(maxsize=maxsize)
        self.lock = threading.Lock()

    def put_latest(self, item):
        with self.lock:
            while True:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            self.q.put_nowait(item)

    def get(self, timeout=None):
        return self.q.get(timeout=timeout)

def monitor_buffer(audio_buffer: Int16Ring, processing_queue: "LatestQueue"):
    chunk_samples = int(CHUNK_DURATION * RATE)
    while True:
        audio_buffer.wait_for(chunk_samples)
        chunk = audio_buffer.pop_chunk(CHUNK_DURATION, OVERLAP)
        if chunk is not None and is_speech_int16(chunk):
            print(f"[proc] чанк ~{len(chunk)/RATE:.2f}s -> обработка")
            processing_queue.put_latest((chunk, time.perf_counter()))

# Глобальные метрики для показа среднего времени
EMA_ALPHA = 0.3
ema_proc_time = None  # float (сек)

# ===================== БЭКЕНД ASR: Vosk =====================
class VoskWrapper:
    """ Обёртка над vosk.KaldiRecognizer для одного 3-сек чанка (int16 → bytes). """
    def __init__(self, model_path: str, rate: int):
        try:
            from vosk import Model, SetLogLevel
        except Exception as e:
            raise RuntimeError("Не установлен пакет 'vosk'. Установите: pip install vosk") from e

        try:
            SetLogLevel(-1)
        except Exception:
            pass

        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Не найдена папка модели: {model_path}\n"
                f"Скачайте и распакуйте модель (например vosk-model-small-ru-0.22) и обновите MODEL_PATH."
            )

        self.rate = rate
        self._model = Model(model_path)

    def __call__(self, inputs):
        from vosk import KaldiRecognizer

        pcm16 = inputs["array"]  # np.int16 (моно)
        if pcm16.dtype != np.int16:
            pcm16 = np.clip(pcm16, -32768, 32767).astype(np.int16, copy=False)
        raw = pcm16.tobytes()

        rec = KaldiRecognizer(self._model, self.rate)
        try:
            rec.SetWords(True)
        except Exception:
            pass

        try:
            rec.AcceptWaveform(raw)
            out = rec.FinalResult()
        except Exception:
            out = rec.Result()

        try:
            j = json.loads(out)
        except Exception:
            j = {"text": ""}

        text = (j.get("text") or "").strip()
        return {"text": text}

def load_asr():
    print("ASR backend: Vosk")
    return VoskWrapper(MODEL_PATH, RATE)

# ===================== РАБОЧИЙ ПОТОК ASR =====================
def asr_worker(proc_q: LatestQueue, asr):
    global ema_proc_time
    last_text = ""
    last_print_ts = 0.0
    seq = 0

    while True:
        try:
            audio_item = proc_q.get(timeout=1.0)
        except queue.Empty:
            continue

        if isinstance(audio_item, tuple):
            audio_chunk, t_ready = audio_item
        else:
            audio_chunk, t_ready = audio_item, time.perf_counter()

        try:
            t0 = time.perf_counter()
            result = asr({"array": audio_chunk, "sampling_rate": RATE})
            t1 = time.perf_counter()

            text = (result.get("text") or "").strip()

            now = time.perf_counter()
            proc_time = t1 - t0
            e2e_time = now - t_ready
            rtf = proc_time / CHUNK_DURATION

            if not text:
                print(f"[done] пусто | t_proc={proc_time:.3f}s | RTF={rtf:.2f} | t_e2e={e2e_time:.3f}s")
                continue

            clean = squash_repeats(text)
            if similar(clean, last_text) and (now - last_print_ts) < PRINT_COOLDOWN:
                print(f"[done] пропуск (повтор) | t_proc={proc_time:.3f}s | RTF={rtf:.2f} | t_e2e={e2e_time:.3f}s")
                continue

            ema_proc_time = proc_time if ema_proc_time is None else (
                EMA_ALPHA * proc_time + (1 - EMA_ALPHA) * ema_proc_time
            )
            seq += 1
            last_text = clean
            last_print_ts = now

            print(
                f"\n#{seq} ▶ Распознано (3с): {clean}\n"
                f"   t_proc={proc_time:.3f}s | RTF={rtf:.2f} | t_e2e={e2e_time:.3f}s | avg={ema_proc_time:.3f}s\n"
            )

        except Exception as e:
            print(f"[asr] Ошибка обработки: {e}", file=sys.stderr)

# ===================== УСТРОЙСТВА =====================
def list_devices():
    print("Доступные устройства для записи:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            print(f"{i}: {dev['name']}")

def choose_device():
    list_devices()
    try:
        device = int(input("Введите номер микрофона: "))
        return device
    except Exception:
        print("Некорректный ввод, использую устройство по умолчанию.")
        return None

# ===================== MAIN =====================
def main():
    # на ARM иногда полезно слегка «понизить» приоритет печати
    try:
        os.nice(2)
    except Exception:
        pass

    device = choose_device()

    # Проверяем, что ALSA/устройство примут наши параметры
    try:
        sd.check_input_settings(device=device, samplerate=RATE,
                                channels=CHANNELS, dtype="int16")
    except Exception as e:
        print(f"[audio] Настройки недопустимы: {e}", file=sys.stderr)
        return

    # Можно принудительно выбрать ALSA как хост-API, если мешает PulseAudio:
    # try:
    #     sd.default.hostapi = [i for i,a in enumerate(sd.query_hostapis()) if a['name'].lower().startswith('alsa')][0]
    # except Exception:
    #     pass

    global audio_buffer
    audio_buffer = Int16Ring(RATE)

    proc_q = LatestQueue(maxsize=PROC_QUEUE_MAX)
    asr = load_asr()

    threading.Thread(target=asr_worker, args=(proc_q, asr), daemon=True).start()
    threading.Thread(target=monitor_buffer, args=(audio_buffer, proc_q), daemon=True).start()

    print("\nНачало записи... Нажмите Ctrl+C для остановки")
    try:
        with sd.InputStream(
            samplerate=RATE,
            channels=CHANNELS,
            dtype="int16",          # сразу int16
            blocksize=BLOCKSIZE,
            latency="high",         # больше аппаратных буферов — меньше overflow
            callback=audio_callback,
            device=device
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nЗавершение записи...")
    except Exception as e:
        print(f"\nАудио-ошибка: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
