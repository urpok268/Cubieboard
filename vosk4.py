# -*- coding: utf-8 -*-
"""
Live ASR (Vosk) — Cubieboard2 raw-optimized + авто частота + управляемая очередь:
- Автоопределение поддерживаемой частоты устройства (исправляет paerror -9997)
- Запись в int16 при частоте устройства (48k/44.1k/...)
- Онлайн-ресемплинг в 16 кГц для Vosk (линейная интерполяция)
- RawInputStream (байты int16), крупные буферы, минимум работ в callback
- VAD отключен
- USE_QUEUE_LIMIT: управляет глубиной очереди чанков (реалтайм vs без потерь)
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

# ===================== ПАРАМЕТРЫ (логические) =====================
VOSK_RATE = 16000          # Vosk ждёт 16 кГц
CHANNELS  = 1
CHUNK_DURATION = 3.0       # 3 секунды окна
OVERLAP        = 0.5

# Очередь/вывод
USE_QUEUE_LIMIT = False     # True = учитывать PROC_QUEUE_MAX (хранить только последний/последние);
                           # False = без ограничений (ничего не выбрасываем)
PROC_QUEUE_MAX  = 2
PRINT_COOLDOWN  = 0.6

# Путь к модели Vosk (папка с model.conf внутри)
MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "../models/vosk-model-small-ru-0.22")

# троттлинг печати из callback
_mic_cnt = 0

# ===================== УТИЛИТЫ =====================
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

# ----------- простой ресемплер (линейная интерполяция) -----------
def resample_int16_linear(x_i16: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Линейный ресемплер int16 -> int16 без внешних зависимостей.
       Для речи работает нормально. Если src_rate == dst_rate — вернёт исходный массив."""
    if src_rate == dst_rate or x_i16.size == 0:
        return x_i16
    x = x_i16.astype(np.float32) / 32767.0
    src_len = x.shape[0]
    dst_len = int(round(src_len * (dst_rate / float(src_rate))))
    if dst_len <= 0:
        return np.zeros(0, dtype=np.int16)
    # координаты исходной и целевой осей времени
    src_idx = np.linspace(0.0, src_len - 1.0, num=src_len, dtype=np.float32)
    dst_idx = np.linspace(0.0, src_len - 1.0, num=dst_len, dtype=np.float32)
    y = np.interp(dst_idx, src_idx, x).astype(np.float32)
    # обратно в int16 с клипом
    np.clip(y, -1.0, 1.0, out=y)
    return (y * 32767.0).astype(np.int16)

# ===================== АУДИО БУФЕР (deque, int16) =====================
class Int16Ring:
    def __init__(self, rate: int, max_seconds: int = 10):
        self.rate = rate
        self.blocks = deque()          # deque[np.ndarray[int16]]
        self.total = 0
        self.lock = threading.Lock()
        self.data_event = threading.Event()
        self.max_len = rate * max_seconds

    def add(self, mono_int16: np.ndarray):
        with self.lock:
            self.blocks.append(mono_int16)
            self.total += mono_int16.size
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
        with self.lock:
            remain = n
            while remain > 0 and self.blocks:
                blk = self.blocks[0]
                if blk.size <= remain:
                    self.total -= blk.size
                    remain -= blk.size
                    self.blocks.popleft()
                else:
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
        consume_n = max(0, chunk_size - overlap_size)
        if consume_n:
            self.consume(consume_n)
        return chunk

# ===================== ПОТОКИ =====================
def audio_callback(indata_bytes, frames, t, status):
    """RawInputStream: indata — bytes. Делаем минимум работы и печати."""
    #if status:
        #print(f"[audio] {status}", file=sys.stderr)

    mono = np.frombuffer(indata_bytes, dtype=np.int16).copy()

    # Если устройство внезапно стерео — схлопнем в моно без лишних копий
    if mono.size == frames * 2:
        mono = ((mono[0::2].astype(np.int32) + mono[1::2].astype(np.int32)) // 2).astype(np.int16)

    global audio_buffer, _mic_cnt
    audio_buffer.add(mono)

    _mic_cnt = (_mic_cnt + 1) % 50
    #if _mic_cnt == 0:
        #print(f"[mic] ok ({frames} frames)")

class LatestQueue:
    """
    Двухрежимная очередь:
    - use_limit=True  : хранит только последний элемент (или последние с maxsize) — минимальная задержка.
    - use_limit=False : обычная неограниченная FIFO-очередь — ничего не выбрасывается.
    """
    def __init__(self, maxsize=1, use_limit=True):
        if use_limit:
            # ограниченная очередь: будем вручную держать только последний элемент
            self.q = queue.Queue(maxsize=maxsize)
            self.use_limit = True
        else:
            # без ограничений: штатная бесконечная очередь
            self.q = queue.Queue(maxsize=0)
            self.use_limit = False
        self.lock = threading.Lock()

    def put_latest(self, item):
        if self.use_limit:
            # старое поведение: хранить только последний (вытеснение старых)
            with self.lock:
                while True:
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        break
                # используем put_nowait — у нас очередь пустая после очистки
                self.q.put_nowait(item)
        else:
            # без ограничений: обычная FIFO
            self.q.put(item)

    def get(self, timeout=None):
        return self.q.get(timeout=timeout)

def monitor_buffer(audio_buffer: Int16Ring, processing_queue: "LatestQueue",
                   capture_rate: int):
    """Достаём чанки в частоте устройства и при необходимости ресемплим их в 16 кГц."""
    chunk_samples = int(CHUNK_DURATION * capture_rate)
    while True:
        audio_buffer.wait_for(chunk_samples)
        chunk = audio_buffer.pop_chunk(CHUNK_DURATION, OVERLAP)  # int16 @ capture_rate
        if chunk is None:
            continue
        if capture_rate != VOSK_RATE:
            chunk = resample_int16_linear(chunk, capture_rate, VOSK_RATE)  # int16 @ 16k
        processing_queue.put_latest((chunk, time.perf_counter()))

# ===================== БЭКЕНД ASR: Vosk =====================
EMA_ALPHA = 0.3
ema_proc_time = None  # float (сек)

class VoskWrapper:
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
        pcm16 = inputs["array"]  # np.int16 (моно @ 16кГц)
        if pcm16.dtype != np.int16:
            pcm16 = np.clip(pcm16, -32768, 32767).astype(np.int16, copy=False)
        raw = pcm16.tobytes()

        rec = KaldiRecognizer(self._model, self.rate)
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
    print("ASR backend: Vosk (16 kHz)")
    return VoskWrapper(MODEL_PATH, VOSK_RATE)

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
            audio_chunk, t_ready = audio_item  # int16 @ 16k
        else:
            audio_chunk, t_ready = audio_item, time.perf_counter()

        try:
            t0 = time.perf_counter()
            result = asr({"array": audio_chunk, "sampling_rate": VOSK_RATE})
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
            dsr = dev.get("default_samplerate", None)
            dsr_str = f" @{int(dsr)}Hz" if dsr else ""
            print(f"{i}: {dev['name']}{dsr_str}")

def choose_device():
    list_devices()
    try:
        device = int(input("Введите номер микрофона: "))
        return device
    except Exception:
        print("Некорректный ввод, использую устройство по умолчанию.")
        return None

def pick_working_samplerate(device) -> int:
    """Подбираем частоту, которую устройство точно примет, чтобы избежать -9997."""
    # 1) пробуем default_samplerate устройства
    try:
        info = sd.query_devices(device)
        dsr = int(info.get("default_samplerate", 0)) or None
    except Exception:
        dsr = None

    def _ok(rate):
        try:
            sd.check_input_settings(device=device, samplerate=rate, channels=CHANNELS, dtype="int16")
            return True
        except Exception:
            return False

    if dsr and _ok(dsr):
        return dsr

    # 2) перебор популярных частот
    candidates = [16000, 48000, 44100, 32000, 22050, 8000]
    for r in candidates:
        if _ok(r):
            return r

    # 3) последний шанс — спросим у PortAudio глобальный default
    try:
        r = int(sd.query_devices(None, kind='input')["default_samplerate"])
        if _ok(r):
            return r
    except Exception:
        pass

    # 4) fallback
    return 16000

# ===================== MAIN =====================
def main():
    # ARM: понижаем приоритет печати (необязательно)
    try:
        os.nice(10)
    except Exception:
        pass

    # Пробуем выбрать ALSA hostapi
    try:
        alsa_idx = next(i for i, a in enumerate(sd.query_hostapis()) if 'alsa' in a['name'].lower())
        sd.default.hostapi = alsa_idx
    except Exception:
        pass

    device = choose_device()

    # Подбираем рабочую частоту устройства
    CAPTURE_RATE = pick_working_samplerate(device)
    print(f"[audio] Используемая частота устройства: {CAPTURE_RATE} Гц (Vosk всегда 16000 Гц)")

    # Проверка входных настроек с найденной частотой
    try:
        sd.check_input_settings(device=device, samplerate=CAPTURE_RATE, channels=CHANNELS, dtype="int16")
    except Exception as e:
        print(f"[audio] Настройки недопустимы даже после подбора частоты: {e}", file=sys.stderr)
        return

    # BLOCKSIZE ~0.20s на частоте устройства
    BLOCKSIZE = int(CAPTURE_RATE * 0.20)

    global audio_buffer
    audio_buffer = Int16Ring(CAPTURE_RATE)

    # Инициализация очереди с учётом флага
    proc_q = LatestQueue(maxsize=PROC_QUEUE_MAX, use_limit=USE_QUEUE_LIMIT)
    asr = load_asr()

    threading.Thread(target=asr_worker, args=(proc_q, asr), daemon=True).start()
    threading.Thread(target=monitor_buffer, args=(audio_buffer, proc_q, CAPTURE_RATE), daemon=True).start()

    print("\nНачало записи... Нажмите Ctrl+C для остановки")
    try:
        with sd.RawInputStream(
            samplerate=CAPTURE_RATE,   # <— частота устройства (чтобы не было -9997)
            channels=CHANNELS,
            dtype="int16",
            blocksize=BLOCKSIZE,
            latency=None,
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
