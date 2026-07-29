#!/usr/bin/env python3
"""Validate a network mic stream (e.g. the IP Webcam app on a phone).

    python scripts/mic_probe.py http://172.20.10.5:8080/audio.opus [seconds] [--save out.wav]

Opens the stream with PyAV, captures a few seconds, and reports whether the Pi
can reach it and whether there's actual signal (RMS/peak). Optionally saves a
WAV you can play back on the BT speaker to hear it:

    pw-play out.wav

Use this to prove the phone<->Pi path works BEFORE setting stt.stream_url and
switching stt.backend to network_whisper. If it can't connect on the shared
iPhone hotspot, that's hotspot client-isolation — switch to the phone's own
hotspot or a WiFi router.
"""
from __future__ import annotations

import sys
import wave


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    url = argv[0]
    seconds = 3.0
    save_path = None
    transcribe = False
    rest = argv[1:]
    i = 0
    while i < len(rest):
        if rest[i] == "--save":
            save_path = rest[i + 1]
            i += 2
        elif rest[i] == "--transcribe":
            transcribe = True
            i += 1
        else:
            seconds = float(rest[i])
            i += 1

    if transcribe:
        return _vad_transcribe(url)

    import av
    import numpy as np
    from av.audio.resampler import AudioResampler

    print(f"opening {url} (timeout 5s)...")
    to = str(5_000_000)
    try:
        container = av.open(url, options={"timeout": to, "rw_timeout": to})
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED to open stream: {exc}")
        print("  -> unreachable. On a shared phone hotspot this is usually client isolation.")
        return 1

    sr = 16000
    resampler = AudioResampler(format="s16", layout="mono", rate=sr)
    want = int(seconds * sr)
    chunks: list[np.ndarray] = []
    got = 0
    try:
        for frame in container.decode(audio=0):
            for rs in resampler.resample(frame):
                a = rs.to_ndarray().reshape(-1).astype("<i2")
                chunks.append(a)
                got += a.size
            if got >= want:
                break
    except Exception as exc:  # noqa: BLE001
        print(f"connected, but decoding failed: {exc}")
        return 1
    finally:
        container.close()

    if not chunks:
        print("connected, but received NO audio frames (is the app actually streaming mic?).")
        return 1

    pcm = np.concatenate(chunks)[:want]
    f = pcm.astype("float32") / 32768.0
    rms = float(np.sqrt(np.mean(f * f))) if f.size else 0.0
    peak = float(np.max(np.abs(f))) if f.size else 0.0
    print(f"OK: {pcm.size} samples @ {sr} Hz mono ({pcm.size / sr:.1f}s)")
    print(f"  RMS={rms:.4f}  peak={peak:.3f}")
    if peak < 0.01:
        print("  WARNING: near-silence. Speak while probing; check the app's mic permission/gain.")
    else:
        print("  signal looks good — speech should transcribe.")

    if save_path:
        with wave.open(save_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        print(f"  saved {save_path}  (play it: pw-play {save_path})")
    return 0


def _vad_transcribe(url: str) -> int:
    """Exercise the exact production path: VAD-gated NetworkMicCapture + Whisper.

    Waits for you to speak (up to vad.max_utterance_s), so timing is forgiving.
    """
    import os
    import sys as _sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from chessmachine.config import load_config
    from chessmachine.voice.network_mic import NetworkMicCapture

    cfg = load_config("config/config.yaml")
    cfg.stt.stream_url = url                       # honor the URL passed on the CLI
    cap = NetworkMicCapture(cfg.stt, cfg.audio)
    print("Loading Whisper model (once)...")
    from faster_whisper import WhisperModel
    model = WhisperModel(cfg.stt.model, device=cfg.stt.device, compute_type=cfg.stt.compute_type)

    print(">>> SPEAK a chess command now (e.g. 'knight to f3')...")
    samples = cap.record_utterance()
    if samples.size == 0:
        print("no speech captured (VAD heard nothing). Speak louder/closer, or raise the app gain.")
        return 1
    segs, _ = model.transcribe(samples, language=cfg.stt.language, beam_size=cfg.stt.beam_size,
                               vad_filter=True, initial_prompt=cfg.stt.prompt or None)
    text = " ".join(s.text.strip() for s in segs).strip()
    print(f"captured {samples.size / 16000:.1f}s")
    print("TRANSCRIPT:", repr(text) if text else "(empty)")
    return 0 if text else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
