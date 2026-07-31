#!/usr/bin/env python3
"""Test the microphone through the SAME code path the game uses.

    python scripts/mictest.py --devices          # what inputs exist; which one is configured
    python scripts/mictest.py --level            # 3s raw level meter (is the mic alive?)
    python scripts/mictest.py --vad              # VAD capture only, no Whisper (fast)
    python scripts/mictest.py                    # full test: VAD capture -> transcript
    python scripts/mictest.py --save /tmp/m.wav  # keep the audio (play: pw-play /tmp/m.wav)
    python scripts/mictest.py --loop             # keep listening until Ctrl-C

Builds the mic with `create_capture()`, so it tests whichever `stt.backend` is
configured (distil_whisper USB mic / esp32_whisper / network_whisper) and exercises the
real `collect_utterance` VAD path — a pass here means the game will hear you too.

Staged on purpose: --level proves the hardware, --vad proves the end-of-utterance logic,
and the default adds Whisper. If the full test is silent, drop a stage to find out where
it breaks.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import wave

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from chessmachine.config import load_config                      # noqa: E402


def _bar(x: float, width: int = 40) -> str:
    n = min(width, int(x * width * 3))        # x3 so speech is visible, not pinned low
    return "#" * n + "-" * (width - n)


def _capture_rate(cfg) -> int:
    """The rate the mic is actually opened at (mirrors AudioCapture._capture_rate)."""
    return cfg.audio.capture_rate or cfg.audio.sample_rate


def show_devices(cfg) -> int:
    import sounddevice as sd
    want = cfg.audio.input_device
    rate = _capture_rate(cfg)
    print(f"configured audio.input_device = {want!r}")
    print(f"  capture_rate = {rate} Hz (mic)  ->  sample_rate = {cfg.audio.sample_rate} Hz (whisper)")
    print(f"PortAudio default (in, out)   = {sd.default.device}")
    print("\ninput devices:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  [{i}] ch={d['max_input_channels']:<3} native={d['default_samplerate']:.0f}Hz"
                  f"  {d['name']}")
    print(f"\ncan the configured device do {rate} Hz mono int16?")
    try:
        sd.check_input_settings(device=want, samplerate=rate, channels=1, dtype="int16")
        print("  OK")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL: {exc}")
        print("  Set audio.capture_rate to a rate the device supports (one of webrtcvad's")
        print("  8000/16000/32000/48000), or audio.input_device: default to let PipeWire adapt.")
        return 1


def level_meter(cfg, seconds: float) -> int:
    import numpy as np
    import sounddevice as sd

    sr = _capture_rate(cfg)          # meter the RAW hardware, at the rate we open it
    blk = int(sr * 0.1)
    print(f"level meter: {seconds:g}s at {sr} Hz — make some noise (Ctrl-C to stop)")
    peak_all = 0.0
    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                            blocksize=blk, device=cfg.audio.input_device) as st:
            for _ in range(int(seconds / 0.1)):
                buf, _ = st.read(blk)
                x = np.asarray(buf).reshape(-1)
                rms, pk = float(np.sqrt((x * x).mean())), float(np.abs(x).max())
                peak_all = max(peak_all, pk)
                print(f"\r  rms {rms:.4f}  peak {pk:.3f}  |{_bar(rms)}|", end="", flush=True)
    except KeyboardInterrupt:
        pass
    print()
    if peak_all < 0.01:
        print("  NEAR SILENCE — check the mic is selected and unmuted "
              "(amixer -c <card> sget Mic).")
        return 1
    print(f"  peak reached {peak_all:.3f} — the mic is picking up sound.")
    return 0


def capture_once(cfg, cap, stt, save: str | None) -> str:
    import numpy as np

    print("\n>>> SPEAK NOW (e.g. \"knight to f3\"), then pause ~1s ...")
    samples = cap.record_utterance()
    if samples is None or len(samples) == 0:
        print("  nothing captured — the VAD heard no speech.")
        print("  If you did speak: lower audio.vad.aggressiveness (2 -> 1), speak closer,")
        print("  or run --level to confirm the mic has signal at all.")
        return ""
    sr = cfg.audio.sample_rate
    x = np.asarray(samples)
    print(f"  captured {x.size / sr:.2f}s  rms {float(np.sqrt((x*x).mean())):.4f}"
          f"  peak {float(np.abs(x).max()):.3f}")
    if save:
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
        with wave.open(save, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        print(f"  saved {save}   (play it back: pw-play {save})")
    if stt is None:
        return ""
    # Transcribe through the SAME method the game uses (DistilWhisperSTT.transcribe),
    # so this reflects the real level-normalization, padding, beam size, and
    # vad_filter setting — not a hand-rolled transcribe call that could drift from it.
    text = stt.transcribe(x)
    print(f"  TRANSCRIPT: {text!r}" if text else
          "  TRANSCRIPT: (empty — audio captured but Whisper found no words)")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--devices", action="store_true", help="list inputs and exit")
    ap.add_argument("--level", nargs="?", type=float, const=3.0, metavar="SECS",
                    help="raw level meter (default 3s), then exit")
    ap.add_argument("--vad", action="store_true", help="VAD capture only, skip Whisper")
    ap.add_argument("--save", metavar="WAV", help="save the captured utterance")
    ap.add_argument("--loop", action="store_true", help="keep capturing until Ctrl-C")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    print(f"stt.backend = {cfg.stt.backend}   input_device = {cfg.audio.input_device!r}"
          f"   vad aggressiveness = {cfg.audio.vad.aggressiveness}")
    print(f"capture {_capture_rate(cfg)} Hz -> whisper {cfg.audio.sample_rate} Hz")

    if args.devices:
        return show_devices(cfg)
    if args.level is not None:
        return level_meter(cfg, args.level)

    # Preflight: catch a mis-pinned device before the (slow) Whisper load.
    if cfg.stt.backend == "distil_whisper":
        import sounddevice as sd
        try:
            sd.check_input_settings(device=cfg.audio.input_device,
                                    samplerate=_capture_rate(cfg),
                                    channels=1, dtype="int16")
        except Exception as exc:  # noqa: BLE001
            print(f"\nthe configured input can't do {_capture_rate(cfg)} Hz mono: {exc}")
            print("run --devices for the fix")
            return 1

    from chessmachine.voice.stt import create_capture
    cap = create_capture(cfg.stt, cfg.audio)
    print(f"capture = {type(cap).__name__}")

    stt = None
    if not args.vad:
        print(f"loading Whisper ({cfg.stt.model}, {cfg.stt.device}/{cfg.stt.compute_type}) "
              f"— takes a few seconds...")
        from faster_whisper import WhisperModel

        from chessmachine.voice.stt import DistilWhisperSTT
        model = WhisperModel(cfg.stt.model, device=cfg.stt.device,
                             compute_type=cfg.stt.compute_type, cpu_threads=cfg.stt.cpu_threads)
        # Wrap the model in the production STT (skip __init__ so no mic is opened)
        # to reuse its exact transcribe path — same audio prep and decode params.
        stt = DistilWhisperSTT.__new__(DistilWhisperSTT)
        stt.cfg = cfg.stt
        stt.model = model

    got = ""
    try:
        while True:
            got = capture_once(cfg, cap, stt, args.save)
            if not args.loop:
                break
            print("  (Ctrl-C to stop)")
    except KeyboardInterrupt:
        print("\nstopped")
    return 0 if (got or args.vad or args.loop) else 1


if __name__ == "__main__":
    raise SystemExit(main())
