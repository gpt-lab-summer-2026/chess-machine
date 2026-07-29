# Phone-as-microphone over WiFi (`network_whisper` STT backend)

Archived setup for using a **phone running the IP Webcam app** as the microphone,
streamed over WiFi to the Pi. Superseded by the MAX4466 (`esp32_whisper`) / USB
mic, but kept here because it works and keeps the voice path entirely off the
Bluetooth radio (so it never contends with the BT speaker output).

The code is still in the tree and dormant — selecting the backend below is all it
takes to re-enable it. Nothing needs to be re-added.

## How it works

The phone runs a mic-streaming server (IP Webcam); the Pi opens that HTTP stream
with **PyAV** only during a listen window, resamples 16 kHz mono, runs the same
`webrtcvad` end-of-utterance logic as the local mic, then closes the stream. Fully
turn-based: WiFi mic and BT speaker are never active at the same time.

## Re-enable it

1. **Same network.** Put the phone and the Pi on the same WiFi so the Pi can reach
   the phone's IP. A shared router is most reliable. A phone hotspot works too but
   may block client-to-client traffic (see caveats).
2. **IP Webcam** on the phone → "Start server". Note the address it shows, e.g.
   `http://172.20.10.3:8080`; the audio endpoint is that + `/audio.opus`.
3. **Probe** reachability + signal from the Pi (speak while it runs):
   ```bash
   cd /home/chess/chess-machine
   .venv/bin/python scripts/mic_probe.py http://172.20.10.3:8080/audio.opus 3 --save /tmp/mic.wav
   pw-play /tmp/mic.wav          # hear it back on the BT speaker
   ```
   Or exercise the real VAD-gated capture + Whisper end to end:
   ```bash
   .venv/bin/python scripts/mic_probe.py http://172.20.10.3:8080/audio.opus --transcribe
   ```
4. **Config** — set the `stt` block in `config/config.yaml`:
   ```yaml
   stt:
     backend: network_whisper
     stream_url: http://172.20.10.3:8080/audio.opus   # your phone's IP Webcam URL
     stream_timeout_s: 5.0
     model: distil-small.en
     device: cpu
     compute_type: int8
     language: en
   ```
5. **Run:** `.venv/bin/chessmachine --config config/config.yaml --no-home` (llama-server up first).

## Code involved (already in the tree)

- `src/chessmachine/voice/network_mic.py` — `NetworkMicCapture` + the testable
  `collect_utterance()` VAD state machine.
- `src/chessmachine/voice/stt.py` — the `network_whisper` branch in `create_stt`.
- `src/chessmachine/config.py` — `SttConfig.stream_url` / `stream_timeout_s`.
- `scripts/mic_probe.py` — reachability/signal probe, `--transcribe` for a live test.

## Caveats / lessons learned

- **Phone mic over Bluetooth is NOT a thing.** Standard BT gives you the phone's
  *media* (A2DP) or *call* audio (HFP), not a general live mic — hence WiFi.
- **Hotspot client isolation.** Verified working on the "Niklas's iPhone" hotspot
  (`172.20.10.0/28`): probe connected, RMS ~0.007, peak ~0.37. Some hotspots block
  client-to-client traffic; if the probe can't connect, switch to the phone's own
  hotspot or a WiFi router.
- **Codec.** `/audio.opus` worked; `/audio.wav` (raw PCM) is a fallback — PyAV
  decodes either, just match `stream_url`.
- **Stream-open latency** (~0.5–1 s) happens inside `listen()` when PyAV connects,
  after the "Your move." cue. Usually masked by reaction time; if the first word
  clips, pre-open the stream on a thread during the cue.
- **Gain/VAD.** The phone's input gain was on the quiet side; if the VAD hears
  nothing, lower `audio.vad.aggressiveness` (2 → 1 or 0) or raise the phone gain.
