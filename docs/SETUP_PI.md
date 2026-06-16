# Raspberry Pi 5 setup

Target: Raspberry Pi 5 (8 GB), 64-bit Raspberry Pi OS (Bookworm). The full stack
fits comfortably in 8 GB: a 3B Q4_K_M SLM (~2.3 GB) + distil-whisper small int8
(~0.2 GB) + Kokoro (~0.3 GB) + Stockfish.

## 1. System packages

```bash
sudo apt update
sudo apt install -y stockfish portaudio19-dev python3-venv python3-pip git
```

`portaudio19-dev` is needed by `sounddevice` (mic/speaker). `stockfish` provides
the engine on PATH.

## 2. Python environment

```bash
git clone <your-fork-url> chess-machine && cd chess-machine
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-pi.txt
pip install -e .
```

## 3. Models

```bash
bash scripts/fetch_models.sh
```

This downloads Kokoro (ONNX + voices) into `models/kokoro/`, prints the
one-liner to pre-cache distil-whisper, and shows how to fetch a 3B GGUF for the
SLM. Stockfish came from apt in step 1.

### Run the SLM server

The default config uses llama.cpp in **server** mode. Build llama.cpp (or
`pip install llama-cpp-python[server]`) and start it:

```bash
llama-server -m models/slm/Llama-3.2-3B-Instruct-Q4_K_M.gguf -c 4096 --port 8080
```

Prefer in-process instead? Set `slm.mode: inproc` and `slm.model_path`, and
`pip install '.[slm]'`. Your QLoRA fine-tune: merge the adapter, convert to GGUF,
and point `model_path` (inproc) or `llama-server -m` (server) at it.

## 4. Configure

```bash
cp config/config.example.yaml config/config.yaml
```

Edit `config/config.yaml`:

- `motion.serial.port` → your ESP32 port (e.g. `/dev/ttyUSB0`; check with
  `ls /dev/ttyUSB* /dev/ttyACM*`). Add yourself to the `dialout` group:
  `sudo usermod -aG dialout $USER` then re-login.
- `motion.geometry.*` → from `scripts/calibrate.py` (see [HARDWARE.md](HARDWARE.md)).
- `audio.input_device` / `output_device` → run
  `python -c "import sounddevice; print(sounddevice.query_devices())"` to find IDs.
- `app.play_as`, `engine.default_difficulty` → taste.

## 5. Flash the ESP32

See [../firmware/esp32_chess/README.md](../firmware/esp32_chess/README.md), then
verify with `python scripts/serial_console.py --port /dev/ttyUSB0`.

## 6. Run

```bash
# Smoke-test logic only, no hardware/models:
python -m chessmachine --dev

# Real run:
python -m chessmachine --config config/config.yaml
```

## 7. Autostart (optional)

`/etc/systemd/system/chess-machine.service`:

```ini
[Unit]
Description=Chess Machine
After=network.target sound.target

[Service]
User=pi
WorkingDirectory=/home/pi/chess-machine
ExecStartPre=/bin/sh -c 'llama-server -m models/slm/model.gguf -c 4096 --port 8080 &'
ExecStart=/home/pi/chess-machine/.venv/bin/python -m chessmachine --config config/config.yaml
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now chess-machine
```

## Performance notes

- **STT:** `distil-whisper/distil-small.en` with `compute_type: int8` is the
  sweet spot on the Pi 5 CPU. Bump to `distil-medium.en` for accuracy if latency
  allows.
- **SLM:** a 3B Q4_K_M at `n_threads: 4` is responsive since prompts are short
  (intent JSON + brief phrasing). Keep `max_tokens` small (the defaults do).
- **TTS:** Kokoro ONNX runs on CPU; first call loads the model (~1 s), then it's
  fast.
- **Engine:** lower `engine.presets.*.movetime_ms` if move latency matters more
  than strength.
