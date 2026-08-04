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
llama-server -m models/slm/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  -c 2048 --parallel 1 -t 3 --cache-reuse 256 --host 127.0.0.1 --port 8080
```

**`--parallel 1` is not optional on a Pi.** Without it llama-server auto-selects
`n_parallel = 4`, which does two bad things:

- It allocates **4 × `-c`** worth of f16 KV cache. At 112 KiB/token for this model
  that is ~900 MiB instead of ~225 MiB (most of the 4.3 GiB RSS people report).
- Prompt cache is **per slot**. The client sends `cache_prompt: true` and the app
  warms the ~750-token system+few-shot prefix at startup, but that primes exactly
  one slot; requests then round-robin into cold ones and re-prefill the whole
  prompt at ~15 tok/s (~50 s). One slot means the warm prefix is always hit.

`-t 3` leaves a core for whisper/Kokoro; measured `pp774` was 15.6 tok/s at 3
threads vs 13.6 at 4. Verify the KV size and slot count after starting:

```bash
curl -s localhost:8080/props | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print('slots', d['total_slots'], \
   'n_ctx', d['default_generation_settings']['n_ctx'])"   # expect: slots 1 n_ctx 2048
```

Do **not** bother with `GGML_VULKAN` (the Pi's V3D is ~80-100x slower than the CPU
for this and won't initialize), `GGML_CPU_KLEIDIAI` (no Q4_K kernels, and slower
where kernels exist), or `GGML_BLAS`. A plain
`cmake -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON` build is already optimal here —
`GGML_CPU_REPACK` (on by default) covers Q4_K on the A76's dotprod path.

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

Two independent pieces, because they have different needs: `llama-server` has no
console UI and should be supervised headlessly; `chessmachine` speaks/listens and
is much easier to babysit if its `you> ...` / `[speaker] ...` console mirror is
visible on the screen attached to the Pi.

### 7a. llama-server — systemd service

Run it as its own service so systemd supervises restarts (don't background a
process from `ExecStartPre` — systemd may reap it). The repo ships the unit
file, adjust `User=`/`WorkingDirectory=`/the model filename for your box:

```bash
sudo cp deploy/llama-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server
curl http://127.0.0.1:8080/health   # {"status":"ok"} once the model has loaded
```

### 7b. chessmachine — pick ONE of these

**Option A — console autologin (recommended if a screen is attached).** If
`raspi-config` (or `/etc/systemd/system/getty@tty1.service.d/`) is set to
auto-login a user on the physical console, launch the game from that user's
`~/.bash_profile` so its console output lands on the attached screen instead of
only `journalctl`. `deploy/boot_chessmachine.sh` does the actual work (waits for
`llama-server`, then runs the app and restarts it if it crashes; Ctrl-C stops the
restart loop and hands back a normal shell — it does NOT log you out):

```bash
# in ~/.bash_profile, guarded to the physical console so an ssh/scp login
# (also a "login shell") can't ALSO spawn it:
if [ -t 0 ] && [ "$(tty 2>/dev/null)" = "/dev/tty1" ] \
   && ! pgrep -u "$USER" -f "[c]hessmachine --config" >/dev/null 2>&1; then
    /home/pi/chess-machine/deploy/boot_chessmachine.sh
fi
```

The `[c]hessmachine` bracket is deliberate: `pgrep -f` matches every process's
full command line, including its own, so an unbracketed pattern matches pgrep's
own argv and the guard would never fire.

**Option B — fully headless (no monitor, no autologin).** Run it as a second
systemd service instead:

```ini
# /etc/systemd/system/chess-machine.service
[Unit]
Description=Chess Machine
After=network.target sound.target llama-server.service
Wants=llama-server.service

[Service]
User=pi
WorkingDirectory=/home/pi/chess-machine
ExecStart=/home/pi/chess-machine/.venv/bin/python -m chessmachine --config config/config.yaml
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now chess-machine
```

Its console mirror then only shows up in `journalctl -u chess-machine -f` (no
physical-console output), which is fine for a headless install but makes live
debugging more awkward than option A.

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
