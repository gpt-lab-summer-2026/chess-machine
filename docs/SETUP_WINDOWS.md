# Running the full voice pipeline on Windows (no motors)

This is the Windows counterpart to [SETUP_PI.md](SETUP_PI.md). It runs the whole
conversational loop — **mic → distil-whisper (STT) → SLM → Stockfish → Kokoro
(TTS) → speaker** — on a desktop, with **motion mocked** (no ESP32 / no gantry).
The SLM runs in-process via `llama-cpp-python`; everything else is unchanged.

Tested on Windows 11 + Python 3.12 + an NVIDIA RTX 3080 (CUDA 13 driver).

## 1. Python

Install Python 3.12 (e.g. `winget install Python.Python.3.12`), open a **new**
terminal so it lands on `PATH`, then from the repo root:

```powershell
pip install -e .              # core package (chess, pyyaml)
```

## 2. Voice + audio dependencies

```powershell
pip install faster-whisper kokoro-onnx sounddevice numpy webrtcvad-wheels
```

> **Why `webrtcvad-wheels`?** The PyPI `webrtcvad` (pinned by the `[voice]`
> extra) compiles from C and fails on Windows without MSVC build tools.
> `webrtcvad-wheels` is a drop-in prebuilt wheel exposing the same `webrtcvad`
> import. If you'd rather not use it at all, set `audio.vad.enabled: false` in
> your config (you then record a fixed window instead of auto-detecting silence).

## 3. SLM — `llama-cpp-python`

### With an NVIDIA GPU (recommended)

The default PyPI install is CPU-only; install a prebuilt **CUDA** wheel instead,
then supply the CUDA 12 runtime DLLs it needs (no full CUDA Toolkit required —
just the driver + these pip packages):

```powershell
# CUDA 12.4 wheel works on any CUDA >= 12.4 driver (incl. CUDA 13):
pip install llama-cpp-python --only-binary=:all: `
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Runtime libraries llama.dll links against:
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12

# Put those DLLs where llama.dll will find them (next to it):
$site = python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
Get-ChildItem -Recurse "$site\nvidia" -Filter *.dll |
    ForEach-Object { Copy-Item $_.FullName "$site\llama_cpp\lib" -Force }

# Verify:
python -c "import llama_cpp; print('gpu:', llama_cpp.llama_supports_gpu_offload())"
# -> ggml_cuda_init: found 1 CUDA devices ...  /  gpu: True
```

Then set `slm.n_gpu_layers: -1` in your config to offload the whole model.

### CPU only

```powershell
pip install llama-cpp-python      # CPU wheel
```

Set `slm.n_gpu_layers: 0`. A 3B Q4 model answers in a few seconds on CPU.

## 4. Models, Stockfish (one command)

```powershell
pwsh scripts\fetch_models.ps1
```

Downloads into the git-ignored `models\` (~2.6 GB total): Kokoro ONNX + voices,
the `Llama-3.2-3B-Instruct-Q4_K_M.gguf` SLM, the latest Stockfish Windows binary
(`models\stockfish\stockfish.exe`), and pre-caches the distil-whisper STT model.

## 5. Configuration

A ready-made local config lives at `config\config.yaml` (git-ignored). It sets
`motion.backend: mock`, absolute paths to the Stockfish binary and the GGUF,
`slm.mode: inproc`, and `slm.n_gpu_layers: -1`. Adjust paths if your checkout is
elsewhere. To choose a specific mic/speaker, list devices with:

```powershell
python -c "import sounddevice as sd; print(sd.query_devices())"
```

and set `audio.input_device` / `audio.output_device` to the device indices
(leave `null` to use the Windows defaults).

## 6. Run

```powershell
python -m chessmachine --config config\config.yaml
```

Startup loads Stockfish, the SLM (onto the GPU), whisper, and Kokoro (~20-30 s),
then greets you through the speaker and listens. Speak a move and it replies:

- "knight to f3", "e2 to e4", "castle kingside", "pawn takes d5" — play your move
- "your move" — let the machine move (when it's its turn)
- "who's winning?", "what's the best move?" — spoken analysis
- "set difficulty to hard", "set elo to 1600" — change strength
- "new game", "take that back", "quit"

Moves are mocked at the motion layer — you'll see `move`/`magnet` ops in the
DEBUG log instead of a gantry moving.

## Troubleshooting

- **`Could not find module 'llama.dll' (or one of its dependencies)`** — the CUDA
  runtime DLLs aren't beside `llama.dll`; redo the copy step in §3, or fall back
  to the CPU wheel.
- **`webrtcvad` build error** — use `webrtcvad-wheels`, or set `audio.vad.enabled: false`.
- **No audio / mic not heard** — Windows Settings → Privacy & security →
  Microphone must allow desktop apps; confirm the right device indices in §5.
- **Slow first run** — faster-whisper downloads `distil-small.en` (~200 MB) on
  first use if you skipped the pre-cache.
