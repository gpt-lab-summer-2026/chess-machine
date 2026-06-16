#!/usr/bin/env bash
# Download the onboard model assets into ./models (git-ignored).
# Run from the repo root on the Raspberry Pi:  bash scripts/fetch_models.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p models/kokoro models/slm

echo "==> Kokoro TTS (ONNX + voices)"
# ~310 MB model + ~26 MB voices. Pinned release assets.
curl -L -o models/kokoro/kokoro-v1.0.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o models/kokoro/voices-v1.0.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

echo
echo "==> distil-whisper is fetched automatically by faster-whisper on first run,"
echo "    cached under ~/.cache/huggingface. To pre-download:"
echo "    python -c \"from faster_whisper import WhisperModel; WhisperModel('distil-small.en')\""
echo
echo "==> Stockfish:   sudo apt install stockfish   (or build from source)"
echo
echo "==> SLM (3B GGUF) for llama.cpp — pick one, e.g.:"
echo "    huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \\"
echo "        Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models/slm"
echo "    then run:  llama-server -m models/slm/Llama-3.2-3B-Instruct-Q4_K_M.gguf -c 4096 --port 8080"
echo
echo "Done. Model files live in ./models (git-ignored)."
