#requires -Version 5
# Download the onboard model assets into .\models (git-ignored) on Windows.
# Counterpart of fetch_models.sh. Run from anywhere:  pwsh scripts\fetch_models.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path models\kokoro, models\slm, models\stockfish | Out-Null

Write-Host "==> Kokoro TTS (ONNX + voices)  ~336 MB"
curl.exe -L --fail -o models\kokoro\kokoro-v1.0.onnx `
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl.exe -L --fail -o models\kokoro\voices-v1.0.bin `
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

Write-Host "==> SLM GGUF (Llama-3.2-3B-Instruct Q4_K_M)  ~1.9 GB"
curl.exe -L --fail -o models\slm\Llama-3.2-3B-Instruct-Q4_K_M.gguf `
  https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf

Write-Host "==> distil-whisper STT (pre-cache; faster-whisper fetches it on first use otherwise)"
python -c "from faster_whisper import WhisperModel; WhisperModel('distil-small.en', device='cpu', compute_type='int8')"

Write-Host "==> Stockfish (latest Windows AVX2 build)"
$rel = Invoke-RestMethod -Uri https://api.github.com/repos/official-stockfish/Stockfish/releases/latest `
  -Headers @{ "User-Agent" = "chess-machine" }
$asset = $rel.assets | Where-Object { $_.name -match "windows-x86-64-avx2\.zip$" } | Select-Object -First 1
if (-not $asset) { $asset = $rel.assets | Where-Object { $_.name -match "windows-x86-64\.zip$" } | Select-Object -First 1 }
curl.exe -L --fail -o models\stockfish\sf.zip $asset.browser_download_url
Expand-Archive models\stockfish\sf.zip -DestinationPath models\stockfish\extracted -Force
$exe = Get-ChildItem -Recurse models\stockfish\extracted -Filter *.exe | Select-Object -First 1
Copy-Item $exe.FullName models\stockfish\stockfish.exe -Force

Write-Host ""
Write-Host "Done. Model files live in .\models (git-ignored)."
Write-Host "SLM on GPU? See docs/SETUP_WINDOWS.md for the CUDA llama-cpp-python steps."
