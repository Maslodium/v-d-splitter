$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $Python)) {
  py -3.12 -m venv (Join-Path $Root ".venv")
}
& $Python -m pip install --upgrade pip setuptools wheel
try {
  $null = Get-Command nvidia-smi -ErrorAction Stop
  & nvidia-smi *> $null
  if ($LASTEXITCODE -eq 0) {
    Write-Host "[scan] NVIDIA GPU detected; installing CUDA Torch wheels."
    & $Python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
  } else {
    throw "nvidia-smi failed"
  }
} catch {
  Write-Host "[scan] NVIDIA GPU not detected; installing CPU Torch wheels."
  & $Python -m pip install torch==2.6.0 torchaudio==2.6.0
}
& $Python -m pip install -r (Join-Path $Root "requirements.txt")
& (Join-Path $Root ".venv\Scripts\pythonw.exe") (Join-Path $Root "gui.py")
