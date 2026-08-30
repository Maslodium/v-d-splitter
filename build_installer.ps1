$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $Python)) {
  py -3.12 -m venv (Join-Path $Root ".venv")
}
& $Python -m pip install --upgrade pip setuptools wheel pyinstaller
$Payload = Join-Path $Root "payload"
if (Test-Path $Payload) { Remove-Item -Recurse -Force $Payload }
New-Item -ItemType Directory -Force $Payload | Out-Null
Get-ChildItem $Root -Force | Where-Object {
  $_.Name -notin @(".git", ".venv", "build", "dist", "payload", "__pycache__") -and
  $_.Name -notlike "*.spec"
} | ForEach-Object {
  $dest = Join-Path $Payload $_.Name
  if ($_.PSIsContainer) { Copy-Item $_.FullName $dest -Recurse -Force }
  else { Copy-Item $_.FullName $dest -Force }
}
& $Python -m PyInstaller `
  --name "Install V-D Splitter" `
  --onefile `
  --console `
  --add-data "$Payload;payload" `
  (Join-Path $Root "installer\bootstrap_installer.py")
Write-Host "Built:" (Join-Path $Root "dist\Install V-D Splitter.exe")
