$ErrorActionPreference = "Stop"
Write-Host "Installing Python packages..."
python -m pip install -r "$PSScriptRoot\requirements.txt"
Write-Host "Running unit tests..."
python "$PSScriptRoot\run.py" test
Write-Host "Running RL smoke test..."
python "$PSScriptRoot\run.py" smoke
Write-Host "Installation and smoke test completed."
