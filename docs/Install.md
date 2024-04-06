# Install m-LoRA

## Linux (Ubuntu, Debian, Fedora, etc.)

### Requirements

- One or more NVIDIA GPUs
  - At least 16GB VRAM per card
  - Cards with Ampere or newer architecture will perform faster
- Installation of the latest graphics driver and CUDA toolkit
- `conda` installed and configured
- Internet connection for automated tasks

### Steps

```bash
# Clone Repository
git clone https://github.com/mikecovlee/mlora
cd mlora
# Optional but recommended
conda create -n mlora python=3.10
conda activate mlora
# Install requirements
pip install -r requirements.txt
# Install extra requirements on Linux
bash install_linux.sh
```

## Verification

```python
import mlora
mlora.setup_logging("INFO")
mlora.get_backend().check_available()
```

Expected output:

```
m-LoRA: NVIDIA CUDA initialized successfully.
```

## macOS

**Note**: macOS with MPS support is an experimental feature.

### Requirements

- Macintosh with Apple Silicon (recommended) or AMD GPUs
- macOS 12.3 or later
- Xcode command-line tools: `xcode-select --install`
- `conda` installed and configured
- Internet connection for automated tasks

## Steps

```bash
# Clone Repository
git clone https://github.com/mikecovlee/mlora
cd mlora
# Optional but recommended
conda create -n mlora python=3.10
conda activate mlora
# Install requirements
pip install -r requirements.txt
```

## Verification

```python
import mlora
mlora.setup_logging("INFO")
mlora.get_backend().check_available()
```

Expected output:

```
m-LoRA: APPLE MPS initialized successfully.
```