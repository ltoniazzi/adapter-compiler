# adapter-compiler
Convert a PyTorch models with adapters to a compiled model to swap adapters at runtime.


## Environment

With CUDA
```bash
python3 -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Withouth CUDA (Linux)


```bash
python3 -m virtualenv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime
```
