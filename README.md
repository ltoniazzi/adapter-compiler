# adapter-compiler
Convert PyTorch models with adapters to compiled models to swap adapters at runtime.


## Run
```bash
python3 adapter_compiler/main.py
```


## Environment (Linux)

With CUDA
```bash
python3 -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Withouth CUDA 


```bash
python3 -m virtualenv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime
```
