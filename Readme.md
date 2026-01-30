Doc https://nvidia.github.io/TensorRT-LLM/installation/linux.html
**Note: Must use cuda compute capability>=8. So don't use Nvidia T4**
```shell
sudo apt install -y python-is-python3 python3-venv
python -m vevn venv
source venv/bin/activate
pip3 install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu130
sudo apt-get -y install libopenmpi-dev python3-dev
pip3 install --upgrade pip setuptools && pip3 install tensorrt_llm
sudo ln -sf /usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0 /usr/lib/x86_64-linux-gnu/libpython3.12.so
```
Run with pytorch backend
```
python main.py
```

Build and run engine
```shell
python ./build_engine.py
python ./run_engine.py
```