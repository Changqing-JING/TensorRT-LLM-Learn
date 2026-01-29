Doc https://nvidia.github.io/TensorRT-LLM/installation/linux.html
```shell
python -m vevn venv
source venv/bin/activate
pip3 install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu130
sudo apt-get -y install libopenmpi-dev python3-dev
pip3 install --upgrade pip setuptools && pip3 install tensorrt_llm
sudo ln -sf /usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0 /usr/lib/x86_64-linux-gnu/libpython3.12.so
```

FAQ
1. 
```
    elif major == 7 and minor.isdigit():
                        ^^^^^^^^^^^^^
AttributeError: 'int' object has no attribute 'isdigit'
```
This is a bug of flashinfer
Apply the flashinfer_core_fix.patch