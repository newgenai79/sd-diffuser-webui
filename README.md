<h1 align='center'>WebUI for Image/Video generation</h1>

<h2 align='center'>Supported Text-2-Image models</h2>
<div align='center'>
	<a href='https://huggingface.co/ostris/Flex.1-alpha'><img src='https://img.shields.io/badge/Flex.1_alpha-red'></a>
    <a href='https://github.com/NVlabs/Sana'><img src='https://img.shields.io/badge/Sana_2K_4K-blue'></a>
	<a href='https://github.com/Tencent/HunyuanDiT'><img src='https://img.shields.io/badge/HunyuanDIT-red'></a>
	<a href='https://github.com/THUDM/CogView3'><img src='https://img.shields.io/badge/CogView_3_Plus-blue'></a>
	<a href='https://github.com/Alpha-VLLM/Lumina-T2X'><img src='https://img.shields.io/badge/Lumina-red'></a>
	<a href='https://github.com/ai-forever/Kandinsky-3'><img src='https://img.shields.io/badge/Kandinsky3-blue'></a>
</div>

<h2 align='center'>Supported Text-2-Video models</h2>
<div align='center'>
    <a href='https://github.com/Lightricks/LTX-Video'><img src='https://img.shields.io/badge/LTX_Video_0.9.1-blue'></a>
	<a href='https://github.com/Tencent/HunyuanVideo/'><img src='https://img.shields.io/badge/HunyuanVideo_GGUF-red'></a>
	<a href='https://github.com/Tencent/HunyuanVideo/'><img src='https://img.shields.io/badge/HunyuanVideo_BitsnBytes-blue'></a>
	<a href='https://github.com/THUDM/CogVideo'><img src='https://img.shields.io/badge/CogVideoX_v1.5_5b-red'></a>
</div>

<h2 align='center'>Supported Image-2-Video models</h2>
<div align='center'>
    <a href='https://github.com/Lightricks/LTX-Video'><img src='https://img.shields.io/badge/LTX_Video_0.9.1-blue'></a>
	<a href='https://github.com/Tencent/HunyuanVideo/'><img src='https://img.shields.io/badge/HunyuanVideo_BitsnBytes-red'></a>
</div>
<hr />
<p align="center">
	<h3 align='center'>Sana 2K</h3>
    <img src="https://huggingface.co/datasets/newgenai79/Windows_wheels/resolve/main/img/Sana1.jpg" width="800"/>
	<h3 align='center'>CogView3-Plus</h3>
    <img src="https://huggingface.co/datasets/newgenai79/Windows_wheels/resolve/main/img/CogView3Plus1.jpg" width="800"/>
	<h3 align='center'>HunyuanDiT</h3>
    <img src="https://huggingface.co/datasets/newgenai79/Windows_wheels/resolve/main/img/HunyuanDiT.jpg" width="800"/>
	<h3 align='center'>Kandinsky3</h3>
    <img src="https://huggingface.co/datasets/newgenai79/Windows_wheels/resolve/main/img/kandinsky3.jpg" width="800"/>
	<h3 align='center'>Lumina</h3>
    <img src="https://huggingface.co/datasets/newgenai79/Windows_wheels/resolve/main/img/Lumina1.jpg" width="800"/>

<p>

<hr />
<h3 align='center'>Installation guide</h3>

<b>Step 1: Clone the repository</b>
```	
git clone https://github.com/newgenai79/diffuser_t2i_webui
```

<b>Step 2: Navigate inside the cloned repository</b>
```	
cd diffuser_t2i_webui
```

<b>Step 3: Create virtual environment</b>
```	
conda create -n dt2i python==3.10.11 -y
```

<b>Step 4: Activate virtual environment</b>
```	
conda activate dt2i
```

<b>Step 5: Install requirements</b>

Windows 
```
pip install -r requirements.txt
```


Linux 
```
pip3 install -r requirements.txt
```

<b>Step 6: Launch gradio based WebUI</b>
```	
python app.py
```

<hr />
<h3 align='center'>Test environment</h3>
<ul>
	<li>Windows 11</li>
	<li>Nvidia RTX 4060 8 GB Laptop GPU + 8 GB shared RAM + 45536 GB Virtual memory / paging file size</li>
	<li><a href="https://www.python.org/downloads/release/python-31011/" target="_blank">Python 3.10.11</a></li>
	<li><a href="https://docs.anaconda.com/miniconda/" target="_blank">Miniconda</a></li>
	<li><a href="https://git-scm.com/" target="_blank">Git</a></li>
	<li><a href="https://developer.nvidia.com/cuda-downloads" target="_blank">CUDA 12.6</a> (Cuda compilation tools, release 12.6, V12.6.77 - Build cuda_12.6.r12.6/compiler.34841621_0)</li>
	<li><a href="https://visualstudio.microsoft.com/vs/community/" target="_blank">Microsoft (R) C/C++ Optimizing Compiler Version 19.29.30157 for x64</a></li>
</ul>

<div align='center'>
Credits: <a href='https://github.com/huggingface/diffusers' target='_blank'>Huggingface/Diffusers team</a>
Note: We are not affliated with Diffusers team
</div>
