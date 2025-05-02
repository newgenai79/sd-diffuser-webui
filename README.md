<h1 align='center'>WebUI for Image/Video generation</h1>

<h2 align='center'>Supported Text-2-Image models</h2>
<div align='center'>
	<a href='https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0'><img src='https://img.shields.io/badge/Lumina_Image_2.0-blue'></a>
    <a href='https://github.com/NVlabs/Sana'><img src='https://img.shields.io/badge/Sana_Sprint_/_Sana_v1.5_1K-red'></a>
	<a href='https://github.com/NVlabs/Sana'><img src='https://img.shields.io/badge/Sana_1K_2K_4K-blue'></a>
	<a href='https://github.com/THUDM/CogView3'><img src='https://img.shields.io/badge/CogView_3_Plus-red'></a>
	<a href='https://huggingface.co/Alpha-VLLM/Lumina-Next-SFT-diffusers'><img src='https://img.shields.io/badge/Lumina_Next_SFT-blue'></a>
	<a href='https://github.com/ai-forever/Kandinsky-3'><img src='https://img.shields.io/badge/Kandinsky3-red'></a>
	<a href='https://huggingface.co/fal/AuraFlow-v0.3'><img src='https://img.shields.io/badge/AuraFlow_0.3-blue'></a>
	<a href='https://github.com/Wan-Video/Wan2.1'><img src='https://img.shields.io/badge/Wan_2.1_1.3B-red'></a>
	<a href='https://github.com/Tencent/HunyuanDiT'><img src='https://img.shields.io/badge/HunyuanDIT-blue'></a>
	<a href='https://github.com/yandex-research/swd'><img src='https://img.shields.io/badge/SWD_SD3.5_Medium_/_Large-red'></a>
</div>

<h2 align='center'>Video generation</h2>
<div align='center'>
	<a href='https://github.com/ali-vilab/VACE'><img src='https://img.shields.io/badge/VACE-blue'></a>
	<a href='https://github.com/Wan-Video/Wan2.1'><img src='https://img.shields.io/badge/Wan_2.1_1.3B-red'></a>
	<a href='https://github.com/Lightricks/LTX-Video'><img src='https://img.shields.io/badge/LTX_Video_0.9.5-blue'></a>
</div>

<h2 align='center'>Supported Image-2-Video models</h2>
<div align='center'>
    <a href='https://github.com/Lightricks/LTX-Video'><img src='https://img.shields.io/badge/LTX_Video_0.9.1-blue'></a>
</div>

<!--h2 align='center'>Supported Video-2-Video models</h2>
<div align='center'>
    <a href='https://github.com/THUDM/CogVideo'><img src='https://img.shields.io/badge/CogVideoX-blue'></a>
	<a href='https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-Pose'><img src='https://img.shields.io/badge/CogVideoX_Fun_Pose-red'></a>
</div-->
<h2 align='center'>Additional</h2>
<div align='center'>
    <img src='https://img.shields.io/badge/Video_interpolation+upscaler-blue'></a>
	<img src='https://img.shields.io/badge/Image_generation_info-red'></a>
</div>

<h2 align='center'>SVDquant - Nunchaku</h2>
<div align='center'>
    <img src='https://img.shields.io/badge/Flux.1_dev-blue'>
	<img src='https://img.shields.io/badge/Flux.1_schnell-red'>
	<img src='https://img.shields.io/badge/Flux.1_dev_Canny-blue'>
	<img src='https://img.shields.io/badge/Flux.1_dev_Depth-red'>
	<img src='https://img.shields.io/badge/Flux.1_dev_Fill-blue'>
	<img src='https://img.shields.io/badge/Flux.1_dev_Redux-red'>
	<a href='https://huggingface.co/shuttleai/shuttle-jaguar'><img src='https://img.shields.io/badge/Shuttle_jaguar-blue'></a>
</div>
<hr />

<!--p align="center">
    <img src="https://huggingface.co/datasets/newgenai79/Windows_wheels/resolve/main/img/1.png" width="800"/>
    <img src="https://huggingface.co/datasets/newgenai79/Windows_wheels/resolve/main/img/2.png" width="800"/>
    <img src="https://huggingface.co/datasets/newgenai79/Windows_wheels/resolve/main/img/3.png" width="800"/>
<p>
<hr /-->


<!--h3 align='center'>To-Do List</h3>
- [ ] Text 2 image: Lumina-Image 2.0 with Lora support<br />
- [ ] Text 2 image: Wan 2.1 - 1.3B<br /-->

<hr />
<h3 align='center'>Installation guide for Windows</h3>

<b>Step 1: Clone the repository</b>
```	
git clone https://github.com/newgenai79/sd-diffuser-webui
```

<b>Step 2: Navigate inside the cloned repository</b>
```	
cd sd-diffuser-webui
```

<b>Step 3: Create virtual environment: using conda/miniconda</b>

```	
conda create -n sddw python==3.11.9 -y
```
<b>Step 4: Activate virtual environment</b>
```	
conda activate sddw
```

<b>Step 5: Install requirements</b>

```
pip install -r requirements.txt
```

<b>Step 5a: Install SVDQuant/nunchaku requirements (optional)</b>

```
pip install -r requirements_svdquant.txt
```

<b>Step 5b: Install these if you want to use DWPose (optional)</b>

```
pip install -U openmim
```

```
mim install mmcv==2.1.0
```

```
mim install mmpose==1.3.2
```

```
mim install mmdet==3.2.0
```

<b>Step 6: Launch gradio based WebUI</b>
```	
python app.py
```

<hr />
<h3 align='center'>Test environment</h3>
<ul>
	<li>Windows 11</li>
	<li>Nvidia RTX 4060 8 GB Laptop GPU + 8 GB shared RAM + 45 GB Virtual memory / paging file size</li>
	<li><a href="https://www.python.org/downloads/release/python-31011/" target="_blank">Python 3.10.11</a></li>
	<li><a href="https://docs.anaconda.com/miniconda/" target="_blank">Miniconda</a></li>
	<li><a href="https://git-scm.com/" target="_blank">Git</a></li>
	<li><a href="https://developer.nvidia.com/cuda-downloads" target="_blank">CUDA 12.6</a> (Cuda compilation tools, release 12.6, V12.6.77 - Build cuda_12.6.r12.6/compiler.34841621_0)</li>
	<li><a href="https://visualstudio.microsoft.com/vs/community/" target="_blank">Microsoft (R) C/C++ Optimizing Compiler Version 19.29.30157 for x64</a></li>
</ul>
<br />
<ul>
	<li>Linux Mint 22 Wilma</li>
	<li>AMD Ryzen 5700 X3D + Nvidia Geforce RTX 4070 12 GB + 32 GB DDR4 RAM</li>
	<li>Python 3.11.9</li>
	<li>Miniconda</li>
	<li>Cuda V12.0.140 - Build cuda_12.0.r12.0/compiler.32267302_0</li>
</ul>
<hr />

<b>All models inherit the license of the original model.</b>
<hr />
<div align='center'>
Credits: 
<br />
<a href='https://github.com/huggingface/diffusers' target='_blank'>Huggingface/Diffusers team</a>
<br />
<a href='https://github.com/modelscope/DiffSynth-Studio' target='_blank'>Diffsynth Studio team</a>
<br />
<a href='https://github.com/mit-han-lab/nunchaku' target='_blank'>SVDQuant</a>
<br />
</div>
