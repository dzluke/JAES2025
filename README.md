# Generating Music Reactive Videos by Applying Network Bending to Stable Diffusion
Luke Dzwonczyk, Carmine-Emanuele Cella, and David Ban | Journal of the Audio Engineering Society, June 2025

Abstract: In this paper we present the first steps towards the creation of a tool which enables artists to create music visualizations using pre-trained, generative, machine learning models. First, we investigate the application of network bending, the process of applying transforms within the layers of a generative network, to image generation diffusion models by utilizing a range of point-wise, tensor-wise, and morphological operators. We identify a number of visual effects that result from various operators, including some that are not easily recreated with standard image editing tools. We find that this process allows for continuous, fine-grain control of image generation which can be helpful for creative applications. Next, we generate music-reactive videos using Stable Diffusion by passing audio features as parameters to network bending operators. Finally, we comment on certain transforms which radically shift the image and the possibilities of learning more about the latent space of Stable Diffusion based on these transforms. This paper is an extended version of the paper ``Network Bending of Diffusion Models" which appeared in the 27th International Conference on Digital Audio Effects.


The code in this repository began as a fork of [Stream Diffusion](https://github.com/cumulo-autumn/StreamDiffusion).
## Installation instructions 

### Windows 

You must first clone the StreamDiffusion repo. This is only for the purpose of creating the necessary conda environment.
```
git clone https://github.com/cumulo-autumn/StreamDiffusion.git
cd StreamDiffusion

conda create -n streamdiffusion python=3.10
conda activate streamdiffusion

pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu121
python setup.py develop easy_install streamdiffusion[tensorrt]

# some fixes to the StreamDiffusion install
conda install "numpy<2"
pip install transformers==4.28.0
pip install huggingface-hub==0.25.2

python -m streamdiffusion.tools.install-tensorrt

```
A simple way to see if StreamDiffusion was properly installed:
```
# while in the StreamDiffusion repo
cd examples
python benchmark/multi.py
```

Once you confirm that StreamDiffusion is properly running on your system, then you can install the necessary dependencies for this repository. Start by cloning this repo:
```
git clone https://github.com/dzluke/JAES2025.git
cd JAES2025

conda activate streamdiffusion
conda install -c conda-forge librosa
conda install scipy
```
Now you can run the code in this repo using the `streamdiffusion` conda environment.

If you get the error `OSError: cannot load library 'libsndfile.dll': error 0x7e`, try:
``` 
pip uninstall soundfile
pip install soundfile
```






