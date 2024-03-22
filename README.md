# E-EPIT
Enhance EPIT for Light Field Image Super-Resolution

# Requirements
- Ubuntu 20.04 (18.04 or higher)
- NVIDIA GPU

# Dependencies
- Python 3.8 (> 3.0)
- PyTorch 1.8.2 (>= 1.8)
- NVIDIA GPU + CUDA 10.2 (or >=11.0)
- Follow [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR) to install other libraries

## Commands for Test
* put the image under "/data_for_inference/SR_5x5_4x" folder
  ```
  $ python inference_enhance.py --model_name EPIT_bs --angRes 5 --scale_factor 4 
  ```
* The results are under "log" folder
