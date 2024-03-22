# E-EPIT
Enhance EPIT for Light Field Image Super-Resolution

# Requirements
- Ubuntu 20.04 (18.04 or higher)
- NVIDIA GPU

# Dependencies
- Python 3.8 (> 3.0)
- PyTorch 1.8.2 (>= 1.8)
- NVIDIA GPU + CUDA 10.2 (or >=11.0)
- 

## Commands for Test
* **Run **`test.py`** to perform network inference. Example for test [model_name] on 5x5 angular resolution for 2x/4xSR:**
  ```
  $ python test.py --model_name [model_name] --angRes 5 --scale_factor 2  
  $ python test.py --model_name [model_name] --angRes 5 --scale_factor 4 
  ```
