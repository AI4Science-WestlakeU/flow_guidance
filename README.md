# On the Guidance of Flow Matching (ICML 2025 spotlight)

[[paper](https://openreview.net/forum?id=pKaNgFzJBy)] | [[arXiv](https://arxiv.org/abs/2502.02150)] | [[model weights](https://drive.google.com/drive/folders/1j4-4xdipNlmhi-eyuVHDXsE4KjYWzkSp?usp=sharing)]

Official repo for the paper [On the Guidance of Flow Matching](https://arxiv.org/abs/2502.02150)

[Ruiqi Feng](https://weenming.github.io/), [Chenglei Yu](https://scholar.google.com/citations?user=lzYSFx4AAAAJ&hl=zh-CN)\*, [Wenhao Deng](https://w3nhao.github.io/)\*, [Peiyan Hu](https://peiyannn.github.io/), [Tailin Wu](https://tailin.org)†

ICML 2025 spotlight

We introduce the first framework for general flow matching guidance, from which new guidance methods are derived and many classical guidance methods are covered as special cases.

![](./fig1.png)

# Synthetic Dataset Experiments

### Installation
In the `synthetic` folder and with Python 3.11 installed, install the following packages:
```bash
conda env create -f environment.yml
conda activate guided_flow
pip install -e .
```

### Datasets
The datasets are generated during training, as the distributions are relatively simple

### Reproducing the Results

First, train the base models:

```bash
bash script/train_cfm.sh
```

Note that to run training-based guidance methods, you need first to train the guidance models
using:
```bash
bash script/train_guidance_matching.sh
```
and 
```bash
bash script/train_ceg.sh
```
for the exact diffusion guidance of the contrastive energy guidance.

Then, guidance methods can be evaluated using the notebook:
`notebooks/fig.ipynb`
and
`notebooks/mc.ipynb`
to reproduce Figure 2 and Figure 4.

You can play around with other notebooks to see the guidance quality of different methods, 
including gradient, contrastive energy guidance, and out g^MC.



# Image Inverse Problem Experiments

### Installation

In the `image` folder and with Python 3.11 installed, install the following packages:
```
pip install -r requirements.txt
pip install -e .
```

### Datasets
We downloaded the Celeba-HQ dataset from Kaggle, which contains 30,000 high-quality celebrity faces resampled to 256px. This dataset was used by NVIDIA in the research paper “Progressive Growing of GANs for Improved Quality, Stability, and Variation.” Before feeding the data into the model, the values were normalized to the range of 0 to 1. The data was then randomly split into 8:1:1 ratios for training, testing, and validation, with the corresponding split file being image/gflow_img/config/celeba_hq_splits.json.

### Reproducing the results

First, download the CelebA-HQ dataset and put it in `./data_cache/celeba_hq_256`.
We will release the data cache file and the pre-trained model checkpoints after the paper is accepted.
For now, to reproduce the results, first train the model of CelebA 256 with 
```
accelerate launch run/main_train.py
```
The model will be saved in `./results/{cfm,ot}_punet256_celeba256`.

Then, evaluate different guidance methods on the three inverse problems using 
1. `bash scripts/g_cov_A.sh`
2. `bash scripts/g_cov_G.sh`
3. `bash scripts/PiGDM.sh`
4. `bash scripts/g_sim_inv_A.sh`
5. `bash scripts/g_MC.sh`



# Offline RL Experiments

### Installation

Change to the `offline_rl` folder. Before installing the offline-rl package, you need to install the mujoco210
```bash
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
# extract to ~/.mujoco/mujoco210
mkdir ~/.mujoco
tar -xvzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco

# Add these lines to your ~/.bashrc or run them in your current shell
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc 
echo 'export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210' >> ~/.bashrc
```

Then, you can install the `gflower` (which stands for Guided Flow Planner) 
package by running the following commands:
```bash
cd ./offline_rl
conda env create -f environment.yml

conda activate gflower

# Install the local package
pip install -e .

# And then you should see the Error because the gym was installed in ver > 0.18 by the auto installation, so you need to install gym 0.18.3

# If you are sudoers, make sure omesa is installed
# apt update
# apt install libosmesa6-dev libgl1-mesa-glx libglfw3 libglx-mesa0 libgl1-mesa-dri

# Install gcc toolchain + GL headers from conda-forge (non-root friendly)
conda install -c conda-forge gcc glew mesalib -y

# final step: install gym 0.18.3
pip install setuptools==57.5.0
pip install wheel==0.37.0
pip install pip==24.0
pip install gym==0.18.3
```

```bash
# (Optional but recommended) rebuild mujoco-py with the conda GCC toolchain.
conda activate gflower
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export LDSHARED="$CC -shared"
export CFLAGS="-Wno-error=incompatible-pointer-types"
export LDFLAGS="-Wl,-rpath,$CONDA_PREFIX/lib"

pip install --force-reinstall --no-build-isolation --no-cache-dir "cython<3" "mujoco-py==2.1.2.14"
```
### Datasets
When running the training scripts, the Locomotion dataset will be automatically downloaded
to ~/.d4rl.

### Reproducing the results

Activate the environment before running the scripts:
```bash
conda activate gflower

# Keep the same compiler exports when training/evaluating to avoid the GCC pointer
# errors (`WrapMjVisual_* incompatible pointer type`)
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export LDSHARED="$CC -shared"
export CFLAGS="-Wno-error=incompatible-pointer-types"
export LDFLAGS="-Wl,-rpath,$CONDA_PREFIX/lib"
```

Run bash from inside the offline_rl folder and run the following command:

1. ```bash run_scripts/train.sh``` to train the base flow matching model
2. ```bash run_scripts/train_value.sh``` to train the value function
3. ```bash run_scripts/eval_gradient.sh``` to evaluate $g^{cov-A}$ and $g^{cov-G}$ of the value function
4. ```bash run_scripts/eval_mc.sh``` to evaluate $g^{MC}$
5. ```bash run_scripts/eval_sim_mc.sh``` to evaluate $g^{sim-MC}$ in simulation
6. ```bash run_scripts/run_guidance_matching.sh``` to train the guidance model $g_\phi$ and evalute its performance.

#### Troubleshoot
1. If you are not sudoers, install the mesa libraries (for omesa) and glew:
```bash
conda install -c conda-forge glew mesalib -y
```

2. If `cython`/`pip` complain about missing `crypt.h`, run:
```bash
apt install libcrypt1
conda activate gflower
cp /usr/include/crypt.h $CONDA_PREFIX/include/python3.8/crypt.h
```

3. `mujoco-py` pointer / GLEW errors on GCC 14+:
   - When compiling you might see `glReadPixels makes pointer from integer`, missing `GL/glew.h`, or `WrapMjVisual_* incompatible pointer type`.
   - Apply the following patch to `${CONDA_PREFIX}/lib/python3.8/site-packages/mujoco_py/gl/eglshim.c` **before** reinstalling `mujoco-py`:
     ```diff
     #include <GL/glew.h>
     +#include <stdio.h>
     +#include <string.h>
     +#include <stdint.h>
     ...
     -    glReadPixels(..., bufferOffset * viewport.width * viewport.height * 3);
     +    glReadPixels(...,
     +                 (GLvoid *)(uintptr_t)(bufferOffset * viewport.width * viewport.height * 3));
     ...
     -    glReadPixels(...,
     -                 bufferOffset * viewport.width * viewport.height * sizeof(short));
     +    glReadPixels(...,
     +                 (GLvoid *)(uintptr_t)(bufferOffset * viewport.width * viewport.height * sizeof(short)));
     ```
   - This adds the missing headers and casts so GCC 12+/15 compiles cleanly with GLEW.


### Acknowledgements

The implementation is based on the repo of [Diffuser](https://github.com/jannerm/diffuser).


# Citation
If you find our work and/or our code useful, please cite us via:

```bibtex
@inproceedings{
  feng2025on,
  title={On the Guidance of Flow Matching},
  author={Feng, Ruiqi and Yu, Chenglei and Deng, Wenhao and Hu, Peiyan and Wu, Tailin},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=pKaNgFzJBy}
}
```

