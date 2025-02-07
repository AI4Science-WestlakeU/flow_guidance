This is the official implementation of the paper "On the Guidance of Flow Matching"

## Synthetic Dataset Experiments

### Installation
With python 3.11, install the following packages:
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

Note that to run training-based guidance methods, you need to first train the guidance models
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

You can play around other notebooks to see the guidance quality of different methods, 
including gradient, contrastive genergy guidance, and out g^MC.



## Image Inverse Problem Experiments

### Installation

With python 3.11, install the following packages:
```
pip install -r requirements.txt
pip install -e .
```

### Datasets
We downloaded the Celeba-HQ dataset from Kaggle, which contains 30,000 high-quality celebrity faces resampled to 256px. This dataset was used by NVIDIA in the research paper “Progressive Growing of GANs for Improved Quality, Stability, and Variation.” Before feeding the data into the model, the values were normalized to the range of 0 to 1. The data was then randomly split into 8:1:1 ratios for training, testing, and validation, with the corresponding split file being image/gflow_img/config/celeba_hq_splits.json.

### Reproducing the results

First, download the CelebA-HQ dataset and put it in `./data_cache/celeba_hq_256`.
We will release the data cache file and the pretrained model checkpoints after the paper is accepted.
For now, to reproduce the reults, first train the model of CelebA 256 with 
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



## Offline RL Experiments

### Installation

Before installing the offline-rl package, you need to install the mujoco210
```bash
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
# extract to ~/.mujoco/mujoco210
mkdir ~/.mujoco
tar -xvzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
# make sure omesa is installed
apt install libosmesa6-dev libgl1-mesa-glx libglfw3 libglx-mesa0 libgl1-mesa-dri
```

Then, you can install the gflower (which stands for Guided Flow Planner) 
package by running the following commands:
```bash
cd ./gflower
conda env create -f environment.yml

conda activate gflower
pip install -e .
conda install -c conda-forge gcc

# And then you should see the Error because the gym was installed in ver > 0.18 by the auto installation

# if missing crypt.h
apt install libcrypt1
cp /usr/include/crypt.h $CONDA_PREFIX/include/python3.8/crypt.h

# final step: install gym 0.18.3
pip install setuptools==57.5.0
pip install wheel==0.37.0
pip install pip==24.0
pip install gym==0.18.3
```
### Datasets
When running the training scripts, the Locomotion dataset will be automatically downloaded
to ~/.d4rl.

### Reproducing the results

Run bash from the inside the offline_rl folder and run the following command:

1. ```bash run_scripts/train.sh``` to train the base flow matching model
2. ```bash run_scripts/train_value.sh``` to train the value function
3. ```bash run_scripts/eval_gradient.sh``` to evaluate $g^{cov-A}$ and $g^{cov-G}$ of the value function
4. ```bash run_scripts/eval_mc.sh``` to evaluate $g^{MC}$
5. ```bash run_scripts/eval_sim_mc.sh``` to evaluate $g^{sim-MC}$ in simulation
6. ```bash run_scripts/run_guidance_matching.sh``` to train the guidance model $g_\phi$ and evalute its performance.

### Acknowledgements

The implementation is based on the repo of [Diffuser](https://github.com/jannerm/diffuser).
