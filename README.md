<div align="center">

# **Deep Learning-Assisted Dynamic Mode Decomposition (DA-DMD) for NRB removal in CARS Spectroscopy**
</div>

**DMD** decomposes noisy input (CARS spectra) into different modes based on their frequency. **Deep learning** part uses SE Block for channel atention to weigh relevance of modes and then a CNN Block to extract the final clean output (Raman spectra). The noise (Non-resonant Background) has low frequency while the clean spectra (Raman signatures) have higher frequencies. This criteria makes the background removal possible. To know more (link to paper). 

<p align="center">
  <img src="images/Methods.png" width="800" alt="DA-DMD method">
  <br>
  <em>DA-DMD method.</em>
</p>

## Usage

**Setup the project:**

* **Step 1:** Clone this repository and create a virtual environment to isolate the dependencies.

```bash
# Clone repo with default name DA_DMD
git clone https://github.com/spectra-analysis/DA_DMD.git
cd DA_DMD

# Create and activate a conda environment
conda create -n "env_dadmd" python=3.10
conda activate env_dadmd
```
* **Step 2:** Install required packages as given in [`requirements.txt`](./requirements.txt), preferably with the same version.
  
```bash
pip install -r requirements.txt
```


**Get started:** Understand the code usage with [`example_dadmd.ipynb`](./example_dadmd.ipynb) notebook that illustrates a training and testing example.

**Training:** You may train a new DA-DMD model using [`train_dadmd.py`](./train_dadmd.py). You may use the given synthetic CARS-Raman data pair or use synthetic generator [1](https://github.com/crimson-project-eu/NRB_removal/blob/main/synthetic-data-generator.py), [2](https://github.com/Junjuri/LUT/blob/main/RSS_Advances_CNN_to_train_with_different_NRBs.py) or [3](https://github.com/Valensicv/Specnet/blob/master/Specnet_Published.ipynb).

**Testing:** You may test the trained model using [`test_dadmd.py`](./test_dadmd.py). 

>**Note:** We did our experiments on Ubuntu 22.04 with Python 3.10, PyTorch 2.2.1 and CUDA 12.1.

## Citation
**Authors:** Adithya Ashok Chalain Valapil, Carl Messerschmidt, Maha Shadaydeh, Michael Schmitt, Jürgen Popp, Joachim Denzler.

**Publisher:** Proceedings of the German Conference on Pattern Recognition (GCPR) 2025

**BibTeX:**

```bibtex
@inproceedings{valapil2025dadmd,
  title     = {Deep Learning-Assisted Dynamic Mode Decomposition for NRB removal in CARS Spectroscopy},
  author    = {Adithya Ashok {Chalain Valapil} and Carl Messerschmidt and Maha Shadaydeh and Michael Schmitt and Jürgen Popp and Joachim Denzler},
  booktitle = {Proceedings of the German Conference on Pattern Recognition (GCPR)},
  note = {Accepted},
  year      = {2025}
}
```
