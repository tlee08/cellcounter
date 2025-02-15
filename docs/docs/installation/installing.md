# Installing

**Step 1:**

Install conda by visiting the [Miniconda downloads page](https://docs.conda.io/en/latest/miniconda.html) and following the prompts to install on your system.

Open the downloaded miniconda file and follow the installation prompts.

**Step 2:**

Open a terminal (Mac or Linux) or Anaconda PowerShell Prompt (Windows) and verify that conda has been installed with the following command.

```zsh
conda --version
```

A response like `conda xx.xx.xx` indicates that it has been correctly installed.

**Step 3:**

Update conda and use the libmamba solver (makes downloading conda programs [MUCH faster](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community)):

```zsh
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

**Step 4:**

Install packages that help Jupyter notebooks read conda environments:

```zsh
conda install -n base nb_conda nb_conda_kernels
```

**Step 5:**

Install the `behavysis_pipeline` conda environment (download [here](https://github.com/tlee08/behavysis_pipeline/blob/main/conda_env.yaml)).

```zsh
conda env create -f path/to/conda_env.yaml
```

**Step 6:**

Install the `DEEPLABCUT` conda environment (download [here](https://github.com/DeepLabCut/DeepLabCut/blob/main/conda-environments/DEEPLABCUT.yaml)).

```zsh
conda env create -f path/to/DEEPLABCUT.yaml
```

**Step 7:**

Install the `simba` conda environment (download [here](https://github.com/tlee08/behavysis_pipeline/blob/main/simba_env.yaml)).

```zsh
conda env create -f path/to/simba_env.yaml
```
