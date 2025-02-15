# Uninstalling

For more information about how to uninstall conda, see [here](https://docs.anaconda.com/free/anaconda/install/uninstall/).

**Step 1:**

Open a terminal (Mac or Linux) or Anaconda PowerShell Prompt (Windows)

**Step 2:**

To uninstall the `behavysis_pipeline_env`, `DEEPLABCUT`, and `simba` conda envs, run the following commands:

```zsh
conda env remove -n behavysis_pipeline_env
conda env remove -n DEEPLABCUT
conda env remove -n simba
```

**Step 3:**

To remove conda, enter the following commands in the terminal.

```zsh
conda install anaconda-clean
anaconda-clean --yes

rm -rf ~/anaconda3
rm -rf ~/opt/anaconda3
rm -rf ~/.anaconda_backup
```

**Step 5:**
Edit your bash or zsh profile so conda it does not look for conda anymore.
Open each of these files (note that not all of them may exist on your computer), `~/.zshrc`, `~/.zprofile`, or `~/.bash_profile`, with the following command.

```zsh
open ~/.zshrc
open ~/.zprofile
open ~/.bash_profile
```