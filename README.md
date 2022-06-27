# SyntheticSequenceEvaluation
Pipeline to evaluate synthetic sequences. 

# Installation & Setup
In the base environment of the lambda server, run the following commands:
```
conda create -n mdhpipeline -c rapidsai -c nvidia -c conda-forge  \
    cuml=22.04 python=3.9 cudatoolkit=11.2 \
    jupyterlab pytorch
conda activate mdhpipeline
export IBM_POWERAI_LICENSE_ACCEPT=yes
pip install -U scikit-learn
pip install mdlearn

conda install matplotlib
conda uninstall pytorch
pip uninstall torch
pip install torch --no-cache-dir
pip install plotly==5.8.2

```
# Running Alphafold
First log in to Lambda. Then add the following to .bashrc by: 
```
vim ~/.bashrc
i
```

```
alias alphafold_container='/software/singularity/bin/singularity exec --nv -B /lambda_stor/ /lambda_stor/data/hsyoo/AlphaFoldImage/alphafold.sif bash'
alias alphafold_env='source /opt/miniconda3/etc/profile.d/conda.sh; conda activate alphafold'
```
Press ESC and then type ```:wq!``` then press Enter to save the changes and exit vim.

Then run ```source ~/.bashrc``` or re-login. 
Then run: 
```
/software/singularity/bin/singularity exec --nv -B /lambda_stor/ -B /software /lambda_stor/data/hsyoo/AlphaFoldImage/alphafold.sif bash
alphafold_env
```


# Running the code
TODO: Update with a description of which commands you use to run your software.

If your code is stored in jupyter notebooks, then specify which notebook you run here.

Note: jupyter notebooks should be stored in the `notebooks` folder


# Data
TODO: Descirption of any data you have stored in the `data/` folder.

# Notes
TODO: Any other comments or open issues

# Tips
For extra help on the following topics, please see the links below:
- GitHub: https://skills.github.com/
- Linux/GitHub/ComputerScience: https://missing.csail.mit.edu/
- Python: https://www.youtube.com/playlist?list=PLQVvvaa0QuDeAams7fkdcwOGBpGdHpXln 
