# SyntheticSequenceEvaluation
Pipeline to evaluate synthetic sequences. 

# Installation & Setup

### Conda setup
In the base environment of the lambda server, run the following commands:
```
conda create -n mdhpipeline -c rapidsai -c nvidia -c conda-forge  \
    cuml=22.04 python=3.9 cudatoolkit=11.2 \
    jupyterlab pytorch
conda activate mdhpipeline
export IBM_POWERAI_LICENSE_ACCEPT=yes
pip install -U scikit-learn

conda install matplotlib
conda uninstall pytorch
pip uninstall torch
pip install torch --no-cache-dir
pip install plotly==5.8.2

```

### PyPI setup
```
python3 -m venv env
source env/bin/activate
pip3 install -U pip setuptools wheel
pip3 install -r requirements/dev.txt
pip3 install -r requirements/requirements.txt
pip3 install -e .
pip3 install torch
pip3 install pytorch-lightning
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
Press ESC and then type ```:wq``` then press Enter to save the changes and exit vim.

Then run ```source ~/.bashrc``` or re-login. 
Then run: 
```
/software/singularity/bin/singularity exec --nv -B /lambda_stor/ -B /software /lambda_stor/data/hsyoo/AlphaFoldImage/alphafold.sif bash
alphafold_env
mkdir examplerun
cd examplerun
cp -r /opt/alphafold/* .
vim test_seq.fasta # this is where you paste in the protein sequence whose 3D structure you would like AlphaFold to predict. 
```

## If running AlphaFold with lambda_fold.py
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python /homes/lind/gene_transformer/examples/folding/lambda_fold.py -i /homes/lind/examplerun/alphafold_fasta -o /homes/lind/MDH-pipeline/run_alphafold/alphafold_results
```
Note that ```-i``` should lead to a directory that contains fasta files, not to the fasta file itself, and ```-o``` should lead to an empty directory.

## If running AlphaFold without using lambda_fold.py
```
./run.sh -d /lambda_stor/data/hsyoo/AlphaFoldData  -o test_out -f test_seq.fasta \
 -t 2020-05-01 -p casp14 -m model_1,model_2,model_3,model_4,model_5 \
-a 0
```
In the last command, ```-a``` indicates the GPU to use. In this case, we are using the 0th GPU. You can use the command ```nvidia-smi``` to check which GPUs are available in this Lambda node.

Now, AlphaFold should start to run. This may take an hour. 

After AlphaFold finishes running, run the following to specify where you would like to store the AlphaFold results. 
```
scp -r username@lambda:/homes/lind/examplerun/ local/directory/path
```
For example, this might be ```scp -r lind@lambda.cels.anl.gov:/homes/lind/examplerun/* ~/Documents/mdh_results``` for me. 

# Running generate.py
## Example 1: Running ```generate_fasta```
```
python3 generate.py --mode get_fasta --config /homes/lind/MDH-pipeline/mdh_gpt.yaml --pt_path /homes/mzvyagin/gpt2_mdh_example/gpt2_earnest_river_122_mdh.pt --fasta_path /homes/lind/MDH-pipeline/fasta/fasta_test3.fasta
```

## Example 2: Running ```fasta_to_embeddings```
```
python3 generate.py --mode get_embeddings --config /homes/lind/MDH-pipeline/mdh_gpt.yaml --pt_path /homes/mzvyagin/gpt2_mdh_example/gpt2_earnest_river_122_mdh.pt --fasta_path /homes/lind/MDH-pipeline/fasta/fasta_test3.fasta --embeddings_output_path /homes/lind/MDH-pipeline/embeddings/embeddings_test3.npy
```

# Running visualize.py
## Example 1: Get t-SNE plots
```
python3 visualize.py --mode tsne --embed_path /homes/mzvyagin/MDH/perlmutter_data/gpt2_generated_embeddings.npy --fasta_path /homes/mzvyagin/MDH/perlmutter_data/globalstep2850.fasta --save_path /homes/lind/MDH-pipeline/visualize/tsne_gen.png --get_subplots True
```
## Example 2: Get UMAP plots
```
python3 visualize.py --mode umap --embed_path /homes/mzvyagin/MDH/perlmutter_data/gpt2_generated_embeddings.npy --fasta_path /homes/mzvyagin/MDH/perlmutter_data/globalstep2850.fasta --save_path /homes/lind/MDH-pipeline/visualize/umap_gen.png --get_subplots True
```
## Example 3: Get embedding distance vs. alignment score plots
```
python3 visualize.py --mode align_plot --embed_path /homes/mzvyagin/MDH/perlmutter_data/inference-test-embeddings.npy --fasta_path /homes/mzvyagin/MDH/gene_transformer/gene_transformer/data/full_mdh_fasta/test.fasta --save_path /homes/lind/MDH-pipeline/visualize/embed_dist_vs_align_score_test.png --alignment_type global --num_workers 70
```
## Example 4: Getting histogram showing mean/max/min alignment scores between two collections of sequences
```
python3 visualize.py --mode align_hist_mean_max_min --embed_path /homes/mzvyagin/mdh_gpt2/likely-wood-857/likely-wood-857-1024concat_seqs_fasta_embeddings_mean.npy --fasta_path /homes/mzvyagin/mdh_gpt2/likely-wood-857/likely-wood-857_1024concat_seqs.fasta --embed_path2 /homes/mzvyagin/mdh_gpt2/likely-wood-857/likely-wood-857-test_fasta_embeddings_mean.npy --fasta_path2 /homes/mzvyagin/MDH/gene_transformer/gene_transformer/data/full_mdh_fasta/test.fasta --save_path /homes/lind/MDH-pipeline/visualize/gen_test_hist_mean_max_min.png --alignment_type global --num_workers 70
```
