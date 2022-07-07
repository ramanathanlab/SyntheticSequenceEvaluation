"""Generate synthetic sequences and embeddings given model weights."""
import os
import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from gene_transformer.model import DNATransformer

# from gene_transformer.model import get_embeddings_using_pt
from gene_transformer.model import LoadPTCheckpointStrategy, LoadDeepSpeedStrategy
from gene_transformer.model import load_from_deepspeed
from gene_transformer.model import inference
from gene_transformer.config import ModelSettings
from gene_transformer.utils import non_redundant_generation, seqs_to_fasta
from gene_transformer.dataset import FASTADataset

from torch.utils.data import DataLoader
from Bio.Seq import Seq
from pathlib import Path
from argparse import ArgumentParser


def generate_fasta(cfg: ModelSettings, pt_path: str, fasta_path: str) -> dict:
    """Given pt or deepspeed file, output generated sequences' fasta files."""
    # obtain model
    print("1")
    if Path(pt_path).suffix == ".pt":
        # load pt file weights
        model = DNATransformer.load_from_checkpoint(
            checkpoint_path=pt_path, strict=False, cfg=cfg
        )
    else:
        # load deepspeed weights
        if cfg.load_from_checkpoint_dir is None:
            raise ValueError("load_from_checkpoint_dir must be set in the config file.")
        model = load_from_deepspeed(
            cfg=cfg, checkpoint_dir=cfg.load_from_checkpoint_dir
        )
    print("2")
    model.cuda()
    print("3")
    # generate non-redundant sequences
    results = non_redundant_generation(model=model.model, tokenizer=model.tokenizer)
    print("4")
    # turn unique sequences to fasta
    unique_seqs = list(results.get("unique_seqs"))
    print("5")
    seqs_to_fasta(seqs=unique_seqs, file_name=fasta_path)
    print("6")
    return results


def fasta_to_embeddings(
    model_strategy, fasta_path, embeddings_output_path
) -> np.ndarray:
    print("_1")
    embeddings = inference(model_strategy, fasta_path, embeddings_output_path)
    print("_2")
    return embeddings


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate sequences and/or embeddings.")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument(
        "--mode", default="get_fasta", type=str, help="get_fasta or get_embeddings"
    )
    parser.add_argument("--pt_path", type=str)
    parser.add_argument("--fasta_path", default="", type=str)
    parser.add_argument(
        "--embeddings_output_path", default="./embeddings.npy", type=Path
    )
    parser.add_argument(
        "--embeddings_model_load", default="pt", type=str, help="deepspeed or pt"
    )
    args = parser.parse_args()
    config = ModelSettings.from_yaml(args.config)

    # set up torch environment
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.set_num_threads(config.num_data_workers)  # type: ignore[attr-defined]
    pl.seed_everything(0)

    print("1_")
    if args.mode == "get_fasta":
        generate_fasta(cfg=config, pt_path=args.pt_path, fasta_path=args.fasta_path)
        print("2_")
    if args.mode == "get_embeddings":
        if not args.fasta_path:
            raise ValueError("Must provide a fasta file to run inference on.")

        if args.embeddings_output_path.exists():
            raise FileExistsError(
                f"embeddings_output_path: {args.embeddings_output_path} already exists!"
            )
        print("3_")
        if args.embeddings_model_load == "pt":
            model_strategy = LoadPTCheckpointStrategy(config, args.pt_path)
            print("4_")
        elif args.inference_model_load == "deepspeed":
            model_strategy = LoadDeepSpeedStrategy(config)
        else:
            raise ValueError(
                f"Invalid embeddings_model_load {args.embeddings_model_load}"
            )
        print("5_")
        fasta_to_embeddings(
            model_strategy, args.fasta_path, args.embeddings_output_path
        )
        print("6_")
