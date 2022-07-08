"""Generate synthetic sequences and embeddings given model weights."""
import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

# from Bio.Seq import Seq
from gene_transformer.config import ModelSettings

# from gene_transformer.dataset import FASTADataset
from gene_transformer.model import (  # DNATransformer,; load_from_deepspeed,
    LoadDeepSpeedStrategy,
    LoadPTCheckpointStrategy,
    ModelLoadStrategy,
    inference,
)
from gene_transformer.utils import non_redundant_generation, seqs_to_fasta

# from torch.utils.data import DataLoader
# from tqdm import tqdm

logger = logging.getLogger(__name__)


def generate_fasta(model_strategy: ModelLoadStrategy, fasta_path: str) -> dict:
    """Given pt or deepspeed file, output generated sequences' fasta files."""
    # obtain model
    model = model_strategy.get_model()
    model.cuda()
    # generate non-redundant sequences
    results = non_redundant_generation(model=model.model, tokenizer=model.tokenizer)
    # turn unique sequences to fasta
    unique_seqs = list(results.get("unique_seqs"))
    seqs_to_fasta(seqs=unique_seqs, file_name=fasta_path)
    return results


def fasta_to_embeddings(
    model_strategy: ModelLoadStrategy, fasta_path: str, embeddings_output_path: str
) -> np.ndarray:
    embeddings = inference(model_strategy, fasta_path, embeddings_output_path)
    return embeddings


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate sequences and/or embeddings.")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument(
        "--mode",
        default="get_fasta",
        type=str,
        required=True,
        help="get_fasta or get_embeddings",
    )
    parser.add_argument("--pt_path", type=str, required=True)
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

    # get model_strategy
    if args.embeddings_model_load == "pt":
        model_strategy = LoadPTCheckpointStrategy(config, args.pt_path)
    elif args.inference_model_load == "deepspeed":
        model_strategy = LoadDeepSpeedStrategy(config)
    else:
        raise ValueError(f"Invalid embeddings_model_load {args.embeddings_model_load}")

    # run corresponding function
    if args.mode == "get_fasta":
        generate_fasta(model_strategy=model_strategy, fasta_path=args.fasta_path)
    elif args.mode == "get_embeddings":
        if not args.fasta_path:
            raise ValueError("Must provide a fasta file to run inference on.")
        if args.embeddings_output_path.exists():
            raise FileExistsError(
                f"embeddings_output_path: {args.embeddings_output_path} already exists!"
            )
        fasta_to_embeddings(
            model_strategy=model_strategy,
            fasta_path=args.fasta_path,
            embeddings_output_path=args.embeddings_output_path,
        )
