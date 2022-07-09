"""Generate synthetic sequences and embeddings given model weights."""
import logging
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytorch_lightning as pl
import torch
from gene_transformer.config import ModelSettings
from gene_transformer.model import (
    LoadDeepSpeedStrategy,
    LoadPTCheckpointStrategy,
    ModelLoadStrategy,
    inference,
)
from gene_transformer.utils import non_redundant_generation, seqs_to_fasta

logger = logging.getLogger(__name__)


def generate_fasta(
    model_strategy: ModelLoadStrategy, fasta_path: Path, num_seqs: int
) -> Dict[str, List[str]]:
    """Given pt file or deepspeed directory, output generated sequences' fasta file.

    Parameters
    ----------
    model_strategy : ModelLoadStrategy
        Model used to generate sequences, depending on whether a pt file
        or a deepspeed directory is given.
    fasta_path : Path
        Path to save fasta sequences.
    num_seqs : int
        Number of non-redundant sequences to generate.

    Returns
    -------
    Dict[str, List[str]]
        Unique generated sequences.
    """
    # obtain model
    model = model_strategy.get_model()
    model.cuda()
    # generate non-redundant sequences
    results = non_redundant_generation(
        model=model.model, tokenizer=model.tokenizer, num_seqs=num_seqs
    )
    results["unique_seqs"] = list(results["unique_seqs"])
    # turn unique sequences to fasta
    seqs_to_fasta(seqs=results["unique_seqs"], file_name=fasta_path)
    return results


def fasta_to_embeddings(
    model_strategy: ModelLoadStrategy, fasta_path: Path, embeddings_output_path: str
) -> np.ndarray:
    """Run inference to generate embeddings for generated sequences.

    Parameters
    ----------
    model_strategy : ModelLoadStrategy
        Model used to generate sequences, depending on whether a pt file or deepspeed is given.
    fasta_path : str
        Path to access fasta sequences.
    embeddings_output_path : str
        Path to save generated embeddings.

    Returns
    -------
    np.ndarray
        Generated embeddings.
    """
    embeddings = inference(model_strategy, str(fasta_path), embeddings_output_path)
    return embeddings


def parse_args() -> Namespace:
    """Parse command line arguments.

    Returns
    -------
    Namespace
        Parsed arguments.
    """
    parser = ArgumentParser(description="Generate sequences or embeddings.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help=".yaml file containing configuration settings",
    )
    parser.add_argument(
        "--mode",
        default="get_fasta",
        type=str,
        required=True,
        help="get_fasta or get_embeddings",
    )
    parser.add_argument(
        "--pt_path", type=str, required=True, help="Path to access pt file"
    )
    parser.add_argument(
        "--fasta_path",
        default="",
        type=Path,
        help="Path to save or access fasta sequences",
    )
    parser.add_argument(
        "--embeddings_output_path",
        default="./embeddings.npy",
        type=Path,
        help="Path to save generated embeddings",
    )
    parser.add_argument(
        "--embeddings_model_load", default="pt", type=str, help="deepspeed or pt"
    )
    parser.add_argument(
        "--num_seqs",
        default=5,
        type=int,
        help="Number of non-redundant sequences to generate",
    )
    return parser.parse_args()


def main():
    """Run generate_fasta or fasta_to_embeddings to generate fasta or embeddings based on parsed arguments.

    Parameters
    ----------
    args : Namespace
        Parsed arguments.

    Raises
    ------
    ValueError
        If embeddings_model_load is not either "pt" or "deepspeed."
    ValueError
        If fasta_path is not provided to generate embeddings.
    FileExistsError
        If embeddings_output_path leads to a file that already exists.
    ValueError
        If mode is not either "get_fasta" or "get_embeddings."
    """
    config = ModelSettings.from_yaml(args.config)

    # set up torch environment
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.set_num_threads(config.num_data_workers)  # pylint: disable=no-member
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
        generate_fasta(
            model_strategy=model_strategy,
            fasta_path=args.fasta_path,
            num_seqs=args.num_seqs,
        )
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
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    args = parse_args()
    main()
