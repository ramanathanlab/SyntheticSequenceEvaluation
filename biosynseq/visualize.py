"""Visualize synthetic sequences using t-SNE, UMAP, and other visualization schemes."""
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

from Bio import SeqIO
import numpy as np
import pandas as pd


from biosynseq import metrics

logger = logging.getLogger("biosynseq.visualize")

def get_embed_avg(embed_path: Path) -> np.ndarray:
    """Given a path to embeddings, return the average embedding.
    Parameters
    ----------
    embed_path : Path
        Path to access embeddings.
        The embeddings could be for the training, validation, testing, or generated sequences.
    Returns
    -------
    np.ndarray
        Average embedding.
    """
    embed = np.load(embed_path)
    embed_avg = embed.mean(axis=1)
    return embed_avg

def get_paint_df(fasta_path: Path) -> pd.DataFrame:
    """Given a path to a fasta file,
    return a dataframe with information of the GC content and sequence length of each DNA sequence,
    as well as the molecular weight and isoelectric point of the protein translated from each DNA sequence.

    Parameters
    ----------
    fasta_path : Path
        Path to access fasta sequences.

    Returns
    -------
    pd.DataFrame
        Dataframe containing information of the GC content, sequence length,
        molecular weight, and isolelectric point derived from each DNA sequence.
    """
    seqs = list(SeqIO.parse(fasta_path, "fasta"))
    # translate DNA seqs to protein seqs; stop translation at the first in-frame stop codon
    protein_seqs = [s.translate(to_stop=True) for s in seqs]
    paint_df = pd.DataFrame(
        {
            "GC": metrics.gc_content(seqs),
            "SequenceLength": metrics.seq_length(seqs),
            "MolecularWeight": metrics.molecular_weight(protein_seqs),
            "IsoelectricPoint": metrics.isoelectric_point(protein_seqs),
        }
    )
    return paint_df


def run_tsne(embed_data: np.ndarray) -> np.ndarray:
    """Given 2-dimensional sequence embeddings, return the transformed data using t-SNE.

    Parameters
    ----------
    embed_data : np.ndarray
        Sequence embeddings to be transformed by t-SNE. Must be 2-dimensional.

    Returns
    -------
    np.ndarray
        Transformed embeddings after running t-SNE.
    """
    from cuml.manifold import TSNE

    # embed_data must be 2D since rapidsai only supports 2D data
    model = TSNE(n_components=2, method="barnes_hut")
    data_proj = model.fit_transform(embed_data)
    return data_proj


def plot_tsne(
    data_proj: np.ndarray,
    paint: np.ndarray,
    paint_name: str,
    tsne_path: Path,
    cmap: str = "viridis",
) -> pd.DataFrame:
    """Plot t-SNE visualizations for each sequence metric and
    save the plots as separate images to the specified directory.

    Parameters
    ----------
    data_proj : np.ndarray
        Transformed embeddings after running t-SNE.
    paint : np.ndarray
        Dataframe containing information of sequence metrics for each DNA sequence.
    paint_name : str
        Name of the sequence metric whose t-SNE visualization will be plotted.
    tsne_path : Path
        Path to save t-SNE plots. Must be a directory.
    cmap : str, optional
        Colormap to visualize, by default "viridis"

    Returns
    -------
    pd.DataFrame
        Dataframe with plotting values.

    Raises
    ------
    ValueError
        If the given tsne_path is not a directory.
    """
    df = pd.DataFrame(
        {
            "z0": data_proj[:, 0],
            "z1": data_proj[:, 1],
            paint_name: paint[: data_proj.shape[0]],
        }
    )
    ax = df.plot.scatter(x="z0", y="z1", c=paint_name, colormap=cmap, alpha=0.4)
    fig = ax.get_figure()
    fig.show()

    # save each tsne plot as a separate png image in the specified directory, tsne_path
    if tsne_path.is_dir():
        fig.savefig(tsne_path / (f"{paint_name}_tsne.png"), dpi=300)
    else:
        raise ValueError(f"{tsne_path} is not a directory!")
    return df


def get_tsne(
    embed_data: np.ndarray, paint_df: pd.DataFrame, tsne_path: Path
) -> Dict[str, pd.DataFrame]:
    """Given 2-dimensional sequence embeddings and sequence metrics dataframe,
    plot and save t-SNE visualizations to specified directory.

    Parameters
    ----------
    embed_data : np.ndarray
        Sequence embeddings to be transformed by t-SNE. Must be 2-dimensional.
    paint_df : pd.DataFrame
        Dataframe containing information of sequence metrics for each DNA sequence.
    tsne_path : Path
        Path to save t-SNE plots. Must be a directory.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dataframes with plotting values.
    """
    data_tsne = run_tsne(embed_data=embed_data)
    return {
        str(key): plot_tsne(
            data_proj=data_tsne,
            paint=paint_df[key].values,
            paint_name=str(key),
            tsne_path=tsne_path,
        )
        for key in paint_df
    }


def run_umap(embed_data: np.ndarray) -> np.ndarray:
    """Given 2-dimensional sequence embeddings, return the transformed data using UMAP.

    Parameters
    ----------
    embed_data : np.ndarray
        Sequence embeddings to be transformed by UMAP. Must be 2-dimensional.

    Returns
    -------
    np.ndarray
        Transformed embeddings after running UMAP.
    """
    from cuml.manifold import UMAP

    # embed_data must be 2D since rapidsai only supports 2D data
    model = UMAP(random_state=10)
    data_proj = model.fit_transform(embed_data)
    return data_proj


def plot_umap(
    data_proj: np.ndarray,
    paint: np.ndarray,
    paint_name: str,
    umap_path: Path,
    cmap: str = "plasma",
) -> pd.DataFrame:
    """Plot UMAP visualizations for each sequence metric and
    save the plots as separate images to the specified directory.

    Parameters
    ----------
    data_proj : np.ndarray
        Transformed embeddings after running UMAP.
    paint : np.ndarray
        Dataframe containing information of sequence metrics for each DNA sequence.
    paint_name : str
        Name of the sequence metric whose UMAP visualization will be plotted.
    umap_path : Path
        Path to save UMAP plots. Must be a directory.
    cmap : str, optional
        Colormap to visualize, by default "plasma"

    Returns
    -------
    pd.DataFrame
        Dataframe with plotting values.

    Raises
    ------
    ValueError
        If the given umap_path is not a directory.
    """
    df = pd.DataFrame(
        {
            "z0": data_proj[:, 0],
            "z1": data_proj[:, 1],
            paint_name: paint[: data_proj.shape[0]],
        }
    )
    ax = df.plot.scatter(x="z0", y="z1", c=paint_name, colormap=cmap, alpha=0.4)
    fig = ax.get_figure()
    fig.show()

    # save each umap plot as a separate png image in the specified directory, umap_path
    if umap_path.is_dir():
        fig.savefig(umap_path / (f"{paint_name}_umap.png"), dpi=300)
    else:
        raise ValueError(f"{umap_path} is not a directory!")
    return df


def get_umap(
    embed_data: np.ndarray, paint_df: pd.DataFrame, umap_path: Path
) -> Dict[str, pd.DataFrame]:
    """Given 2-dimensional sequence embeddings and sequence metrics dataframe,
    plot and save UMAP visualizations to specified directory.

    Parameters
    ----------
    embed_data : np.ndarray
        Sequence embeddings to be transformed by UMAP. Must be 2-dimensional.
    paint_df : pd.DataFrame
        Dataframe containing information of sequence metrics for each DNA sequence.
    umap_path : Path
        Path to save UMAP plots. Must be a directory.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dataframes with plotting values.
    """
    data_umap = run_umap(embed_data=embed_data)
    return {
        str(key): plot_umap(
            data_proj=data_umap,
            paint=paint_df[key].values,
            paint_name=str(key),
            umap_path=umap_path,
        )
        for key in paint_df
    }


def parse_args() -> Namespace:
    """Parse command line arguments.

    Returns
    -------
    Namespace
        Parsed arguments.
    """
    parser = ArgumentParser(description="Generate sequences or embeddings.")
    parser.add_argument(
        "--mode", type=str, required=True, help="Allowed inputs: get_tsne, get_umap"
    )
    parser.add_argument(
        "--embed_path",
        type=Path,
        help="Path to access embeddings. Embeddings could be for training, validation, testing, or generated sequences.",
    )
    parser.add_argument(
        "--fasta_path", type=Path, help="Path to access fasta sequences."
    )
    parser.add_argument(
        "--tsne_path",
        type=Path,
        help="Path to save t-SNE plots. Must lead to a directory, not a file.",
    )
    parser.add_argument(
        "--umap_path",
        type=Path,
        help="Path to save UMAP plots. Must lead to a directory, not a file.",
    )
    return parser.parse_args()


def main() -> None:
    logger.debug("1")
    if args.mode == "get_tsne":
        logger.debug("2")
        if args.tsne_path is None:
            logger.debug("2_")
            raise ValueError("tsne_path is not specified.")
        logger.debug("2__")
        embed_avg = metrics.get_embed_avg(embed_path=args.embed_path)
        logger.debug("3")
        paint_df = get_paint_df(fasta_path=args.fasta_path)
        logger.debug("4")
        get_tsne(embed_data=embed_avg, paint_df=paint_df, tsne_path=args.tsne_path)
        logger.debug("5")
    elif args.mode == "get_umap":
        logger.debug("6")
        if args.umap_path is None:
            logger.debug("6_")
            raise ValueError("umap_path is not specified.")
        logger.debug("6__")
        embed_avg = metrics.get_embed_avg(embed_path=args.embed_path)
        logger.debug("7")
        paint_df = get_paint_df(fasta_path=args.fasta_path)
        logger.debug("8")
        get_umap(embed_data=embed_avg, paint_df=paint_df, umap_path=args.umap_path)
    else:
        logger.debug("9")
        raise ValueError(f"Invalid mode: {args.mode}")
    logger.debug("10")


if __name__ == "__main__":
    args = parse_args()
    main()
