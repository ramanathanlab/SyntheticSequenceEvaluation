"""Visualize synthetic sequences using t-SNE, UMAP, and other visualization schemes."""
import logging
from argparse import ArgumentParser, Namespace
from ctypes import alignment
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO

from biosynseq import metrics

logger = logging.getLogger("biosynseq.visualize")


def get_paint_df(fasta_path: Path) -> pd.DataFrame:
    """Collect a set of scalar arrays to paint with.

    Given a path to a fasta file, return a dataframe with information
    of the GC content and sequence length of each DNA sequence, as well
    as the molecular weight and isoelectric point of the protein translated
    from each DNA sequence.

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
    # seqs = list(SeqIO.parse(fasta_path, "fasta"))
    # # translate DNA seqs to protein seqs; stop translation at the first in-frame stop codon
    # protein_seqs = [s.translate(to_stop=True) for s in seqs]

    seqs = metrics.get_seqs_from_fasta(fasta_path=fasta_path)
    protein_seqs = metrics.get_seqs_from_fasta(
        fasta_path=fasta_path, translate_to_protein=True
    )
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


def plot_AlignScore_EmbedDist(
    avg_scores_df: pd.DataFrame, save_path: Path, alignment_type: str = "global"
) -> str:
    """Plot the Pairwise Alignment Score (Global or Local) vs. Embedding L2 Distance,
    and save the plot to the specified directory.

    Parameters
    ----------
    avg_scores_df : pd.DataFrame
        Three-column dataframe comparing the average L2 distance,
    standard deviation of the L2 distance, and the pairwise alignment scores.
    save_path : Path
        Path to save the Pairwise Alignment Score vs. Embedding L2 Distance plot. Must be a directory.
    alignment_type : str, optional
        "global" or "local", by default "global."

    Returns
    -------
    str
        Statement indicating that plot saving has been complete.

    Raises
    ------
    ValueError
        If alignment_type is neither "global" nor "local."
    ValueError
        If the path to save the plot is not a directory.
    """
    if alignment_type == "global":
        align_key = "Global Alignment Score"
    elif alignment_type == "local":
        align_key = "Local Alignment Score"
    else:
        raise ValueError(f"Invalid alignment type: {alignment_type}")

    lower_bound = avg_scores_df["avg_embed_dist"] - avg_scores_df["stdev_embed_dist"]
    upper_bound = avg_scores_df["avg_embed_dist"] + avg_scores_df["stdev_embed_dist"]

    plt.plot(
        avg_scores_df[align_key],
        avg_scores_df["avg_embed_dist"],
        linewidth=3,
        label="average embedding distance",
    )
    plt.fill_between(
        avg_scores_df[align_key],
        lower_bound,
        upper_bound,
        alpha=0.3,
        label="stdev embedding distance",
    )
    plt.ylabel("L2 Embedding Distance", fontsize=14)
    plt.xlabel(align_key, fontsize=14)
    plt.title(align_key + " vs. L2 Embedding Distance")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()

    if save_path.is_dir():
        plt.savefig(save_path / ("AlignScore_EmbedDist.png"), dpi=300)
    else:
        raise ValueError(f"{save_path} is not a directory!")

    return f"Alignment Score vs. Embedding Distance plot has been saved to {save_path}."


def parse_args() -> Namespace:
    """Parse command line arguments.

    Returns
    -------
    Namespace
        Parsed arguments.
    """
    parser = ArgumentParser(description="Generate sequences or embeddings.")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Allowed inputs: get_tsne, get_umap, get_align_plot",
    )
    parser.add_argument(
        "--embed_path",
        type=Path,
        required=True,
        help="Path to access embeddings. Embeddings could be for training, validation, testing, or generated sequences.",
    )
    parser.add_argument(
        "--fasta_path", type=Path, required=True, help="Path to access fasta sequences."
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
    parser.add_argument(
        "--align_plot_path",
        type=Path,
        help="Path to save the Alignment Score vs. Embedding Distance plot. Must be a directory.",
    )
    parser.add_argument(
        "--alignment_type", default="global", type=str, help="global or local"
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Number of concurrent processes of execution.",
    )
    parser.add_argument(
        "--match_score",
        default=1.0,
        type=float,
        help="Match score to calculate global or local alignment scores using Align.PairwiseAligner.",
    )
    parser.add_argument(
        "--mismatch_score",
        default=0.0,
        type=float,
        help="Mismatch score to calculate to calculate global or local alignment scores using Align.PairwiseAligner.",
    )
    parser.add_argument(
        "--open_gap_score",
        default=0.0,
        type=float,
        help="Open gap score to calculate to calculate global or local alignment scores using Align.PairwiseAligner.",
    )
    parser.add_argument(
        "--extend_gap_score",
        default=0.0,
        type=float,
        help="Extend gap score to calculate to calculate global or local alignment scores using Align.PairwiseAligner.",
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
    elif args.mode == "get_align_plot":
        logger.debug("9")
        if args.align_plot_path is None:
            raise ValueError("align_plot_path is not specified.")
        logger.debug("10")
        embed_avg = metrics.get_embed_avg(embed_path=args.embed_path)
        logger.debug("11")
        protein_seqs = metrics.get_seqs_from_fasta(
            fasta_path=args.fasta_path, translate_to_protein=True
        )
        logger.debug("12")
        protein_align_scores_matrix = metrics.alignment_scores_parallel(
            seqs1_rec=protein_seqs,
            seqs2_rec=protein_seqs,
            alignment_type=args.alignment_type,
            num_workers=args.num_workers,
            match_score=args.match_score,
            mismatch_score=args.mismatch_score,
            open_gap_score=args.open_gap_score,
            extend_gap_score=args.extend_gap_score,
        )
        logger.debug("13")
        scores_df = metrics.get_scores_df(
            embed_avg=embed_avg,
            scores_matrix=protein_align_scores_matrix,
            alignment_type=args.alignment_type,
        )
        logger.debug("14")
        avg_scores_df = metrics.get_avg_scores_df(
            scores_df=scores_df, alignment_type=args.alignment_type
        )
        logger.debug("15")
        plot_AlignScore_EmbedDist(
            avg_scores_df=avg_scores_df,
            save_path=args.align_plot_path,
            alignment_type=args.alignment_type,
        )
    else:
        logger.debug("16")
        raise ValueError(f"Invalid mode: {args.mode}")
    logger.debug("17")


if __name__ == "__main__":
    args = parse_args()
    main()
