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


def get_paint_df(fasta_path: Path, embed_path: Path) -> pd.DataFrame:
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
    # get DNA sequences
    dna_seqs = metrics.get_seqs_from_fasta(fasta_path=fasta_path)
    # clip DNA sequences to embedding length
    embed = np.load(embed_path)
    dna_seqs = dna_seqs[: len(embed)]
    # translate DNA to protein
    protein_seqs = metrics.dna_to_protein_seqs(dna_seqs=dna_seqs)

    paint_df = pd.DataFrame(
        {
            "GC": metrics.gc_content(dna_seqs),
            "SequenceLength": metrics.seq_length(dna_seqs),
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


# def plot_tsne(
#     data_proj: np.ndarray,
#     paint: np.ndarray,
#     paint_name: str,
#     tsne_path: Path,
#     cmap: str = "viridis",
# ) -> pd.DataFrame:
#     """Plot t-SNE visualizations for each sequence metric and
#     save the plots as separate images to the specified directory.

#     Parameters
#     ----------
#     data_proj : np.ndarray
#         Transformed embeddings after running t-SNE.
#     paint : np.ndarray
#         Dataframe containing information of sequence metrics for each DNA sequence.
#     paint_name : str
#         Name of the sequence metric whose t-SNE visualization will be plotted.
#     tsne_path : Path
#         Path to save t-SNE plots. Must be a directory.
#     cmap : str, optional
#         Colormap to visualize, by default "viridis."

#     Returns
#     -------
#     pd.DataFrame
#         Dataframe with plotting values.

#     Raises
#     ------
#     ValueError
#         If the given tsne_path is not a directory.
#     """
#     df = pd.DataFrame(
#         {
#             "z0": data_proj[:, 0],
#             "z1": data_proj[:, 1],
#             paint_name: paint[: data_proj.shape[0]],
#         }
#     )
#     ax = df.plot.scatter(x="z0", y="z1", c=paint_name, colormap=cmap, alpha=0.4)
#     fig = ax.get_figure()
#     fig.show()

#     # save each tsne plot as a separate png image in the specified directory, tsne_path
#     if tsne_path.is_dir():
#         fig.savefig(tsne_path / (f"{paint_name}_tsne.png"), dpi=300)
#     else:
#         raise ValueError(f"{tsne_path} is not a directory!")
#     return df


# def plot_tsne_subplots(
#     data_proj: np.ndarray,
#     paint_df: pd.DataFrame,
#     tsne_path: Path,
#     cmap: str = "viridis",
# ) -> Dict[str, pd.DataFrame]:
#     """Plot t-SNE visualizations for each sequence metric as subplots and
#     save the plots as a collective image with subplots in the specified directory.

#     Parameters
#     ----------
#     data_proj : np.ndarray
#         Transformed embeddings after running t-SNE.
#     paint_df : pd.DataFrame
#         Dataframe containing information of sequence metrics for each DNA sequence.
#     tsne_path : Path
#         Path to save the t-SNE plot. Must be a directory.
#     cmap : str, optional
#         Colormap to visualize, by default "viridis."

#     Returns
#     -------
#     Dict[str, pd.DataFrame]
#         Dataframes with plotting values.

#     Raises
#     ------
#     ValueError
#         If the given tsne_path is not a directory.
#     """
#     df_dict = {}
#     nrows = 2
#     ncols = 2
#     plt.figure(figsize=(15, 12))

#     for n, key in enumerate(paint_df):
#         paint_name = key
#         paint = paint_df[key].values

#         # create new dataframe for the subplot
#         df = pd.DataFrame(
#             {
#                 "z0": data_proj[:, 0],
#                 "z1": data_proj[:, 1],
#                 paint_name: paint[: data_proj.shape[0]],
#             }
#         )
#         df_dict[key] = df

#         # add subplot
#         ax = plt.subplot(nrows, ncols, n + 1)
#         df.plot.scatter(x="z0", y="z1", ax=ax, c=paint_name, colormap=cmap, alpha=0.4)
#         fig = ax.get_figure()
#         fig.show()
#         plt.tight_layout()

#     # save each tsne plot as a collective image with subplots in the specified directory, tsne_path
#     if tsne_path.is_dir():
#         fig.savefig(tsne_path / (f"SeqMetrics_tsne.png"), dpi=300)
#     else:
#         raise ValueError(f"{tsne_path} is not a directory!")
#     return df_dict


# def get_tsne(
#     embed_data: np.ndarray,
#     paint_df: pd.DataFrame,
#     tsne_path: Path,
#     get_subplots: bool = False,
# ) -> Dict[str, pd.DataFrame]:
#     """Given 2-dimensional sequence embeddings and sequence metrics dataframe,
#     plot and save t-SNE visualizations to specified directory.

#     Parameters
#     ----------
#     embed_data : np.ndarray
#         Sequence embeddings to be transformed by t-SNE. Must be 2-dimensional.
#     paint_df : pd.DataFrame
#         Dataframe containing information of sequence metrics for each DNA sequence.
#     tsne_path : Path
#         Path to save t-SNE plots. Must be a directory.
#     get_subplots : bool, optional
#         True: save t-SNE plots as a collective image with subplots;
#         False: save t-SNE plots as separate images.

#     Returns
#     -------
#     Dict[str, pd.DataFrame]
#         Dataframes with plotting values.
#     """
#     data_tsne = run_tsne(embed_data=embed_data)
#     if get_subplots:
#         return plot_tsne_subplots(
#             data_proj=data_tsne, paint_df=paint_df, tsne_path=tsne_path
#         )
#     else:
#         return {
#             str(key): plot_tsne(
#                 data_proj=data_tsne,
#                 paint=paint_df[key].values,
#                 paint_name=str(key),
#                 tsne_path=tsne_path,
#             )
#             for key in paint_df
#         }


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


# def plot_umap(
#     data_proj: np.ndarray,
#     paint: np.ndarray,
#     paint_name: str,
#     umap_path: Path,
#     cmap: str = "plasma",
# ) -> pd.DataFrame:
#     """Plot UMAP visualizations for each sequence metric and
#     save the plots as separate images to the specified directory.

#     Parameters
#     ----------
#     data_proj : np.ndarray
#         Transformed embeddings after running UMAP.
#     paint : np.ndarray
#         Dataframe containing information of sequence metrics for each DNA sequence.
#     paint_name : str
#         Name of the sequence metric whose UMAP visualization will be plotted.
#     umap_path : Path
#         Path to save UMAP plots. Must be a directory.
#     cmap : str, optional
#         Colormap to visualize, by default "plasma"

#     Returns
#     -------
#     pd.DataFrame
#         Dataframe with plotting values.

#     Raises
#     ------
#     ValueError
#         If the given umap_path is not a directory.
#     """
#     df = pd.DataFrame(
#         {
#             "z0": data_proj[:, 0],
#             "z1": data_proj[:, 1],
#             paint_name: paint[: data_proj.shape[0]],
#         }
#     )
#     ax = df.plot.scatter(x="z0", y="z1", c=paint_name, colormap=cmap, alpha=0.4)
#     fig = ax.get_figure()
#     fig.show()

#     # save each umap plot as a separate png image in the specified directory, umap_path
#     if umap_path.is_dir():
#         fig.savefig(umap_path / (f"{paint_name}_umap.png"), dpi=300)
#     else:
#         raise ValueError(f"{umap_path} is not a directory!")
#     return df


# def plot_umap_subplots(
#     data_proj: np.ndarray, paint_df: pd.DataFrame, umap_path: Path, cmap: str = "plasma"
# ) -> Dict[str, pd.DataFrame]:
#     """Plot UMAP visualizations for each sequence metric as subplots and
#     save the plots as a collective image with subplots in the specified directory.

#     Parameters
#     ----------
#     data_proj : np.ndarray
#         Transformed embeddings after running UMAP.
#     paint_df : pd.DataFrame
#         Dataframe containing information of sequence metrics for each DNA sequence.
#     tsne_path : Path
#         Path to save the UMAP plot. Must be a directory.
#     cmap : str, optional
#         Colormap to visualize, by default "plasma."

#     Returns
#     -------
#     Dict[str, pd.DataFrame]
#         Dataframes with plotting values.

#     Raises
#     ------
#     ValueError
#         If the given umap_path is not a directory.
#     """
#     df_dict = {}
#     nrows = 2
#     ncols = 2
#     plt.figure(figsize=(15, 12))

#     for n, key in enumerate(paint_df):
#         paint_name = key
#         paint = paint_df[key].values
#         df = pd.DataFrame(
#             {
#                 "z0": data_proj[:, 0],
#                 "z1": data_proj[:, 1],
#                 paint_name: paint[: data_proj.shape[0]],
#             }
#         )
#         df_dict[key] = df

#         # add subplot
#         ax = plt.subplot(nrows, ncols, n + 1)
#         df.plot.scatter(x="z0", y="z1", ax=ax, c=paint_name, colormap=cmap, alpha=0.4)
#         fig = ax.get_figure()
#         fig.show()
#         plt.tight_layout()

#     # save each umap plot as a collective image with subplots in the specified directory, umap_path
#     if umap_path.is_dir():
#         fig.savefig(umap_path / (f"SeqMetrics_umap.png"), dpi=300)
#     else:
#         raise ValueError(f"{umap_path} is not a directory!")
#     return df_dict


# def get_umap(
#     embed_data: np.ndarray,
#     paint_df: pd.DataFrame,
#     umap_path: Path,
#     get_subplots: bool = False,
# ) -> Dict[str, pd.DataFrame]:
#     """Given 2-dimensional sequence embeddings and sequence metrics dataframe,
#     plot and save UMAP visualizations to specified directory.

#     Parameters
#     ----------
#     embed_data : np.ndarray
#         Sequence embeddings to be transformed by UMAP. Must be 2-dimensional.
#     paint_df : pd.DataFrame
#         Dataframe containing information of sequence metrics for each DNA sequence.
#     umap_path : Path
#         Path to save UMAP plots. Must be a directory.
#     get_subplots : bool, optional
#         True: save UMAP plots as a collective image with subplots;
#         False: save UMAP plots as separate images.

#     Returns
#     -------
#     Dict[str, pd.DataFrame]
#         Dataframes with plotting values.
#     """
#     data_umap = run_umap(embed_data=embed_data)
#     if get_subplots:
#         return plot_umap_subplots(
#             data_proj=data_umap, paint_df=paint_df, umap_path=umap_path
#         )
#     else:
#         return {
#             str(key): plot_umap(
#                 data_proj=data_umap,
#                 paint=paint_df[key].values,
#                 paint_name=str(key),
#                 umap_path=umap_path,
#             )
#             for key in paint_df
#         }


def plot_cluster(
    data_proj: np.ndarray,
    paint: np.ndarray,
    paint_name: str,
    cluster_path: Path,
    tsne_umap: str = "umap",
    cmap: str = "plasma",
) -> pd.DataFrame:
    """Plot t-SNE or UMAP visualizations for each sequence metric and
    save the plots as separate images to the specified directory.

    Parameters
    ----------
    data_proj : np.ndarray
        Transformed embeddings after running t-SNE or UMAP.
    paint : np.ndarray
        Dataframe containing information of sequence metrics for each DNA sequence.
    paint_name : str
        Name of the sequence metric whose t-SNE or UMAP visualization will be plotted.
    cluster_path : Path
        Path to save plots. Must be a directory.
    cmap : str, optional
        Colormap to visualize, by default "plasma."

    Returns
    -------
    pd.DataFrame
        Dataframe with plotting values.

    Raises
    ------
    ValueError
        If the given cluster_path is not a directory.
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

    # save each plot as a separate png image in the specified directory, cluster_path
    if cluster_path.is_dir():
        fig.savefig(cluster_path / (f"{paint_name}_{tsne_umap}.png"), dpi=300)
    else:
        raise ValueError(f"{cluster_path} is not a directory!")
    return df


def plot_cluster_subplots(
    data_proj: np.ndarray,
    paint_df: pd.DataFrame,
    cluster_path: Path,
    tsne_umap: str = "umap",
    cmap: str = "plasma",
) -> Dict[str, pd.DataFrame]:
    """Plot t-SNE or UMAP visualizations for each sequence metric as subplots and
    save the plots as a collective image with subplots in the specified directory.

    Parameters
    ----------
    data_proj : np.ndarray
        Transformed embeddings after running t-SNE or UMAP.
    paint_df : pd.DataFrame
        Dataframe containing information of sequence metrics for each DNA sequence.
    cluster_path : Path
        Path to save plots. Must be a directory.
    cmap : str, optional
        Colormap to visualize, by default "plasma."

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dataframes with plotting values.

    Raises
    ------
    ValueError
        If the given cluster_path is not a directory.
    """
    df_dict = {}
    nrows = 2
    ncols = 2
    plt.figure(figsize=(15, 12))

    for n, key in enumerate(paint_df):
        paint_name = key
        paint = paint_df[key].values

        # create new dataframe for the subplot
        df = pd.DataFrame(
            {
                "z0": data_proj[:, 0],
                "z1": data_proj[:, 1],
                paint_name: paint[: data_proj.shape[0]],
            }
        )
        df_dict[key] = df

        # add subplot
        ax = plt.subplot(nrows, ncols, n + 1)
        df.plot.scatter(x="z0", y="z1", ax=ax, c=paint_name, colormap=cmap, alpha=0.4)
        fig = ax.get_figure()
        fig.show()
        plt.tight_layout()

    # save each plot as a collective image with subplots in the specified directory, cluster_path
    if cluster_path.is_dir():
        fig.savefig(cluster_path / (f"SeqMetrics_{tsne_umap}.png"), dpi=300)
    else:
        raise ValueError(f"{cluster_path} is not a directory!")
    return df_dict


def get_cluster(
    embed_data: np.ndarray,
    paint_df: pd.DataFrame,
    cluster_path: Path,
    tsne_umap: str = "umap",
    get_subplots: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Given 2-dimensional sequence embeddings and sequence metrics dataframe,
    plot and save t-SNE or UMAP visualizations to specified directory.

    Parameters
    ----------
    embed_data : np.ndarray
        Sequence embeddings to be transformed by t-SNE or UMAP. Must be 2-dimensional.
    paint_df : pd.DataFrame
        Dataframe containing information of sequence metrics for each DNA sequence.
    cluster_path : Path
        Path to save plots. Must be a directory.
    get_subplots : bool, optional
        True: save plots as a collective image with subplots;
        False: save plots as separate images.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dataframes with plotting values.
    """
    if tsne_umap == "tsne":
        data_cluster = run_tsne(embed_data=embed_data)
    else:
        data_cluster = run_umap(embed_data=embed_data)

    if get_subplots:
        return plot_cluster_subplots(
            data_proj=data_cluster,
            paint_df=paint_df,
            cluster_path=cluster_path,
            tsne_umap=tsne_umap,
        )
    else:
        return {
            str(key): plot_cluster(
                data_proj=data_cluster,
                paint=paint_df[key].values,
                paint_name=str(key),
                cluster_path=cluster_path,
                tsne_umap=tsne_umap,
            )
            for key in paint_df
        }


def plot_embed_dist_vs_align_score(
    avg_scores_df: pd.DataFrame,
    save_path: Path,
    alignment_type: str = "global",
    plot_title: str = "",
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
        label="avg embed dist",
    )
    plt.fill_between(
        avg_scores_df[align_key],
        lower_bound,
        upper_bound,
        alpha=0.3,
        label="stdev embed dist",
    )
    plt.ylabel("Embedding L2 Distance", fontsize=14)
    plt.xlabel(align_key, fontsize=14)
    plt.title(plot_title)
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
        help="Allowed inputs: tsne, umap, align_plot",
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
        "--cluster_path",
        type=Path,
        help="Path to save t-SNE or UMAP plots. Must lead to a directory, not a file.",
    )
    # parser.add_argument(
    #     "--tsne_path",
    #     type=Path,
    #     help="Path to save t-SNE plots. Must lead to a directory, not a file.",
    # )
    # parser.add_argument(
    #     "--umap_path",
    #     type=Path,
    #     help="Path to save UMAP plots. Must lead to a directory, not a file.",
    # )
    parser.add_argument(
        "--get_subplots",
        type=bool,
        help="True: save t-SNE or UMAP plots as a collective image with subplots; False: save t-SNE or UMAP plots as separate images.",
    )
    parser.add_argument(
        "--align_plot_path",
        type=Path,
        help="Path to save the Alignment Score vs. Embedding Distance plot. Must be a directory.",
    )
    parser.add_argument(
        "--align_plot_title",
        default="",
        type=str,
        help="Title for embed dist vs. align score plot.",
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
    if (args.mode == "tsne") or (args.mode == "umap"):
        if args.cluster_path is None:
            raise ValueError("cluster_path is not specified.")
        embed_avg = metrics.get_embed_avg(embed_path=args.embed_path)
        paint_df = get_paint_df(fasta_path=args.fasta_path, embed_path=args.embed_path)
        get_cluster(
            embed_data=embed_avg,
            paint_df=paint_df,
            cluster_path=args.cluster_path,
            tsne_umap=args.mode,
            get_subplots=args.get_subplots,
        )
        print(f"Cluster plots have been saved to {args.cluster_path}.")

    # if args.mode == "get_tsne":
    #     if args.tsne_path is None:
    #         raise ValueError("tsne_path is not specified.")
    #     embed_avg = metrics.get_embed_avg(embed_path=args.embed_path)
    #     paint_df = get_paint_df(fasta_path=args.fasta_path, embed_path=args.embed_path)
    #     get_tsne(
    #         embed_data=embed_avg,
    #         paint_df=paint_df,
    #         tsne_path=args.tsne_path,
    #         get_subplots=args.get_subplots,
    #     )
    #     print(f"t-SNE plots have been saved to {args.tsne_path}.")
    # elif args.mode == "get_umap":
    #     if args.umap_path is None:
    #         raise ValueError("umap_path is not specified.")
    #     embed_avg = metrics.get_embed_avg(embed_path=args.embed_path)
    #     paint_df = get_paint_df(fasta_path=args.fasta_path, embed_path=args.embed_path)
    #     get_umap(
    #         embed_data=embed_avg,
    #         paint_df=paint_df,
    #         umap_path=args.umap_path,
    #         get_subplots=args.get_subplots,
    #     )
    #     print(f"UMAP plots have been saved to {args.umap_path}.")

    elif args.mode == "align_plot":
        if args.align_plot_path is None:
            raise ValueError("align_plot_path is not specified.")
        embed_avg = metrics.get_embed_avg(embed_path=args.embed_path)
        dna_seqs = metrics.get_seqs_from_fasta(fasta_path=args.fasta_path)
        embed = np.load(args.embed_path)
        dna_seqs = dna_seqs[: len(embed)]  # clip DNA sequence to embedding length
        protein_seqs = metrics.dna_to_protein_seqs(dna_seqs=dna_seqs)
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
        scores_df = metrics.get_scores_df(
            embed_avg=embed_avg,
            scores_matrix=protein_align_scores_matrix,
            alignment_type=args.alignment_type,
        )
        avg_scores_df = metrics.get_avg_scores_df(
            scores_df=scores_df, alignment_type=args.alignment_type
        )
        plot_embed_dist_vs_align_score(
            avg_scores_df=avg_scores_df,
            save_path=args.align_plot_path,
            alignment_type=args.alignment_type,
            plot_title=args.align_plot_title,
        )
        print(
            f"AlignScore vs. EmbedDist Plot has been saved to {args.align_plot_path}."
        )
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    args = parse_args()
    main()
