"""Visualize synthetic sequences using t-SNE, UMAP, and other visualization schemes."""
import logging
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    dna_seqs = metrics.get_seqs_from_fasta(fasta_path)
    # clip DNA sequences to embedding length
    embed = np.load(embed_path)
    dna_seqs = dna_seqs[: len(embed)]
    # translate DNA to protein
    protein_seqs = metrics.dna_to_protein_seqs(dna_seqs)

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


def run_umap(
    embed_data: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    random_state: int = 10,
) -> np.ndarray:
    """Given 2-dimensional sequence embeddings, return the transformed data using UMAP.

    Parameters
    ----------
    embed_data : np.ndarray
        Sequence embeddings to be transformed by UMAP. Must be 2-dimensional.
    n_neighbors : int, optional
        The size of local neighborhood (in terms of number of neighboring sample points)
        used for manifold approximation. Larger values result in more global views of the
        manifold, while smaller values result in more local data being preserved.
        In general values should be in the range 2 to 100.
    min_dist : float, optional
        The effective minimum distance between embedded points. Smaller values will result
        in a more clustered/clumped embedding where nearby points on the manifold are drawn
        closer together, while larger values will result on a more even dispersal of points.
        The value should be set relative to the spread value, which determines the scale at
        which embedded points will be spread out.
    spread : float, optional
        The effective scale of embedded points. In combination with min_dist this determines
        how clustered/clumped the embedded points are.
    random_state : int, optional
        The seed used by the random number generator during embedding initialization and
        during sampling used by the optimizer.

    Returns
    -------
    np.ndarray
        Transformed embeddings after running UMAP.
    """
    from cuml.manifold import UMAP

    # embed_data must be 2D since rapidsai only supports 2D data
    model = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        random_state=random_state,
    )
    data_proj = model.fit_transform(embed_data)
    return data_proj


def plot_cluster(
    data_proj: np.ndarray,
    paint: np.ndarray,
    paint_name: str,
    save_path: Optional[Path] = None,
    cmap: str = "plasma",
) -> pd.DataFrame:
    """Plot a scatter plot and, if save_path is given, save the plots.

    Parameters
    ----------
    data_proj : np.ndarray
        Coordinates of the scatter points to be plotted.
    paint : np.ndarray
        Values of each scatter point.
    paint_name : str
        Name of the scatter plot.
    save_path : Optional[Path], optional
        Path to save plots, by default None. If given, should have the format
        "directory/file_name_to_be_created.png".
    cmap : str, optional
        Colormap to visualize, by default "plasma."

    Returns
    -------
    pd.DataFrame
        Dataframe with plotting coordinates and plotting values.
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

    if save_path is not None:
        # get the file name and the suffix of save_path separately
        split_tup = os.path.splitext(save_path)
        file_name = split_tup[0]
        suffix = split_tup[1]
        # add paint_name to the file name
        save_path = str(file_name) + "-" + str(paint_name) + str(suffix)

        fig.savefig(save_path, dpi=300)
        print(f"Your plot has been saved to {save_path}")
    return df


def plot_cluster_subplots(
    data_proj: np.ndarray,
    paint_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    cmap: str = "plasma",
) -> Dict[str, pd.DataFrame]:
    """Plot scatter plots as subplots and, if save_path is given, save the plot.
    Parameters
    ----------
    data_proj : np.ndarray
        Coordinates of the scatter points to be plotted. Coordinates are the same for
        each of the subplots.
    paint_df : pd.DataFrame
        Dataframe containing values of each scatter point for each of the subplots.
    save_path : Optional[Path], optional
        Path to save plots, by default None. If given, should have the format
        "directory/file_name_to_be_created.png".
    cmap : str, optional
        Colormap to visualize, by default "plasma."

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dataframes with plotting coordinates and plotting values.
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

    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        print(f"Your plot has been saved to {save_path}")
    return df_dict


def get_cluster(
    embed_data: np.ndarray,
    paint_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    tsne_umap: str = "umap",
    get_subplots: bool = False,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_spread: float = 1.0,
    umap_random_state: int = 10,
) -> Dict[str, pd.DataFrame]:
    """Given 2-dimensional sequence embeddings and sequence metrics dataframe,
    plot and save t-SNE or UMAP visualizations to specified directory.

    Parameters
    ----------
    embed_data : np.ndarray
        Sequence embeddings to be transformed by t-SNE or UMAP. Must be 2-dimensional.
    paint_df : pd.DataFrame
        Dataframe containing information of sequence metrics for each DNA sequence.
    save_path : Optional[Path], optional
        Path to save plots, by default None. If given, should have the format
        "directory/file_name_to_be_created.png".
    tsne_umap : str, optional
        "tsne" or "umap" to specify the type of cluster plot, by default "umap."
    get_subplots : bool, optional
        True: save plots as a collective image with subplots;
        False: save plots as separate images.
    n_neighbors : int, optional
        The size of local neighborhood (in terms of number of neighboring sample points)
        used for manifold approximation. Larger values result in more global views of the
        manifold, while smaller values result in more local data being preserved.
        In general values should be in the range 2 to 100.
    min_dist : float, optional
        The effective minimum distance between embedded points. Smaller values will result
        in a more clustered/clumped embedding where nearby points on the manifold are drawn
        closer together, while larger values will result on a more even dispersal of points.
        The value should be set relative to the spread value, which determines the scale at
        which embedded points will be spread out.
    spread : float, optional
        The effective scale of embedded points. In combination with min_dist this determines
        how clustered/clumped the embedded points are.
    random_state : int, optional
        The seed used by the random number generator during embedding initialization and
        during sampling used by the optimizer.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dataframes with plotting values.
    """
    if tsne_umap == "tsne":
        data_proj = run_tsne(embed_data=embed_data)
    else:
        data_proj = run_umap(
            embed_data=embed_data,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            spread=umap_spread,
            random_state=umap_random_state,
        )

    if get_subplots:
        return plot_cluster_subplots(
            data_proj=data_proj,
            paint_df=paint_df,
            save_path=save_path,
        )

    return {
        str(key): plot_cluster(
            data_proj=data_proj,
            paint=paint_df[key].values,
            paint_name=str(key),
            save_path=save_path,
        )
        for key in paint_df
    }


def plot_metrics_hist(
    paint_dfs: List[pd.DataFrame], labels: List[str], save_path: Optional[Path] = None
) -> None:
    """Plot the sequence metrics histograms across generated, test, validation, and/or
    training sequences, with each subplot representing a metric, and save the plot to the
    specified directory.

    Parameters
    ----------
    paint_dfs : List[pd.DataFrame]
        List of painted dataframes containing the paint_df for each type of sequences.
    labels : List[str]
        List of labels containing the names of the type of sequences, in the order that these
        sequence types are arranged in paint_dfs.
    save_path : Optional[Path], optional
        Path to save the metrics histograms, by default None. If given, should have the format
        "directory/file_name_to_be_created.png".
    """
    ncols = 2
    nrows = int(np.ceil(len(paint_dfs[0].columns) / 2))
    plt.figure(figsize=(15, 12))  # width, height

    for n, key in enumerate(paint_dfs[0]):
        plt.subplot(nrows, ncols, n + 1)
        for i in range(len(paint_dfs)):
            plt.hist(paint_dfs[i][key], bins=None, alpha=0.5, label=labels[i], log=True)
        plt.legend(loc="upper left")
        plt.title(key)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Metrics histogram plot has been saved to {save_path}.")


def plot_embed_dist_vs_align_score(
    avg_scores_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    alignment_type: str = "global",
    plot_title: str = "",
) -> None:
    """Plot the Pairwise Alignment Score (Global or Local) vs. Embedding L2 Distance,
    and save the plot to the specified directory.

    Parameters
    ----------
    avg_scores_df : pd.DataFrame
        Three-column dataframe comparing the average L2 distance,
    standard deviation of the L2 distance, and the pairwise alignment scores.
    save_path : Optional[Path], optional
        Path to save the embedding L2 distance vs. alignment score plot, by default None.
        If given, should have the format "directory/file_name_to_be_created.png".
    alignment_type : str, optional
        "global" or "local", by default "global."
    plot_title : str, optional
        Title of the plot, by default "".

    Raises
    ------
    ValueError
        If alignment_type is neither "global" nor "local."
    """
    align_key = metrics._get_alignment_name(alignment_type)

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

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(
            f"Embedding distance vs. alignment score plot has been saved to {save_path}."
        )


def plot_align_hist_mean_max_min(
    scores_matrix: np.ndarray, save_path: Optional[Path] = None, plot_title: str = ""
) -> Dict[str, np.ndarray]:
    """Plot a histogram showing the distributions of mean, max, and min alignment scores
    between the alignment of two collections of sequences.

    Parameters
    ----------
    scores_matrix : np.ndarray
        Alignment scores matrix aligning two collections of sequences seqs1 and seqs2.
        Matrix should have the dimension M * N, where M is the length of seqs1, and N
        is the length of seqs2.
    save_path : Optional[Path], optional
        Directory to save the plot, by default Path(""). If given, should have the format
        "directory/file_name_to_be_created.png".
    plot_title : str, optional
        Title of the plot, by default "".

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of mean, max, and min alignment scores
    """
    # compute the mean, max, min alignment score values
    mean_scores = metrics.get_mean_align_scores(scores_matrix)
    max_scores = metrics.get_max_align_scores(scores_matrix)
    min_scores = metrics.get_min_align_scores(scores_matrix)

    plt.hist(mean_scores, bins=None, alpha=0.5, label="mean")
    plt.hist(max_scores, bins=None, alpha=0.5, label="max")
    plt.hist(min_scores, bins=None, alpha=0.5, label="min")
    plt.xlabel("Alignment Score")
    plt.ylabel("Counts")
    plt.title(plot_title)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Histogram align mean-max-min plot has been saved to {save_path}.")

    # save the mean/max/min scores and return them in a dictionary
    scores_dict = {"mean": mean_scores, "max": max_scores, "min": min_scores}
    return scores_dict


def parse_args() -> Namespace:
    """Parse command line arguments.

    Returns
    -------
    Namespace
        Parsed arguments.
    """
    parser = ArgumentParser(description="Visualize synthetic sequences.")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Allowed inputs: tsne, umap, align_plot, align_hist_mean_max_min",
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
        "--embed_path2",
        type=Path,
        help="Path to access embeddings for a second set of sequences. Embeddings could be for training, validation, testing, or generated sequences.",
    )
    parser.add_argument(
        "--fasta_path2",
        type=Path,
        help="Path to access a second set of fasta sequences.",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        required=True,
        help='Path to save plots. Should have the format "directory/file_name_to_be_created.png".',
    )
    parser.add_argument(
        "--get_subplots",
        type=bool,
        help="True: save t-SNE or UMAP plots as a collective image with subplots; False: save t-SNE or UMAP plots as separate images.",
    )
    parser.add_argument(
        "--umap_n_neighbors",
        default=15,
        type=int,
        help="Size of local neighborhood (in terms of the number of neighboring sample points) to run UMAP.",
    )
    parser.add_argument(
        "--umap_min_dist",
        default=0.1,
        type=float,
        help="Effective minimum distance between embedded points to run UMAP.",
    )
    parser.add_argument(
        "--umap_spread",
        default=1.0,
        type=float,
        help="Effective scale of embedded points to run UMAP.",
    )
    parser.add_argument(
        "--umap_random_state",
        default=10,
        type=int,
        help="Seed used by the random number generator during embedding initialization and during sampling used by the optimizer to run UMAP.",
    )
    parser.add_argument(
        "--plot_title",
        default="",
        type=str,
        help="Title for a plot.",
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
    """Visualize synthetic sequences.

    Parameters
    ----------
    args : Namespace
        Parsed arguments.

    Raises
    ------
    ValueError
        If mode is invalid.
    """
    if (args.mode == "tsne") or (args.mode == "umap"):
        embed_avg = metrics.get_embed_avg(args.embed_path)
        paint_df = get_paint_df(args.fasta_path, args.embed_path)
        get_cluster(
            embed_data=embed_avg,
            paint_df=paint_df,
            save_path=args.save_path,
            tsne_umap=args.mode,
            get_subplots=args.get_subplots,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
            umap_spread=args.umap_spread,
            umap_random_state=args.umap_random_state,
        )
    elif args.mode == "align_plot":
        """
        required argparse arguments to pass through:
        --save_path
        --mode
        --embed_path
        --fasta_path
        --alignment_type (default="global")
        --plot_title (default="")
        --num_workers (default=1)
        --match_score (default=1.0)
        --mismatch_score (default=0.0)
        --open_gap_score (default=0.0)
        --extend_gap_score (default=0.0)
        """
        embed_avg = metrics.get_embed_avg(args.embed_path)
        dna_seqs = metrics.get_seqs_from_fasta(args.fasta_path)
        embed = np.load(args.embed_path)
        dna_seqs = dna_seqs[: len(embed)]  # clip DNA sequence to embedding length
        protein_seqs = metrics.dna_to_protein_seqs(dna_seqs)
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
            scores_df=scores_df,
            alignment_type=args.alignment_type,
        )
        plot_embed_dist_vs_align_score(
            avg_scores_df=avg_scores_df,
            save_path=args.save_path,
            alignment_type=args.alignment_type,
            plot_title=args.plot_title,
        )
    elif args.mode == "align_hist_mean_max_min":
        """
        required argparse arguments to pass through:
        --save_path
        --mode
        --embed_path
        --fasta_path
        --embed_path2
        --fasta_path2
        --alignment_type (default="global")
        --plot_title (default="")
        --num_workers (default=1)
        --match_score (default=1.0)
        --mismatch_score (default=0.0)
        --open_gap_score (default=0.0)
        --extend_gap_score (default=0.0)
        """
        # get the protein sequences for the first set of nucleotide sequences
        dna_seqs1 = metrics.get_seqs_from_fasta(args.fasta_path)
        embed1 = np.load(args.embed_path)
        dna_seqs1 = dna_seqs1[: len(embed1)]  # clip DNA sequence to embedding length
        protein_seqs1 = metrics.dna_to_protein_seqs(dna_seqs1)

        # get the protein sequences for the second set of nucleotide sequences
        dna_seqs2 = metrics.get_seqs_from_fasta(args.fasta_path2)
        embed2 = np.load(args.embed_path2)
        dna_seqs2 = dna_seqs2[: len(embed2)]  # clip DNA sequence to embedding length
        protein_seqs2 = metrics.dna_to_protein_seqs(dna_seqs2)

        # compute pairwise alignment scores matrix
        proteins12_align_scores_matrix = metrics.alignment_scores_parallel(
            seqs1_rec=protein_seqs1,
            seqs2_rec=protein_seqs2,
            alignment_type=args.alignment_type,
            num_workers=args.num_workers,
            match_score=args.match_score,
            mismatch_score=args.mismatch_score,
            open_gap_score=args.open_gap_score,
            extend_gap_score=args.extend_gap_score,
        )

        # plot the histogram distributions of mean, max, and min alignment scores
        plot_align_hist_mean_max_min(
            proteins12_align_scores_matrix,
            save_path=args.save_path,
            plot_title=args.plot_title,
        )
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    args = parse_args()
    main()
