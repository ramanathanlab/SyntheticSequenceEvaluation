"""Sequence Metrics."""
import functools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from Bio import Align, SeqIO, SeqUtils
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint
from scipy.spatial import distance_matrix
from tqdm import tqdm


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


def get_seqs_from_fasta(
    fasta_path: Path, translate_to_protein: bool = False
) -> List[SeqRecord]:
    """Given a fasta file, obtain its DNA sequences or translated protein sequences.

    Parameters
    ----------
    fasta_path : Path
        Path to access the fasta sequences.
    translate_to_protein : bool, optional
        Whether to translate the DNA sequences to protein sequneces, by default False.

    Returns
    -------
    List[SeqRecord]
        DNA or protein sequences from the fasta file.
    """
    seqs = list(SeqIO.parse(fasta_path, "fasta"))
    if translate_to_protein:
        seqs = dna_to_protein_seqs(dna_seqs=seqs)
    return seqs


def dna_to_protein_seqs(dna_seqs: List[SeqRecord]) -> List[SeqRecord]:
    """Translate DNA sequences to protein sequences.
    Stop translation at the first in-frame stop codon

    Parameters
    ----------
    dna_seqs : List[SeqRecord]
        List of DNA sequences.

    Returns
    -------
    List[SeqRecord]
        List of protein sequences.
    """
    return [seq.translate(to_stop=True) for seq in dna_seqs]


def gc_content(seqs: List[SeqRecord]) -> List[float]:
    """Given a list of DNA sequences, return each sequence's GC content.

    Parameters
    ----------
    seqs : List[SeqRecord.SeqRecord[str]]
        A list of DNA sequences.

    Returns
    -------
    List
        GC content of each DNA sequence.
    """
    return [SeqUtils.GC(rec.seq) for rec in seqs]


def seq_length(seqs: List[SeqRecord]) -> List[float]:
    """Given a list of DNA sequences, return each sequence's length.

    Parameters
    ----------
    seqs : List[SeqRecord.SeqRecord[str]]
        A list of DNA sequences.

    Returns
    -------
    List
        Length of each DNA sequence.
    """
    return [len(rec.seq) for rec in seqs]


def molecular_weight(protein_seqs: List[SeqRecord]) -> List[float]:
    """Given a list of protein sequences, return each protein's molecular weight.

    Parameters
    ----------
    protein_seqs : List[SeqRecord.SeqRecord[str]]
        A list of protein sequences.

    Returns
    -------
    List
        Molecular weight of each protein.
    """
    return [SeqUtils.molecular_weight(rec.seq, "protein") for rec in protein_seqs]


def isoelectric_point(protein_seqs: List[SeqRecord]) -> List[float]:
    """Given a list of protein sequences, return each protein's isoelectric point.

    Parameters
    ----------
    protein_seqs : List[SeqRecord.SeqRecord[str]]
        A list of protein sequences.

    Returns
    -------
    List
        Isoelectric point of each protein.
    """
    return [IsoelectricPoint(seq).pi() for seq in protein_seqs]


def compute_alignment_scores(
    target_seq: Seq,
    query_seqs: List[SeqRecord],
    alignment_type: str = "global",
    match_score: float = 1.0,
    mismatch_score: float = 0.0,
    open_gap_score: float = 0.0,
    extend_gap_score: float = 0.0,
) -> np.ndarray:
    """Compute global or local pairwise alignment scores between a target sequence and an array of query sequences.

    Compute global or local pairwise alignment scores (DNA, RNA, or protein) between a target sequence
    and an array of query sequences, given score calculation configurations. Return an array
    of scores, each score being the pairwise alignment score between the target sequence
    and each of the query sequences.

    Parameters
    ----------
    target_seq : Seq
        Sequence to align against.
    query_seqs : List[SeqRecord]
        Sequences to align.
    alignment_type : str, optional
        "global" or "local", by default "global."
    match_score : float, optional
        Score for each matched alignment, by default 1.0.
    mismatch_score : float, optional
        Score for each mismatched alignment, by default 0.0.
    open_gap_score : float, optional
        Score for each gap opening, by default 0.0.
    extend_gap_score : float, optional
        Score for each gap extension, by default 0.0.

    Returns
    -------
    np.ndarray
        An array of pairwise alignment scores between the target sequence and an array of query sequences.
    """
    aligner = Align.PairwiseAligner(
        mode=alignment_type,
        match_score=match_score,
        mismatch_score=mismatch_score,
        open_gap_score=open_gap_score,
        extend_gap_score=extend_gap_score,
    )
    scores = np.array([aligner.align(target_seq, seq.seq).score for seq in query_seqs])

    return scores


def alignment_scores_parallel(
    seqs1_rec: List[SeqRecord],
    seqs2_rec: List[SeqRecord],
    alignment_type: str = "global",
    match_score: float = 1.0,
    mismatch_score: float = 0.0,
    open_gap_score: float = 0.0,
    extend_gap_score: float = 0.0,
    num_workers: int = 1,
) -> np.ndarray:
    """Compute pairwise alignment scores between all sequences in seqs1_rec and seqs2_rec.
    Sequences can be for DNA, RNA, or protein.

    Parameters
    ----------
    seqs1_rec : List[SeqRecord]
        First collection of sequences.
    seqs2_rec : List[SeqRecord]
        Second collection of sequences.
    alignment_type : str, optional
        "global" or "local", by default "global."
    match_score : float, optional
        Score for each matched alignment, by default 1.0.
    mismatch_score : float, optional
        Score for each mismatched alignment, by default 0.0.
    open_gap_score : float, optional
        Score for each gap opening, by default 0.0.
    extend_gap_score : float, optional
        Score for each gap extension, by default 0.0.
    num_workers : int, optional
        Number of concurrent processes of execution, by default 1.

    Returns
    -------
    np.ndarray
        A scores matrix containing pairwise alignment scores between all sequences in seqs1_rec and seqs2_rec.

    Raises
    ------
    ValueError
        If alignment_type is neither "global" nor "local."
    """
    if alignment_type not in ("global", "local"):
        raise ValueError(f"Invalid alignment_type: {alignment_type}")

    # save sequences as Seq objects rather than SeqRecord objects, since
    # PairwiseAligner must work with Seq objects, not SeqRecord objects
    target_seqs = (rec.seq for rec in seqs1_rec)

    alignment_fn = functools.partial(
        compute_alignment_scores,
        query_seqs=seqs2_rec,
        alignment_type=alignment_type,
        match_score=match_score,
        mismatch_score=mismatch_score,
        open_gap_score=open_gap_score,
        extend_gap_score=extend_gap_score,
    )

    chunksize = max(1, len(seqs1_rec) // num_workers)
    scores_matrix = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for scores in tqdm(
            executor.map(alignment_fn, target_seqs, chunksize=chunksize)
        ):
            scores_matrix.append(scores)

    return np.array(scores_matrix)


def triu_flatten(
    a: np.ndarray, subtract_diag: bool = False, only_positive: bool = False
) -> np.ndarray:
    """Get upper triangular matrix and flatten.

    Parameters
    ----------
    a : np.ndarray
        Matrix of shape (N, N).
    subtract_diag : bool, default=False
        Subtract off the diagonal before flattening.
    only_positive : bool, default=False
        Only returns positive entries.
    """
    out = np.triu(a - np.diag(np.diag(a))) if subtract_diag else np.triu(a)
    out = out.flatten()

    if only_positive:
        out = out[out > 0]

    return out


def get_embed_dist_flatten(embed_avg: np.ndarray) -> np.ndarray:
    """Given the average sequence embeddings for a number of sequences,
    get the flattened distance matrix between all embeddings.

    Parameters
    ----------
    embed_avg : np.ndarray
        Average sequence embeddings for a number of sequences.

    Returns
    -------
    np.ndarray
        Flattened distance matrix between all embeddings.
    """
    # get embedding l2 distance matrix
    embed_dist = distance_matrix(embed_avg, embed_avg)
    # get upper triangular matrix and flatten
    return triu_flatten(embed_dist, only_positive=True)


def _get_alignment_name(alignment_type: str) -> str:
    """Convert algorithm name to clean format.

    Parameters
    ----------
    alignment_type : str
        Algorithm type "local" or "global".

    Returns
    -------
    str
        Output name with clean format.

    Raises
    ------
    ValueError
        If alignment_type is neither "global" nor "local".
    """
    if alignment_type == "global":
        return "Global Alignment Score"
    if alignment_type == "local":
        return "Local Alignment Score"
    raise ValueError(f"Invalid alignment type: {alignment_type}")


def get_scores_df(
    embed_avg: np.ndarray, scores_matrix: np.ndarray, alignment_type: str = "global"
) -> pd.DataFrame:
    """Compute a two-column scores dataframe comparing the embedding
    L2 distance and the pairwise alignment scores, given the average
    embeddings for the sequences, the scores matrix, and the alignment type.

    Parameters
    ----------
    embed_avg : np.ndarray
        Average sequence embeddings for a number of sequences.
    scores_matrix : np.ndarray
        Scores matrix containing pairwise alignment scores.
    alignment_type : str, optional
        "global" or "local", by default "global."

    Returns
    -------
    pd.DataFrame
        Two-column dataframe comparing the embedding L2 distance and the
        pairwise alignment scores.

    Raises
    ------
    ValueError
        If alignment_type is neither "global" nor "local".
    """
    align_key = _get_alignment_name(alignment_type)
    embed_dist_upper = get_embed_dist_flatten(embed_avg)
    scores_upper = triu_flatten(scores_matrix, subtract_diag=True, only_positive=True)
    scores_df = pd.DataFrame(
        {"Embedding L2 Distance": embed_dist_upper, align_key: scores_upper}
    )
    return scores_df


def get_avg_scores_df(
    scores_df: pd.DataFrame, alignment_type: str = "global"
) -> pd.DataFrame:
    """Compute a three-column dataframe comparing the average L2 distance,
    standard deviation of the L2 distance, and the pairwise alignment scores.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Two-column dataframe with "Embedding L2 Distance" and either
        "Global Alignment Score" or "Local Alignment Score" as its columns.
    alignment_type : str, optional
        "global" or "local", by default "global."

    Returns
    -------
    pd.DataFrame
        Three-column dataframe comparing the average L2 distance,
    standard deviation of the L2 distance, and the pairwise alignment scores.

    Raises
    ------
    ValueError
        If alignment_type is neither "global" nor "local".
    """

    align_key = _get_alignment_name(alignment_type)
    unique_scores = scores_df[align_key].unique()

    # calculate embedding distance average
    avg_embed_dist = [
        np.mean(scores_df["Embedding L2 Distance"][scores_df[align_key] == score])
        for score in sorted(unique_scores)
    ]

    # calculate embedding distance standard deviation
    stdev_embed_dist = [
        np.std(scores_df["Embedding L2 Distance"][scores_df[align_key] == score])
        for score in sorted(unique_scores)
    ]

    # construct dataframe
    avg_scores_df = pd.DataFrame(
        {
            "avg_embed_dist": avg_embed_dist,
            "stdev_embed_dist": stdev_embed_dist,
            align_key: sorted(unique_scores),
        }
    )
    return avg_scores_df


def get_mean_align_scores(scores_matrix: np.ndarray) -> np.ndarray:
    """Given an alignment scores matrix aligning two collections of sequences
    seqs1 and seqs2, return the mean alignment scores for the alignments
    between each sequence in seqs1 and all sequences in seqs2.

    Parameters
    ----------
    scores_matrx : np.ndarray
        Alignment scores matrix aligning two collections of sequences seqs1 and seqs2.
        Matrix should have the dimension M * N, where M is the length of seqs1, and N
        is the length of seqs2.

    Returns
    -------
    np.array
        Mean alignment scores for the alignments between each sequence in
        seqs1 and all sequences in seqs2.
    """
    return np.mean(scores_matrix, axis=1)


def get_max_align_scores(scores_matrix: np.ndarray) -> np.ndarray:
    """Given an alignment scores matrix aligning two collections of sequences
    seqs1 and seqs2, return the max alignment scores for the alignments
    between each sequence in seqs1 and all sequences in seqs2.

    Parameters
    ----------
    scores_matrx : np.ndarray
        Alignment scores matrix aligning two collections of sequences seqs1 and seqs2.
        Matrix should have the dimension M * N, where M is the length of seqs1, and N
        is the length of seqs2.

    Returns
    -------
    np.array
        Max alignment scores for the alignments between each sequence in
        seqs1 and all sequences in seqs2.
    """
    return np.amax(scores_matrix, axis=1)


def get_min_align_scores(scores_matrix: np.ndarray) -> np.ndarray:
    """Given an alignment scores matrix aligning two collections of sequences
    seqs1 and seqs2, return the min alignment scores for the alignments
    between each sequence in seqs1 and all sequences in seqs2.

    Parameters
    ----------
    scores_matrx : np.ndarray
        Alignment scores matrix aligning two collections of sequences seqs1 and seqs2.
        Matrix should have the dimension M * N, where M is the length of seqs1, and N
        is the length of seqs2.

    Returns
    -------
    np.array
        Min alignment scores for the alignments between each sequence in
        seqs1 and all sequences in seqs2.
    """
    return np.amin(scores_matrix, axis=1)


def select_seqs_from_alignment(
    scores_matrix: np.ndarray, num_seqs_selected: int, select_type: str
) -> Dict[str, np.ndarray]:
    """Select a specified number of sequences with the highest mean or max alignment scores
    and return the sequences' corresponding indices and alignment score values.

    Parameters
    ----------
    scores_matrix : np.ndarray
        Alignment scores matrix aligning two collections of sequences seqs1 and seqs2.
        Matrix should have the dimension M * N, where M is the length of seqs1, and N
        is the length of seqs2.
    num_seqs_selected : int
        Number of sequences to select.
    select_type : str
        "mean" or "max".

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing two keys, "selected seqs indices" or "selected seqs scores".

    Raises
    ------
    ValueError
        If select_type is not either "mean" or "max".
    """
    if type == "mean":
        scores = get_mean_align_scores(scores_matrix=scores_matrix)
    elif type == "max":
        scores = get_max_align_scores(scores_matrix=scores_matrix)
    else:
        raise ValueError(f"Invalid select type: {select_type}. Must be mean or max.")

    selected_dict = {}
    scores_sorted_inds = np.argsort(scores)

    selected_inds = scores_sorted_inds[-num_seqs_selected:]
    selected_dict["selected seqs indices"] = selected_inds

    selected_scores = scores[selected_inds]
    selected_dict["selected seqs scores"] = selected_scores

    return selected_dict
