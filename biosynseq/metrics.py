"""Sequence Metrics."""
import functools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List

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
    # get embedding distance
    embed_dist = distance_matrix(embed_avg, embed_avg)

    # get upper triangular matrix and flatten
    embed_dist_upper = np.triu(embed_dist).flatten()
    embed_dist_upper = embed_dist_upper[embed_dist_upper > 0]

    return embed_dist_upper


def get_scores_flatten(scores_matrix: np.ndarray) -> np.ndarray:
    """Given a scores matrix, flatten all scores into a one-dimensional array.
    scores_matrix can be global_scores_matrix or local_scores_matrix.

    Parameters
    ----------
    scores_matrix : np.ndarray
        Scores matrix containing pairwise alignment scores.

    Returns
    -------
    np.ndarray
        One-dimensional array of flattened scores.
    """
    # need to subtract from diagonal since alignment scores between the same seqeunce will be large positive values
    scores_upper = np.triu(scores_matrix - np.diag(np.diag(scores_matrix))).flatten()
    scores_upper = scores_upper[scores_upper > 0]
    return scores_upper


def get_scores_df(
    embed_avg: np.ndarray, scores_matrix: np.ndarray, alignment_type: str = "global"
) -> pd.DataFrame:
    """Compute a two-column scores dataframe comparing the embedding L2 distance and the pairwise alignment scores,
    given the average embeddings for the sequences, the scores matrix, and the alignment type.

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
        Two-column dataframe comparing the embedding L2 distance and the pairwise alignment scores.

    Raises
    ------
    ValueError
        If alignment_type is neither "global" nor "local."
    """
    # scores_matrix could be either global_scores_matrix or local_scores_matrix, specified by global_score = True or False

    if alignment_type == "global":
        align_key = "Global Alignment Score"
    elif alignment_type == "local":
        align_key = "Local Alignment Score"
    else:
        raise ValueError(f"Invalid alignment type: {alignment_type}")

    embed_dist_upper = get_embed_dist_flatten(embed_avg=embed_avg)
    scores_upper = get_scores_flatten(scores_matrix=scores_matrix)
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
        If alignment_type is neither "global" nor "local."
    """

    if alignment_type == "global":
        align_key = "Global Alignment Score"
    elif alignment_type == "local":
        align_key = "Local Alignment Score"
    else:
        raise ValueError(f"Invalid alignment type: {alignment_type}")

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
