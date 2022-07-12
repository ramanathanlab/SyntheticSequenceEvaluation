"""Sequence Metrics."""
import numpy as np
from pathlib import Path
from Bio import SeqUtils
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint


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


def gc_content(seqs):
    return [SeqUtils.GC(rec.seq) for rec in seqs]


def seq_length(seqs):
    return [len(rec.seq) for rec in seqs]


def molecular_weight(protein_seqs):
    return [SeqUtils.molecular_weight(rec.seq, "protein") for rec in protein_seqs]


def isoelectric_point(protein_seqs):
    return [IsoelectricPoint(seq).pi() for seq in protein_seqs]
