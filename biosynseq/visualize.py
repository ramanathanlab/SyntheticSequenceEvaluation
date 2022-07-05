"""Visualize synthetic sequences using t-SNE and UMAP."""
import os
import numpy as np
import matplotlib.pyplot as plt # need to import matplotlib before pandas
import pandas as pd
from pathlib import Path
from mdlearn.utils import plot_scatter
from typing import List, Dict

from Bio import SeqIO, SeqUtils, SeqRecord
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

