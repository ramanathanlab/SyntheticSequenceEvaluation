"""Run AlphaFold on synthetic sequences."""
import concurrent.futures as cf
import os
import subprocess
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm import tqdm

# This assumes you are running from within the alphafold singularity container
cmd_template = """
/opt/alphafold/run.sh -d /lambda_stor/data/hsyoo/AlphaFoldData  -o {} -f \
{} -t 2020-05-01 -p casp14 -m model_1,model_2,model_3,model_4,model_5 \
-a {} &> {}
"""


def run_single(filename: Path, gpu: str, output_dir: Path) -> None:
    """Run AlphaFold on a single sequence.

    Parameters
    ----------
    filename : Path
        Name of output file.
    gpu : str
        GPU to run the process.
    output_dir : Path
        Output path to save the AlphaFold results.
    """
    output_dir = output_dir / filename.with_suffix("").name
    output_dir.mkdir(parents=True)
    logfile = output_dir / filename.with_suffix(".log").name
    cmd = cmd_template.format(output_dir, filename, gpu, logfile)
    subprocess.run(cmd, shell=True, capture_output=True)  # Blocking
    done_file = Path("/tmp/{}.done".format(filename.name))
    while not done_file.exists():
        time.sleep(30)


def process(input_dir: Path, output_dir: Path) -> None:
    """Run AlphaFold in parallel.

    Parameters
    ----------
    input_dir : Path
        Path to access the fasta file whose sequences AlphaFold will run on.
    output_dir : Path
        Output path to save the AlphaFold results.

    Raises
    ------
    ValueError
        If CUDA_VISIBLE_DEVICES is not set.
    """
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpus is None:
        raise ValueError("Please set CUDA_VISIBLE_DEVICES")
    available_gpus = set(gpus.split(","))
    num_workers = len(available_gpus)

    # For each fasta file create a subprocess and launch alphafold on one gpu
    files = list(input_dir.glob("*.fasta"))
    print("NUMBER OF FILES: {}".format(len(files)))

    futures = []
    with cf.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for file in tqdm(files):
            print("Available gpus: ", available_gpus)
            gpu = available_gpus.pop()
            future = executor.submit(run_single, file, gpu, output_dir)
            futures.append(future)
            if not available_gpus:
                # Process fasta files in batches
                finished = cf.wait(futures, return_when=cf.ALL_COMPLETED)
                print("No available, available: ", available_gpus)
                for future in finished.done:
                    print("Finished, available: ", available_gpus)
                    gpu = future.result()  # Return gpu when finished with it
                    available_gpus.add(gpu)


def parse_args() -> Namespace:
    """Parse command line arguments.

    Returns
    -------
    Namespace
        Parsed arguments.
    """
    parser = ArgumentParser(description="Run AlphaFold on synthetic sequences.")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=Path,
        required=True,
        help="Path to access fasta files to run AlphaFold.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        required=True,
        help="Path to save AlphaFold results.",
    )
    return parser.parse_args()


def main() -> None:
    """Run AlphaFold in parallel."""
    process(args.input_dir, args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main()
