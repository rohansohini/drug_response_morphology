import os

from cajal import sample_seg
from cajal.qgw import quantized_gw_parallel


def needs_recompute(output_path: str, input_dir: str) -> bool:
    '''
    Whether output_path needs to be (re)computed: it doesn't exist yet,
    or something in input_dir has been modified more recently than it -
    e.g. extract_masks re-extracted a frame because the upstream cell
    selection changed. A plain exists-only check would otherwise keep
    reusing a cached output that no longer matches its current inputs.

    Inputs:
        output_path: the cached file whose freshness is in question
        input_dir:   directory of files it was computed from
    Outputs:
        True if output_path is missing or older than something in input_dir
    '''
    if not os.path.exists(output_path):
        return True
    output_mtime = os.path.getmtime(output_path)
    return any(
        os.path.getmtime(os.path.join(input_dir, f)) > output_mtime
        for f in os.listdir(input_dir)
    )


def run_cajal(
    infolder:   str,
    out_dir:    str,
    mask_type:  str,
    n_points:   int,
    n_jobs:     int,
    n_clusters: int,
) -> None:
    '''
    Run the CAJAL pipeline (icdm + quantized GW distances) on a folder of segmentation masks.
    Skips icdm computation if the output already exists.

    Inputs:
        infolder:   directory containing segmentation mask TIFFs
        out_dir:    output directory for CAJAL CSVs
        mask_type:  'cell' or 'nuc', used for output file naming
        n_points:   number of points to sample per cell
        n_jobs:     number of parallel processes
        n_clusters: number of clusters for quantized GW
    '''
    icdm_csv = os.path.join(out_dir, f"{mask_type}_qgw_{n_points:03d}P_icdm.csv")
    dist_csv = os.path.join(out_dir, f"{mask_type}_qgw_{n_points:03d}P_dists.csv")

    if not needs_recompute(icdm_csv, infolder):
        print(f"  Skipping icdm computation: {icdm_csv} already exists and is up to date")
    else:
        sample_seg.compute_icdm_all(
            infolder,
            icdm_csv,
            n_sample=n_points,
            num_processes=n_jobs,
            discard_cells_with_holes=False,
            only_longest=True,
        )

    quantized_gw_parallel(
        intracell_csv_loc=icdm_csv,
        num_processes=n_jobs,
        num_clusters=n_clusters,
        out_csv=dist_csv,
    )
