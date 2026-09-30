import os

from cajal import sample_seg
from cajal.qgw import quantized_gw_parallel

###### Ported unchanged from vem_fast_prolif_cajal/cajal_pipeline/run_cajal.py


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

    if os.path.exists(icdm_csv):
        print(f"  Skipping icdm computation: {icdm_csv} already exists")
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
