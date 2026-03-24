from pathlib import Path

from cellcounter.funcs.viewer_funcs import view_arrs_from_pfm

if __name__ == "__main__":
    # =========================================
    # CHANGE FOLDER PATHS HERE
    analysis_root_dir = "/path/to/analysis_output_folder"
    # CHANGE IMAGE NAME
    image_name = "proj_name"
    ##############################
    analysis_root_dir = Path(analysis_root_dir)
    analysis_img_dir = analysis_root_dir / image_name

    # Trimmer
    trimmer = (
        slice(None, None, None),
        slice(None, None, None),
        slice(None, None, None),
    )
    # Images to run
    # COMMENT OUT THE IMAGES THAT YOU DON'T WANT TO VIEW
    arrs_to_run = [
        # "raw",
        # "ref",
        # "annot",
        # "downsmpl1",
        # "downsmpl2",
        # "trimmed",
        # "bounded",
        # "regresult",
        # "premask_blur",
        # "mask_fill",
        # "mask_outline",
        # "mask_reg",
        "bgrm",
        "dog",
        "adaptv",
        "threshd",
        "threshd_volumes",
        "threshd_filt",
        "maxima",
        "wshed_volumes",
        "wshed_filt",
        # "threshd_final",
        # "maxima_final",
        # "wshed_final",
        # "points_raw",
        # "heatmap_raw",
        # "points_trfm",
        # "heatmap_trfm",
    ]
    # Viewing the images

    view_arrs_from_pfm(
        analysis_img_dir,
        arrs_to_run,
        trimmer,
        tuning=True,  # SET TO TRUE FOR TUNING AND FALSE FOR FINAL
    )
