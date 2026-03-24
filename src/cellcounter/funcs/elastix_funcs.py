"""ITK-Elastix registration and transformation wrappers.

Provides:
- registration(): Image-to-image registration with affine + bspline
- transformation_coords(): Transform point coordinates between image spaces
- transformation_img(): Apply transform to a new image

Uses ITK-Elastix for deformable registration of microscopy images
to reference atlas.
"""

import logging
import os
import re
from pathlib import Path

import itk
import numpy as np
import numpy.typing as npt
import pandas as pd

from cellcounter.constants import CACHE_DIR, Coords
from cellcounter.funcs.io_funcs import silent_remove, write_tiff

logger = logging.getLogger(__name__)


def registration(
    fixed_img_fp: Path | str,
    moving_img_fp: Path | str,
    output_img_fp: Path | str,
    affine_fp: Path | str | None = None,
    bspline_fp: Path | str | None = None,
) -> npt.NDArray:
    """Uses ITKElastix for image registration.

    Params:
        fixed_img_fp: Filepath to the fixed (reference) image.
        moving_img_fp: Filepath to the moving image to be registered.
        output_img_fp: Filepath for the output registered image.
        affine_fp: Optional filepath to custom affine parameter file.
        bspline_fp: Optional filepath to custom bspline parameter file.
    """
    fixed_img_fp = Path(fixed_img_fp)
    moving_img_fp = Path(moving_img_fp)
    output_img_fp = Path(output_img_fp)
    output_img_dir = output_img_fp.parent

    # Load images as float (required for elastix)
    fixed_image = itk.imread(str(fixed_img_fp), itk.F)
    moving_image = itk.imread(str(moving_img_fp), itk.F)

    # Set up parameter object with affine and bspline maps
    parameter_object = itk.ParameterObject.New()

    # Affine parameter map
    if affine_fp is not None:
        affine_fp = Path(affine_fp)
        parameter_object.AddParameterFile(str(affine_fp))
    else:
        params_affine = itk.ParameterObject.GetDefaultParameterMap("affine")
        parameter_object.AddParameterMap(params_affine)

    # Bspline parameter map
    if bspline_fp is not None:
        bspline_fp = Path(bspline_fp)
        parameter_object.AddParameterFile(str(bspline_fp))
    else:
        params_bspline = itk.ParameterObject.GetDefaultParameterMap("bspline")
        parameter_object.AddParameterMap(params_bspline)

    # Run registration using OO interface
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    elastix_object.SetParameterObject(parameter_object)
    elastix_object.SetLogToConsole(True)

    # Update filter (required)
    elastix_object.UpdateLargestPossibleRegion()

    # Get results
    result_image = elastix_object.GetOutput()
    result_transform_parameters = elastix_object.GetTransformParameterObject()

    # Save transform parameters to output directory for later use with transformix
    for i in range(result_transform_parameters.GetNumberOfParameterMaps()):
        param_map = result_transform_parameters.GetParameterMap(i)
        itk.ParameterObject.WriteParameterFile(
            param_map, str(output_img_dir / f"TransformParameters.{i}.txt")
        )

    # Save output image
    arr = np.asarray(result_image)
    write_tiff(arr, output_img_fp)

    # Removing temporary and unnecessary elastix files
    for i in output_img_dir.iterdir():
        if re.search(r"^IterationInfo.(\d+).R(\d+).txt$", str(i.name)):
            silent_remove(i)

    return arr


def transformation_coords(
    coords: pd.DataFrame,
    moving_img_fp: Path | str,
    output_img_fp: Path | str,
) -> pd.DataFrame:
    """Get transformation coords from elastix.

    Uses the transformation parameter output from registration to transform
    cell coordinates from the fixed image space to moving image space.

    Params:
        coords: A pd.DataFrame of points, with the columns `x`, `y`, and `z`.
        moving_img_fp: Filepath of the moved image from registration (typically the reference image).
        output_img_fp: Filepath of the outputted image from registration (the fixed image after warping). Important that the TransformParameters files are in this folder.

    Returns:
        A pd.DataFrame of the transformed coordinates from the fixed image space to the moving image space.
    """
    moving_img_fp = Path(moving_img_fp)
    output_img_fp = Path(output_img_fp)
    # Getting the output image directory (i.e. where registration results are stored)
    reg_dir = output_img_fp.parent
    # Store temporary transformix outputs
    out_dir = CACHE_DIR / f"transformed_coords_{os.getpid()}"
    out_dir.mkdir(exist_ok=True)

    # Load moving image
    moving_image = itk.imread(str(moving_img_fp), itk.F)

    # Create fixed points file
    # NOTE: xyz, NOT zyx
    _make_fixed_points_file(
        coords[[Coords.X.value, Coords.Y.value, Coords.Z.value]].values,
        out_dir / "temp.dat",
    )

    # Load transform parameters from registration
    transform_parameter_object = itk.ParameterObject.New()
    transform_parameter_object.AddParameterFile(
        str(reg_dir / "TransformParameters.0.txt")
    )
    transform_parameter_object.AddParameterFile(
        str(reg_dir / "TransformParameters.1.txt")
    )

    # Set up Transformix object
    transformix_object = itk.TransformixFilter.New(moving_image)
    transformix_object.SetFixedPointSetFileName(str(out_dir / "temp.dat"))
    transformix_object.SetTransformParameterObject(transform_parameter_object)
    transformix_object.SetOutputDirectory(str(out_dir))
    transformix_object.SetLogToConsole(False)

    # Execute transformation
    transformix_object.UpdateLargestPossibleRegion()

    # Converting transformix output to df
    coords_transformed = _transformix_file2coords(str(out_dir / "outputpoints.txt"))
    # # Clean up temporary files
    # silent_remove(out_dir)
    return coords_transformed


def _make_fixed_points_file(coords: npt.NDArray, fixed_points_fp: Path | str) -> None:
    """Get fixed points from file.

    https://simpleelastix.readthedocs.io/PointBasedRegistration.html

    Takes a list of `(x, y, z, ...)` arrays
    and converts it to the .pts file format for transformix.
    """
    fixed_points_fp = Path(fixed_points_fp)
    with fixed_points_fp.open(mode="w") as f:
        f.write("index\n")
        f.write(f"{coords.shape[0]}\n")
        for i in np.arange(coords.shape[0]):
            f.write(f"{coords[i, 0]} {coords[i, 1]} {coords[i, 2]}\n")


def _transformix_file2coords(output_points_fp: str) -> pd.DataFrame:
    """Transform file to coordinates.

    Takes filename of the transformix output points
    and converts it to a pd.DataFrame of points.

    Params:
        output_points_fp: Filename of the ouput points.

    Returns:
        pd.DataFrame of points with the columns `x`, `y`, and `z`.
    """
    try:
        df = pd.read_csv(output_points_fp, header=None, sep=";")
    except pd.errors.EmptyDataError:
        # If there are no points, then return empty df
        return pd.DataFrame(columns=[Coords.Z.value, Coords.Y.value, Coords.X.value])
    df.columns = df.loc[0].str.strip().str.split(r"\s").str[0]
    # Try either "OutputIndexFixed" or "OutputPoint"
    df = df["OutputPoint"].apply(
        lambda x: [float(i) for i in x.replace(" ]", "").split("[ ")[1].split()]
    )
    return pd.DataFrame(
        df.values.tolist(), columns=[Coords.X.value, Coords.Y.value, Coords.Z.value]
    )[[Coords.Z.value, Coords.Y.value, Coords.X.value]]


def transformation_img(
    moving_img_fp: Path | str,
    output_img_fp: Path | str,
) -> npt.NDArray:
    """Transform image.

    Uses the transformation parameter output from registration to transform
    an image from the moving image space to fixed image space.

    Params:
        moving_img_fp: Filepath of the moving image to transform.
        output_img_fp: Filepath of the output image from registration. Important that the TransformParameters files are in the parent folder.

    Returns:
        A numpy array of the transformed image.
    """
    moving_img_fp = Path(moving_img_fp)
    output_img_fp = Path(output_img_fp)

    # Getting the output image directory (i.e. where registration results are stored)
    reg_dir = output_img_fp.parent

    # Store temporary transformix outputs
    out_dir = CACHE_DIR / f"transformed_img_{os.getpid()}"
    out_dir.mkdir(exist_ok=True)

    # Load moving image
    moving_image = itk.imread(str(moving_img_fp), itk.F)

    # Load transform parameters from registration
    transform_parameter_object = itk.ParameterObject.New()
    transform_parameter_object.AddParameterFile(
        str(reg_dir / "TransformParameters.0.txt")
    )
    transform_parameter_object.AddParameterFile(
        str(reg_dir / "TransformParameters.1.txt")
    )

    # Set up Transformix object
    transformix_object = itk.TransformixFilter.New(moving_image)
    transformix_object.SetTransformParameterObject(transform_parameter_object)
    transformix_object.SetOutputDirectory(str(out_dir))
    transformix_object.SetLogToConsole(False)

    # Execute transformation
    transformix_object.UpdateLargestPossibleRegion()

    # Get result image
    result_image = transformix_object.GetOutput()

    # Clean up temporary files
    silent_remove(out_dir)

    return np.asarray(result_image)
