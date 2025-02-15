import os
import re

import numpy as np
import pandas as pd

from cellcounter import ELASTIX_ENABLED
from cellcounter.constants import CACHE_DIR, Coords
from cellcounter.utils.io_utils import silent_remove, write_tiff
from cellcounter.utils.logging_utils import init_logger_file
from cellcounter.utils.misc_utils import import_extra_error_func

# Optional dependency: elastix
if ELASTIX_ENABLED:
    import SimpleITK as sitk
else:
    import_extra_error_func("elastix")()


class ElastixFuncs:
    logger = init_logger_file(__name__)

    @classmethod
    def registration(
        cls,
        fixed_img_fp: str,
        moving_img_fp: str,
        output_img_fp: str,
        affine_fp: None | str = None,
        bspline_fp: None | str = None,
    ) -> np.ndarray:
        """
        Uses SimpleElastix (a plugin for SimpleITK)

        Params:
            TODO
        """
        output_img_dir = os.path.split(output_img_fp)[0]
        # Setting up Elastix object
        elastix_img_filt = sitk.ElastixImageFilter()
        # Setting the fixed, moving, and output image filepaths
        elastix_img_filt.SetFixedImage(sitk.ReadImage(fixed_img_fp))
        elastix_img_filt.SetMovingImage(sitk.ReadImage(moving_img_fp))
        elastix_img_filt.SetOutputDirectory(output_img_dir)
        # Parameter maps: translation, affine, bspline
        # Translation
        # parameter_map_translation = sitk.GetDefaultParameterMap("translation")
        # elastix_img_filt.SetParameterMap(parameter_map_translation)
        # Affine
        if affine_fp is not None:
            params_affine = sitk.ReadParameterFile(affine_fp)
        else:
            params_affine = sitk.GetDefaultParameterMap("affine")
        elastix_img_filt.SetParameterMap(params_affine)
        # Bspline
        if affine_fp is not None:
            params_bspline = sitk.ReadParameterFile(bspline_fp)
        else:
            params_bspline = sitk.GetDefaultParameterMap("bspline")
        elastix_img_filt.AddParameterMap(params_bspline)
        # Setting feedback and logging settings
        elastix_img_filt.LogToFileOff()
        elastix_img_filt.LogToConsoleOn()
        # Running registration
        elastix_img_filt.Execute()
        # Saving output file
        res_img = elastix_img_filt.GetResultImage()
        # sitk.WriteImage(res_img, output_img_fp)
        write_tiff(sitk.GetArrayFromImage(res_img), output_img_fp)
        # Removing temporary and unecessary elastix files
        for i in os.listdir(output_img_dir):
            # Removing IterationInfo files
            if re.search(r"^IterationInfo.(\d+).R(\d+).txt$", i):
                silent_remove(os.path.join(output_img_dir, i))
        return sitk.GetArrayFromImage(res_img)

    @classmethod
    def transformation_coords(
        cls,
        coords: pd.DataFrame,
        moving_img_fp: str,
        output_img_fp: str,
    ) -> pd.DataFrame:
        """
        Uses the transformation parameter output from registration to transform
        cell coordinates from the fixed image space to moving image space.

        Params:
            coords: A pd.DataFrame of points, with the columns, `x`, `y`, and `z`.
            moving_img_fp: Filepath of the moved image from registration (typically the reference image).
            output_img_fp: Filepath of the outputted image from registration (the fixed image after warping). Important that the TransformParameters files are in this folder.

        Returns:
            A pd.DataFrame of the transformed coordinated from the fixed image space to the moving image space.
        """
        # Getting the child pid of the process
        # coords = coords.compute() if isinstance(coords, Delayed) else coords
        # Getting the output image directory (i.e. where registration results are stored)
        reg_dir = os.path.dirname(output_img_fp)
        # Using CACHE_DIR to store temporary transformix outputs
        out_dir = os.path.join(CACHE_DIR, f"transformed_coords_{os.getpid()}")
        os.makedirs(out_dir, exist_ok=True)
        # Setting up Transformix object
        transformix_img_filt = sitk.TransformixImageFilter()
        # Setting the fixed points and moving and output image filepaths
        # Converting cells array to a fixed_points file
        # NOTE: xyz, NOT zyx
        cls._make_fixed_points_file(
            coords[[Coords.X.value, Coords.Y.value, Coords.Z.value]].values,
            os.path.join(out_dir, "temp.dat"),
        )
        transformix_img_filt.SetFixedPointSetFileName(os.path.join(out_dir, "temp.dat"))
        transformix_img_filt.SetMovingImage(sitk.ReadImage(moving_img_fp))
        transformix_img_filt.SetOutputDirectory(out_dir)
        # Transform parameter maps: from registration - affine, bspline
        transformix_img_filt.SetTransformParameterMap(
            sitk.ReadParameterFile(os.path.join(reg_dir, "TransformParameters.0.txt"))
        )
        transformix_img_filt.AddTransformParameterMap(
            sitk.ReadParameterFile(os.path.join(reg_dir, "TransformParameters.1.txt"))
        )
        # Setting feedback and logging settings
        transformix_img_filt.LogToFileOff()
        transformix_img_filt.LogToConsoleOff()
        # Execute cell transformation
        transformix_img_filt.Execute()
        # Converting transformix output to df
        coords_transformed = cls._transformix_file2coords(os.path.join(out_dir, "outputpoints.txt"))
        # Removing temporary and unecessary transformix files
        # silentremove(out_dir)
        return coords_transformed

    @staticmethod
    def _make_fixed_points_file(coords: np.ndarray, fixed_points_fp: str) -> None:
        """
        https://simpleelastix.readthedocs.io/PointBasedRegistration.html

        Takes a list of `(x, y, z, ...)` arrays and converts it to the .pts file format for transformix.
        """
        with open(fixed_points_fp, "w") as f:
            f.write("index\n")
            f.write(f"{coords.shape[0]}\n")
            for i in np.arange(coords.shape[0]):
                f.write(f"{coords[i, 0]} {coords[i, 1]} {coords[i, 2]}\n")

    @staticmethod
    def _transformix_file2coords(output_points_fp: str) -> pd.DataFrame:
        """
        Takes filename of the transformix output points and converts it to a pd.DataFrame of points.

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
        df = df["OutputPoint"].apply(lambda x: [float(i) for i in x.replace(" ]", "").split("[ ")[1].split()])
        return pd.DataFrame(df.values.tolist(), columns=[Coords.X.value, Coords.Y.value, Coords.Z.value])[
            [Coords.Z.value, Coords.Y.value, Coords.X.value]
        ]

    @classmethod
    def transformation_img(
        cls,
        moving_img_fp: str,
        output_img_fp: str,
    ) -> np.ndarray:
        """
        Uses the transformation parameter output from registration to transform
        cell coordinates from the fixed image space to moving image space.

        Params:
            coords: A pd.DataFrame of points, with the columns, `x`, `y`, and `z`.
            moving_img_fp: Filepath of the moved image from registration (typically the reference image).
            output_img_fp: Filepath of the outputted image from registration (the fixed image after warping). Important that the TransformParameters files are in this folder.

        Returns:
            A pd.DataFrame of the transformed coordinated from the fixed image space to the moving image space.
        """
        # Getting the output image directory (i.e. where registration results are stored)
        reg_dir = os.path.dirname(output_img_fp)
        # Using CACHE_DIR to store temporary transformix outputs
        out_dir = os.path.join(CACHE_DIR, f"transformed_coords_{os.getpid()}")
        os.makedirs(out_dir, exist_ok=True)
        # Setting up Transformix object
        transformix_img_filt = sitk.TransformixImageFilter()
        # Setting the fixed points and moving and output image filepaths
        # Converting cells array to a fixed_points file
        # transformix_img_filt.SetFixedPointSetFileName(
        #     os.path.join(out_dir, "temp.dat")
        # )
        transformix_img_filt.SetMovingImage(sitk.ReadImage(moving_img_fp))
        transformix_img_filt.SetOutputDirectory(out_dir)
        # Transform parameter maps: from registration - affine, bspline
        transformix_img_filt.SetTransformParameterMap(
            sitk.ReadParameterFile(os.path.join(reg_dir, "TransformParameters.0.txt"))
        )
        transformix_img_filt.AddTransformParameterMap(
            sitk.ReadParameterFile(os.path.join(reg_dir, "TransformParameters.1.txt"))
        )
        # Setting feedback and logging settings
        transformix_img_filt.LogToFileOff()
        transformix_img_filt.LogToConsoleOff()
        # Execute cell transformation
        transformix_img_filt.Execute()
        # Saving output file
        res_img = transformix_img_filt.GetResultImage()
        # # sitk.WriteImage(res_img, output_img_fp)
        # write_tiff(sitk.GetArrayFromImage(res_img), output_img_fp)
        # Removing temporary and unecessary transformix files
        silent_remove(out_dir)
        # return coords_transformed
        return sitk.GetArrayFromImage(res_img)
