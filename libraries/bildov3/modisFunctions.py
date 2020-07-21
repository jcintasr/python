#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:22:01 2020

@author: jcintasr-work
"""

class NoneDsError(ValueError):
    pass

def createBitmask(bit_array, quality="best", qc_band="State_1km", na_value=0):
    if qc_band == "State_1km":

        if quality == "best":
            cloud_state = (bit_array >> 0) & 3
            cloud_state = cloud_state*2
            cloud_state[cloud_state == 0] = 1
            cloud_state[cloud_state != 1] = 0

            cloud_shadow = (bit_array >> 2) & 1
            cloud_shadow = cloud_shadow * 2
            cloud_shadow[cloud_shadow != 2] = 1
            cloud_shadow[cloud_shadow == 2] = 0

            land = (bit_array >> 3) & 1

            land_shallow_water = (bit_array >> 3) & 3
            land_shallow_water[land_shallow_water != 1] = 0


            land_deep_water = (bit_array >> 3) & 5
            land_deep_water[land_deep_water >= 1] = 1
            land_deep_water[land_deep_water < 1] = 0

            aerosol = (bit_array >> 6) & 3
            aerosol[aerosol != 3] = 1
            aerosol[aerosol == 3] = 0

            # I'm getting rid of all the cirrus
            cirrus = (bit_array >> 8) & 3
            cirrus = cirrus * 2
            cirrus[cirrus == 0] = 1
            cirrus[cirrus > 1] = 0

            cloud_flag = (bit_array >> 10) & 1
            cloud_flag *= 2
            cloud_flag[cloud_flag == 0] = 1
            cloud_flag[cloud_flag != 1] = 0

            fire_flag = (bit_array >> 11) & 1
            fire_flag *= 2
            fire_flag[fire_flag == 0] = 1
            fire_flag[fire_flag != 1] = 0

            snow_flag = (bit_array >> 12) & 1
            snow_flag *= 2
            snow_flag[snow_flag == 0] = 1
            snow_flag[snow_flag != 1] = 0

            adjacent_to_cloud = (bit_array >> 13) & 1
            adjacent_to_cloud *= 2
            adjacent_to_cloud[adjacent_to_cloud == 0] = 1
            adjacent_to_cloud[adjacent_to_cloud != 1] = 0

            out = cloud_state * cloud_shadow * land * land_deep_water * land_shallow_water
            out = out * aerosol * cirrus * cloud_flag * fire_flag * snow_flag * adjacent_to_cloud

        else:
            print("Not implemented yet")
            return None

    else:
        print("Not implemented yet")
        return None

    # Getting an integer mask
    out = out.astype(int)
    if na_value != 0:
        out[out == 0] = na_value

    return out


def getBandsInfo(row, band, column):
    import pandas as pd

    newDF = pd.DataFrame.from_dict(row.subdatasets, "index")
    if type(column) is list:
        band_path = newDF[column].loc[band].to_dict("records")[0]

    else:
        band_path = newDF.loc[band].loc[column]
    return band_path


def getNamesFromHDFFile(HDFFile, sep="_"):
    tmpName = HDFFile.split(':')
    bandName = tmpName[len(tmpName)-1]
    bandName = bandName.split(sep)[2]
    tmpName = tmpName[len(tmpName)-3].split("/")
    tmpName = tmpName[len(tmpName)-1]
    tmpName = tmpName.split('"')[0]
    # product, ajdate, tile, version, datetime, format = tmpName.split(".")
    return tmpName


def createMasksVRT(lista, output_folder="modis/tmp"):
    import gdal
    import numpy as np
    from spatialFunctions import readArrays_

    from modisFunctions import createBitmask

    output_file = f"{output_folder}/tmp.vrt"

    vrt_options = gdal.BuildVRTOptions(separate=True)
    vrtDs = gdal.BuildVRT(output_file, lista, options=vrt_options)

    stack = readArrays_(vrtDs).copy()

    del vrtDs

    list_masks = list()
    for i in range(stack.shape[0]):
        list_masks.append(createBitmask(stack[i]))

    out_stack = np.stack(list_masks)

    return out_stack


def getBestAngleMosaic(bit_band, angle_band, refl_band,
                       output_folder="modis/tmp",
                       multiple_bands=False,
                       new_dimension_factor=4):
    """ 
    Parameters
n    ----------
    bit_band : list
        DESCRIPTION.
    angle_band : list
        DESCRIPTION.
    refl_band : list
        DESCRIPTION.

    Returns
    -------
    numpy.array
    """
    import numpy as np
    import gdal
    import os
    from spatialFunctions import readArrays_

    # getting . Sorting so they are in the same order
    # print(bit_band)
    masks = list(map(lambda x: x["band_path"], bit_band))
    masks.sort()
    angles = list(map(lambda x: x["band_path"], angle_band))
    angles.sort()

    if multiple_bands:
        reflectance = list()
        for bands in refl_band:
            tmpBands = list(map(lambda x: x["band_path"], bands))
            tmpBands.sort()
            reflectance.append(tmpBands)
    else:
        reflectance = list(map(lambda x: x["band_path"], refl_band))
        reflectance.sort()

    masks_stack = createMasksVRT(masks)

    # Defining vrt file options and output (for parallel processing)
    trash, ajdate, tile, version, time, formato = masks[0].split('"')[
        1].split(".")
    vrt_output_angles = f"{output_folder}/{tile}/{tile}_{ajdate}_angles.vrt"
    vrt_options = gdal.BuildVRTOptions(separate=True)
    angles_ds = gdal.BuildVRT(vrt_output_angles, angles, options=vrt_options)

    # load stack of arrays
    angles_stack = readArrays_(angles_ds)
    del angles_ds

    # Create mask stack
    from spatialFunctions import whichMinAngle_withoutClouds
    stackMask = whichMinAngle_withoutClouds(masks_stack, angles_stack)
    del masks_stack, angles_stack

    # Creating array with the right dimension
    stackMask_newDimension = np.repeat(stackMask, new_dimension_factor, axis=1)
    stackMask_newDimension = np.repeat(
        stackMask_newDimension, new_dimension_factor, axis=2)
    del stackMask

    # Applying mask
    if multiple_bands:
        for bands in reflectance:
            tmp_vrt = bands[0].split(":")
            tmp_vrt = tmp_vrt[len(tmp_vrt)-1].split("_")
            output_name_refl = tmp_vrt[len(tmp_vrt)-2]
            output_refl_vrt = f"{output_folder}/{tile}/{tile}_{ajdate}_{output_name_refl}.vrt"

            reflDs = gdal.BuildVRT(output_refl_vrt, bands, options=vrt_options)
            if reflDs is None:
                print("Unable to create vrt file")
                raise NoneDsError("datasourcer is wrong")

            ncols = reflDs.RasterXSize
            nrows = reflDs.RasterYSize
            refl_arrays = readArrays_(reflDs)

            list_output = list(map(lambda x, y: x*y,
                                   refl_arrays,
                                   stackMask_newDimension))

            # reshape arrays into one column
            one_column = list(map(lambda x: np.reshape(x, (-1, nrows*ncols, 1)),
                                  list_output))
            del list_output

            # concattenate into an uniquie array with bands as columns (useful for later on too)
            narray = np.concatenate(one_column, axis=2)
            del one_column

            # getting max value of the pixel (no clouds, best angle)
            best_version = np.max(narray, axis=2)
            del narray

            # returning to original shape
            best_out = np.reshape(best_version, (nrows, ncols))

            # writing into disk
            from spatialFunctions import create_raster
            output_refl_tif = f"{output_folder}/{tile}/{tile}_{ajdate[1:]}_{output_name_refl}.tif"
            create_raster(reflDs, output_refl_tif, best_out, gdal.GDT_Float32)

            del reflDs
            os.remove(output_refl_vrt)

    else:
        bands = reflectance
        tmp_vrt = bands[0].split(":")
        tmp_vrt = tmp_vrt[len(tmp_vrt)-1].split("_")
        output_name_refl = tmp_vrt[len(tmp_vrt)-2]
        output_refl_vrt = f"{output_folder}/{tile}/{tile}_{ajdate}_{output_name_refl}.vrt"

        reflDs = gdal.BuildVRT(output_refl_vrt, bands, options=vrt_options)
        ncols = reflDs.RasterXSize
        nrows = reflDs.RasterYSize
        refl_arrays = readArrays_(reflDs)

        list_output = list(map(lambda x, y: x*y,
                               refl_arrays,
                               stackMask_newDimension))

        # reshape arrays into one column
        one_column = list(map(lambda x: np.reshape(x, (-1, nrows*ncols, 1)),
                              list_output))
        del list_output

        # concattenate into an uniquie array with bands as columns (useful for later on too)
        narray = np.concatenate(one_column, axis=2)
        del one_column

        # getting max value of the pixel (no clouds, best angle)
        best_version = np.max(narray, axis=2)
        del narray

        # returning to original shape
        best_out = np.reshape(best_version, (nrows, ncols))

        # writing into disk
        from spatialFunctions import create_raster
        output_refl_tif = f"{output_folder}/{tile}/{tile}_{ajdate[1:]}_{output_name_refl}.tif"
        print(output_refl_tif)
        create_raster(reflDs, output_refl_tif, best_out, gdal.GDT_Float32)

        del reflDs
        os.remove(output_refl_vrt)

    os.remove(vrt_output_angles)
    return "Done!"


def createMODISFolders(path, tile=None, product=None):
    import os

    if tile and product:
        os.makedirs(f"{path}/{tile}/{product}", exist_ok=True)

    if tile and not product:
        os.makedirs(f"{path}/{tile}", exist_ok=True)

    if not tile and product:
        os.makedirs(f"{path}/{product}")


def getDekadsVRT(folder_path, band, return_files=False, output="/tmp/tmp.vrt"):
    """
    Creates vrt file from dekads.
    """

    import gdal
    import os

    def joinIf(a, b, endswith):
        if b.endswith(endswith):
            return f"{a}/{b}"

    files = os.listdir(folder_path)
    files.sort()
    list_files = list(map(lambda x, y: joinIf(x, y, f"{band}.tif"),
                          [folder_path]*len(files),
                          files
                          )
                      )
    # Getting rig of None values
    list_files = list(filter(lambda x: x is not None, list_files))

    # Building VRT file
    vrt_ds = gdal.BuildVRT(
        output, list_files, separate=True, resampleAlg="bilinear")
    if vrt_ds is None:
        print("Ups! vrt_ds is None. Something is wrong in the creation process")
        return None
    if return_files:
        return [vrt_ds, list_files]
    else:
        return vrt_ds


def getXArray3D(array, geotransform, third_dimension, labels=["time", "Y", "X"]):
    """
    Generate a three dimensional array with labeled dimensions
    """
    import xarray as xr
    import numpy as np

    gt = geotransform
    ndeep, nrows, ncols = array.shape

    X = np.arange(gt[0], gt[0]+(gt[1]*ncols), gt[1])[:ncols]
    Y = np.arange(gt[3], gt[3]+(gt[5]*nrows), gt[5])[:nrows]

    print(f"array: {array.shape} \nXshape: {X.shape}  Yshape: {Y.shape}")

    if len(third_dimension) != ndeep:
        print("third dimension with the wrong length")
        return 1

    xarray = xr.DataArray(array,
                          coords=[third_dimension, Y, X],
                          dims=labels
                          )

    return xarray
