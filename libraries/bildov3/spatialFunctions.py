#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:33:31 2020

@author: jcintasr-work
"""

import os
import gdal
import ogr
import osr
import numpy as np
# import matplotlib.pyplot as plt
# import time


class NoneDsErrorClass(ValueError):
    pass
    #print(ValueError)


def get_rasterExtent(raster, dictionary=False):
    if type(raster) is str:
        r = gdal.Open(raster)
    else:
        r = raster
    ulx, xres, xskew, uly, yskew, yres = r.GetGeoTransform()
    lrx = ulx + (r.RasterXSize * xres)
    rly = uly + (r.RasterYSize * yres)

    # xmin, xmax, ymin and ymax
    extent = [ulx, lrx, rly, uly]
    if dictionary:
        return({raster: extent})
    else:
        return (extent)


def getPolyBoundary(raster):
    """
    Creates polygon from extent
    """

    if type(raster) is str:
        r = gdal.Open(raster)
    else:
        r = raster

    srs = r.GetProjection()
    lx, rx, ly, uy = get_rasterExtent(r)

    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(lx, uy)  # ul corner
    ring.AddPoint(rx, uy)  # ur corner
    ring.AddPoint(rx, ly)  # lr corner
    ring.AddPoint(lx, ly)  # ll corner
    ring.AddPoint(lx, uy)  # ul corner again to close polygon

    poly = ogr.Geometry(org.wkbPolygon)
    poly.AddGeometry(ring)

    return poly


def getCoordinates(pointGeometry):
    x = pointGeometry.GetX()
    y = pointGeometry.GetY()
    return([x, y])


def getPixelIndexFromCoordinates(pointGeometry, GeoTransform):
    x, y = getCoordinates(pointGeometry)

    # Adjusting coordinates to pixel indexes.
    # It only works on porjections with no rotation
    xorigin = GeoTransform[0]
    xscale = GeoTransform[1]
    yorigin = GeoTransform[3]
    yscale = GeoTransform[5]

    pi = int((y - yorigin)/yscale)
    pj = int((x - xorigin)/xscale)

    return([pi, pj])


def ds_readAsArray(ds, index0=False):
    # This images seems to not be 0 indexed code(I don't know how)
    if index0 is True:
        nBands = ds.RasterCount
        rango = range(nBands)
    else:
        nBands = ds.RasterCount + 1
        rango = range(1, nBands)

    arrayList = list()
    for k in rango:
        tmpBand = ds.GetRasterBand(k)
        if tmpBand is not None:
            tmpArray = tmpBand.ReadAsArray()
            arrayList.append(tmpArray)

    return np.array(arrayList)


def normalise(array):
    minimum, maximum = array.min(), array.max()
    normal = (array - minimum)*((255)/(maximum - minimum))+0
    return normal


def plotRGB(nArray, r=3, g=2, b=1, normalization=False):
    import matplotlib.pyplot as plt

    if type(nArray) is gdal.Dataset:
        nArray = ds_readAsArray(nArray)

    if type(nArray) is not np.ndarray:
        print("nArray must be an array or a gdal.Dataset")
        return None

    red = nArray[:][:][r-1]
    green = nArray[:][:][g-1]
    blue = nArray[:][:][b-1]

    if normalization:
        def normalise(array):
            minimum, maximum = array.min(), array.max()
            normal = (array - minimum)*((255)/(maximum - minimum))+0
            return normal

        red = normalise(red)
        green = normalise(green)
        blue = normalise(blue)

    rgb = np.dstack((red, green, blue)).astype(int)

    plt.imshow(rgb)


def getLayerExtent(layer_path):
    longitud = len(layer_path.split("."))
    driver_name = layer_path.split(".")[longitud - 1]
    if driver_name == "gpkg":
        driver = ogr.GetDriverByName("GPKG")
    elif driver_name == "shp":
        driver = ogr.GetDriverByName("ESRI Shapefile")

    elif driver_name == "kml":
        driver = ogr.GetDriverByName("KML")

    ds = driver.Open(layer_path)
    xmin, xmax, ymin, ymax = ds.GetLayer().GetExtent()
    extent = [xmin, ymin, xmax, ymax]

    del ds

    return extent


def create_raster(in_ds, fn, data, data_type, nodata=None, driver="GTiff",
                  band_names=None, createOverviews=False, crs=None,
                  GeoTransform=None, rows_cols=None, return_ds=False):
    """
    Based on Geoprocessing with python.
    Create a one-band GeoTiff

    in_ds         - datasource to copy projection and geotransform from
    fn            - path to the file to create
    data          - NumPy array containing data to write
    data_type     - output data type
    nodata        - optional NoData value
    band_names    - optional. It gives a name to each band for easier identification. It has to have same length than data dimensons.
    """

    driver = gdal.GetDriverByName(driver)
    #     print(band_names)
    # Creating out raster

    # Getting columns and rows
    if rows_cols is None:
        # columns = in_ds.RasterXSize
        # rows = in_ds.RasterYSize
        lengthShape = len(data.shape)
        if lengthShape > 2:
            nbands, rows, columns = data.shape
        else:
            nbands = 1
            rows, columns = data.shape
    else:
        if type(rows_cols) is not tuple:
            print("rows_cols has to be a tuple")
            return None
        # rows = rows_cols[0]
        # columns = rows_cols[1]
        rows = rows_cols[1]
        columns = rows_cols[2]

    out_ds = driver.Create(fn, columns, rows, nbands, data_type)
    print(out_ds)
    if(out_ds is None):
        print("output creation failed!. Unable to create output datasource")
        return None

    if GeoTransform is not None:
        out_ds.SetGeoTransform(GeoTransform)
    else:
        out_ds.SetGeoTransform(in_ds.GetGeoTransform())

    # Assigning out raster projection and geotransform
    if crs is None:
        out_ds.SetProjection(in_ds.GetProjection())
    else:
        srs = osr.SpatialReference()
        if type(crs) is int:
            srs.ImportFromEPSG(crs)
        elif type(crs) is str:
            try:
                srs.ImportFromWkt(crs)
            except:
                srs.ImportFromProj4(crs)

        elif type(crs) is osr.SpatialReference:
            srs = crs

        out_ds.SetProjection(srs.ExportToWkt())

    # Iterate through bands if necessary
    if nbands > 1:
        for k in range(0, nbands):
            out_band = out_ds.GetRasterBand(k + 1)
            if nodata is not None:
                out_band.SetNoDataValue(nodata)
            # out_band.WriteArray(data[:, :, k])
            out_band.WriteArray(data[k, :, :])

            if band_names is not None:
                out_band.SetDescription(band_names[k])
                metadata = out_band.GetMetadata()
                metadata = f"TIFFTAG_IMAGEDESCRIPTION={band_names[k]}"
                out_band.SetMetadata(metadata)

    else:
        out_band = out_ds.GetRasterBand(1)
        if nodata is not None:
            out_band.SetNoDataValue(nodata)

        out_band.WriteArray(data)
    #         print(out_band.ReadAsArray())

    out_band.FlushCache()
    out_band.ComputeStatistics(False)
    if createOverviews:
        out_band.BuildOverViews('average', [2, 4, 8, 16, 32])

    if return_ds:
        del out_band
        return out_ds
    else:
        del out_band
        del out_ds
        return "Done!"


def create_layer(output, feature_list,
                 driver_name="GPKG", crs=25830,
                 geom_type=ogr.wkbPolygon, data_type=ogr.OFTReal):
    """
    output_name         -  Name of the file to craete with extension
    feature_dictionary  -  list with two elements, geometry and a list with a dictionary with the name of the field at the keys and its values at the values.
    driver_name         -  driver to use. GPKG by default.
    epsg                -  epsg code to assign projection
    geom_type           -  geom type of the geometry suplied
    data_type           -  data_type of the values
    """

    # Getting name of the output without path and extension
    output_layer_tmp = output.split("/")
    output_layer_tmp2 = output_layer_tmp[len(output_layer_tmp) - 1]
    output_layer_name = output_layer_tmp2.split(".")[0]
    #     print(output_layer_name)

    # Getting srs
    out_srs = osr.SpatialReference()

    if type(crs) is int:
        out_srs.ImportFromEPSG(crs)
    elif type(crs) is str:
        out_srs.ImportFromWkt(crs)
    #     print(out_srs)

    # create output layer
    driver = ogr.GetDriverByName(driver_name)
    if os.path.exists(output):
        driver.DeleteDataSource(output)
    out_ds = driver.CreateDataSource(output)
    if out_ds is None:
        print("output data source is None")
        return 1
    out_layer = out_ds.CreateLayer(
        output_layer_name, geom_type=geom_type, srs=out_srs)

    # very important matter to reset Reading after define out layer
    out_layer.ResetReading()
    #     print(out_layer)

    # Iterate through list to get fields and create them
    count = 0
    for feature in feature_list:
        diccionario_tmp = feature[1]
        diccionario = diccionario_tmp[0]

        fieldNames = []
        for field in diccionario.keys():
            # Checking if the field alerady exists
            if count > 0:
                for f in range(out_layer.GetLayerDefn().GetFieldCount()):
                    fieldNames.append(
                        out_layer.GetLayerDefn().GetFieldDefn(f).name)

            if field not in fieldNames:
                outFieldDefn = ogr.FieldDefn(field, data_type)
                out_layer.CreateField(outFieldDefn)

            count += 1

    # Get Layer Definition
    out_layerDefn = out_layer.GetLayerDefn()
    #     print(out_layerDefn.GetGeomFieldDefn())
    #     print(out_layerDefn)

    # Iterate through list to get geometries, fields and values
    # it = 0
    for feature in feature_list:
        geomwkt = feature[0]
        geom = ogr.CreateGeometryFromWkt(geomwkt)

        diccionario_tmp = feature[1]
        diccionario = diccionario_tmp[0]

        ofeat = ogr.Feature(out_layerDefn)
        ofeat.SetGeometry(geom)
        for field, value in diccionario.items():
            ofeat.SetField(field, value)

        #             print(field, value*1.0)

        out_layer.CreateFeature(ofeat)

    out_layer.SyncToDisk()
    out_ds = None


def layer_within_raster(raster_extent, layer_geom, lesser_lextent=False):
    """
    check if a layer is inside the raster
    :param raster_extent: extent of the raster
    :param layer_geom: layer geom
    :param lesser_lextent: If True a smaller extent is evaluated
    :return:
    """
    rxmin, rxmax, rymin, rymax = raster_extent
    lxmin, lxmax, lymin, lymax = layer_geom.GetEnvelope()

    if lesser_lextent:
        # Getting a smaller bounding box
        lxmin = lxmin + 100
        lxmax = lxmax - 100
        lymin = lymin + 100
        lymax = lymax - 100

    i = 0
    if lxmin >= rxmin:  # 1. upper left corner
        i += 1
    if lymax <= rymax:  # 2. upper right corner
        i += 1
    if lxmax <= rxmax:  # 3. lower right corner
        i += 1
    if lymin >= rymin:  # 4. lower left corner
        i += 1

    if i == 4:
        out = True
    else:
        out = False
    return (out)


def compareSpatialReference(ds1, ds2):
    if type(ds1) is gdal.Dataset:
        tmp1 = ds1.GetProjection()
        proj1 = osr.SpatialReference()
        proj1.ImportFromWkt(tmp1)

    elif type(ds1) is ogr.DataSource:
        proj1 = ds1.GetLayer().GetSpatialRef()

    elif type(ds1) is osr.SpatialReference:
        proj1 = ds1

    if type(ds2) is gdal.Dataset:
        tmp2 = ds2.GetProjection()
        proj2 = osr.SpatialReference()
        proj2.ImportFromWkt(tmp2)

    elif type(ds2) is ogr.DataSource:
        proj2 = ds2.GetLayer().GetSpatialRef()

    elif type(ds2) is osr.SpatialReference:
        proj2 = ds2

    if proj1.IsSame(proj2):
        return True
    else:
        return False


def reproject(image, output_folder=None, crs_to=25830, returnPath=False,
              driver="GTiff", resampling=gdal.GRA_NearestNeighbour):
    """
    This function reprojects a raster image
    :param image: path to raster image
    :param output_folder: output folder where the output image will be saved
    :param epsg_to: coordinate epsg code to reproject into
    :param memDs: If True, it returns the output path
    :return: returns a virtual data source
    """

    if driver == "MEM":
        returnPath = False
        output_folder = None

    if output_folder is None:
        driver = "MEM"
        returnPath = False

    else:
        splitted = image.split("/")
        lenout = len(splitted)
        out_name = splitted[lenout-1]
        output = f"{output_folder}/reprojeted_{out_name}"

    dataset = gdal.Open(image)
    srs = osr.SpatialReference()
    if type(crs_to) is int:
        srs.ImportFromEPSG(crs_to)
    elif type(crs_to) is str:
        srs.ImportFromWkt(crs_to)

    vrt_ds = gdal.AutoCreateWarpedVRT(
        dataset, None, srs.ExportToWkt(), resampling)

    if returnPath:
        # cols = vrt_ds.RasterXSize
        # rows = vrt_ds.RasterYSize
        gdal.GetDriverByName(driver).CreateCopy(output, vrt_ds)
        return(output)

    else:
        return(vrt_ds)


def naming_convention(raster_path, geometry):
    """
    Creates naming based on the raster name and geometries: date_xmin-ymax_sentineltile_band
    :param raster_path: Path to raster file
    :param geometry: geom
    :return:
    """
    # xmin, xmax, ymin, ymax
    xmin, xmax, ymin, ymax = geometry.GetEnvelope()
    splitted = raster_path.split("/")
    len_splitted = len(splitted)
    name_tmp1 = splitted[len_splitted - 1]
    name = name_tmp1.split(".")[0]
    # name_splitted = name.split("_")
    # if len(name_splitted) < 3:
    outaname = f"{name}_{float(xmin)}-{float(ymax)}"
    # else:
    #     sent_tile = name_splitted[0]
    #     band = name_splitted[2]
    #     date_tmp = name_splitted[1]
    #     date = date_tmp.split("T")[0]

    #     # outaname = f"{date}_{int(xmin)}_{int(ymax)}_{sent_tile}_{band}"
    #     outaname = f"{date}_{float(xmin)}-{float(ymax)}_{sent_tile}_{band}"
    return (outaname)


def masking_tiles(layer_tiles,
                  raster_path,
                  output_folder,
                  field="fid_id",
                  naming=False,
                  extent=False,
                  lesser_lextent=False,
                  reproyectar=False,
                  crs=None
                  ):
    """
    It creates tiles from a raster image based on a grid previously created
    :param layer_tiles: Path to grid
    :param raster_path: Path to raster
    :param output_folder: Path to output folder
    :param field: Field with cut tiles with
    :param naming: Apply naming rule
    :param extent: Cut with extent
    :param lesser_lextent: create an smaller extent
    :param reproyectar: If True, reprojection is applied
    :param epsg: EPSG code of the srs to reproject into
    :return:
    """
    if os.path.exists(output_folder) is False:
        os.mkdir(output_folder)

    if reproyectar:
        raster_path2 = raster_path
        raster_path = reproject(raster_path, "/tmp",
                                crs_to=crs, return_output_path=True)
        print(raster_path)

    driver = ogr.GetDriverByName("GPKG")
    ds = driver.Open(layer_tiles)
    layer = ds.GetLayer()
    for feature in layer:
        geom = feature.geometry()
        fid = feature.GetField(field)
        if naming:
            if reproyectar:
                out_name = naming_convention(raster_path2, geom)
            else:
                out_name = naming_convention(raster_path, geom)
        else:
            out_tmp = raster_path.split("/")
            out_tmp2 = out_tmp[len(out_tmp) - 1]
            out_name = out_tmp2.split(".")[0]

        output = f"{output_folder}/{out_name}.tif"

        if extent:
            raster_extent = get_rasterExtent(raster_path)
            sepuede = layer_within_raster(
                raster_extent, geom, lesser_lextent=lesser_lextent)

            if sepuede:
                xmin, xmax, ymin, ymax = geom.GetEnvelope()
                lextent = [xmin, ymin, xmax, ymax]

                ds2 = gdal.Warp(output,
                                raster_path,
                                format="GTiff",
                                outputBounds=lextent)

                del ds2

        else:
            ds2 = gdal.Warp(output,
                            raster_path,
                            format="GTiff",
                            cutlineDSName=layer_tiles,
                            cutlineWhere=f"{field} = '{fid}'",
                            cropToCutline=True)
            del ds2

    layer.ResetReading()
    ds.FlushCache()

    del ds


def whichMin(to_compare, values, axis=0):
    """
    Returns an array with the values minimum to_compares is found. To_compare and values must have the same order. Note that, when values in to_compare are the same, then the minimum value is returned.

    Parameters
    ----------
    to_compare : TYPE
        DESCRIPTION.
    values : TYPE
        DESCRIPTION.

    Returns
    -------
    array with the same dimensions.

    """
    # Checking for arrays in the same dimension
    if to_compare.shape != values.shape:
        print("Error!! Both arrays haven't the same dimensions")
        return None

    # Create boolean mask (True or False)
    mask = np.ma.make_mask(to_compare == np.amin(to_compare, axis=axis))

    # Applying mask and getting maximum value in case two are selected
    if np.issubdtype(values.dtype, np.datetime64):
        # https://stackoverflow.com/questions/45793044/numpy-where-typeerror-invalid-type-promotion
        arrout = np.where(mask == True, values, np.datetime64("NaT"))
        arrout = np.nanmax(arrout, axis=0)

    else:
        arrout = np.min(values * mask, axis=axis)

    return arrout


def whichMax(to_compare, values, axis=0):
    """
    Returns an array with the maximum values of to compare are found. "to_compare" and values must have the same order. Note that, when values in to_compare ar the same, then the maximum value is returned.

    Parameters
    ----------
    to compare: TYPE
        Description
    values: TYPE
        DESCRIPTION

    Returns
    -------
    array with the same dimensions
    """

    # Checking that arrays are in the same dimension
    if to_compare.shape != values.shape:
        print("Error!! Both arrays haven't the same dimensions")
        return None

    # Create boolean mask (True or False)
    mask = np.ma.make_mask(to_compare == np.nanmax(to_compare, axis=axis))

    # Applying mask and getting maximum value in case two are selected
    if np.issubdtype(values.dtype, np.datetime64):
        # https://stackoverflow.com/questions/45793044/numpy-where-typeerror-invalid-type-promotion
        arrout = np.where(mask == True, values, np.datetime64("NaT"))
        arrout = np.nanmax(arrout, axis=0)
    else:
        arrout = np.max(values * mask, axis=0)

    return arrout


def whichMinAngle_withoutClouds(values, angles):

    # angles + 0.01 in case there is a perfect 0 angle.
    # This way min angles now are the maximum angles. To get rid of the 0 case later on,
    # when merging with values I will want the maximum value (minimum), so the 0 (no cloud),
    # is not on may way
    mask_angles = ((1/(angles+0.01))*100)
    mask_values = np.ma.make_mask(values != 0)

    # Getting the right format (each pixel in a row with the K dimensions in columns)
    maskVal = mask_values * 0b1
    maskVal[maskVal == 0] = 0b0

    coso = mask_angles * mask_values
    coso2 = coso == np.amax(coso, axis=0)
    ole = values * coso2
    # ole2 = np.max(ole, axis = 0)

    return ole


def readArrays_(ds, newColsRows=None):
    """   

    Parameters
    ----------
    ds : TYPE
        DESCRIPTION.
    newColsRows : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if newColsRows is None:

        nbands = ds.RasterCount + 1
        rango = range(1, nbands)

        arrayList = list()
        for k in rango:
            tmpBand = ds.GetRasterBand(k)
            # maybe here I can get out band name

            if tmpBand is not None:
                arrayList.append(tmpBand.ReadAsArray())

    else:
        new_cols, new_rows = newColsRows

        old_cols = ds.RasterXSize
        old_rows = ds.RasterYSize

        height_factor = old_rows/new_rows
        width_factor = old_cols/new_cols

        if height_factor > 1 or width_factor > 1:
            print("Angles resolution larger than tile. NOT IMPLEMENTED")
            return None

        nbands = ds.RasterCount + 1
        rango = range(1, nbands)

        arrayList = list()
        for k in rango:
            tmpBand = ds.GetRasterBand(k)

            if tmpBand is not None:
                arrayList.append(tmpBand.ReadAsArray(0, 0,
                                                     old_cols, old_rows,
                                                     newColsRows[0], newColsRows[1]))

    return np.stack(arrayList)


def createProjWin(geotransform, ncols, nrows):
    ulX, width, wshift, ulY, hshift, height = geotransform
    lrX = ulX + ncols * width
    lrY = ulY + nrows * height

    return [ulX, ulY, lrX, lrY]


def reprojectProjWin(srcProj, dstProj, projWin):
    from pyproj import transform

    if srcProj == dstProj:
        print("Same projection. Returning original projWin")
        return projWin

    ulX, ulY, lrX, lrY = projWin
    projUlX, projUlY = transform(srcProj, dstProj, ulX, ulY)
    projLrX, projLrY = transform(srcProj, dstProj, lrX, lrY)

    return [projUlX, projUlY, projLrX, projLrY]


def resampleRasters(toDs, fromDs, output, resampleAlg="bilinear"):

    # Loading libraries
    import bildo
    from osgeo import gdal, osr
    import numpy as np

    # Getting information needed
    to_projection = toDs.GetProjection()
    to_ncols = toDs.RasterXSize
    to_nrows = toDs.RasterYSize
    to_geotransform = toDs.GetGeoTransform()

    from_projection = fromDs.GetProjection()

    # Getting osr objects
    to_srs = osr.SpatialReference()
    from_srs = osr.SpatialReference()
    to_srs.ImportFromWkt(to_projection)
    from_srs.ImportFromWkt(from_projection)

    # creating projwins
    to_projwin = createProjWin(to_geotransform, to_ncols, to_nrows)
    from_projwin = reprojectProjWin(to_projection, from_projection, to_projwin)

    # Reprojecting
    # First crop fromDs in its srs

    if to_srs.IsSame(from_srs):
        print("CRS is the same between both rasters. Cropping and resampling")
        try:
            outDs = gdal.Translate(output,
                                   fromDs,
                                   projWin=to_projwin,
                                   outputBounds=to_projwin,
                                   # height=to_nrows,
                                   # width=to_ncols,
                                   xRes=to_geotransform[1],
                                   yRes=abs(to_geotransform[5]),
                                   resampleAlg=resampleAlg
                                   )

            if outDs is None:
                print("gdal.Translated with xRes and yRes failed. Trying height and widht instead")
                raise NoneDsErrorClass(
                    "gdal.Translate returned None. Trying height, width instead")

        except:
            outDs = gdal.Translate(output,
                                   fromDs,
                                   projWin=to_projwin,
                                   height=to_nrows,
                                   width=to_ncols,
                                   # xRes=to_geotransform[1],
                                   # yRes=abs(to_geotransform[5]),
                                   outputBounds=to_projwin,
                                   resampleAlg=resampleAlg
                                   )

        if outDs is None:
            raise NoneDsErrorClass(
                "gdal.Translate was unable to create a data source. None returned!")

        return outDs

    #tmp_outputfile = f"{tmp_folder}/from_cropped.tif"
    # gettign name
    output_name = output.split("/")
    output_name = output_name[len(output_name)-1]

    # print("Translating to fromDs projwin")
    # from_cropped = gdal.Translate(f"/tmp/from_cropped_{output_name}",
    #                               fromDs,
    #                               projWin=from_projwin,
    #                               outputBounds = from_projwin,
    #                               resampleAlg=resampleAlg)
    # #print(f"fromDs_proj: {from_projection} \n from_projwin: {from_projwin}")
    
    # if from_cropped is None:
    #     print("Cropped operation in source projection (fromDs one) returned None")
    #     return None
    # elif len(np.unique(from_cropped.GetRasterBand(1).ReadAsArray())) == 1:
    #     raise RuntimeError("from_cropped failed. Just an unique value is returned")
   
    #ERASE AFTER FIXED
    #return from_cropped
    # Second, reproject cropped image to sinusoidal
    #tmp_outputfile2 = f"{tmp_folder}/cropped_to_sinu.tif"
    if resampleAlg == "nearest":
        resampleAlg_warp = "near"
    else:
        resampleAlg_warp = resampleAlg

    print("Resampling toDs CRS and projwin extent")
    to_projwin_warp = [to_projwin[0], to_projwin[3], to_projwin[2], to_projwin[1]]
    cropped_sinu = gdal.Warp("/vsimem/cropped_sinu_{output_name}",
                             fromDs,
                             srcSRS = from_srs,
                             dstSRS=to_srs,
                             #xRes = to_geotransform[1],
                             #yRes = to_geotransform[5],
                             height = to_nrows,
                             width = to_ncols,
                             outputBounds = to_projwin_warp,
                             resampleAlg=resampleAlg_warp)
    if cropped_sinu is None:
        raise RuntimeError("Warp operation returned None")
    elif len(np.unique(cropped_sinu.GetRasterBand(1).ReadAsArray())) == 1:
        print("Warping to target SRS failed?. Just one value returned")
    # # It failed when rasters has an unique value
   
    #return cropped_sinu
    #del from_cropped
    # os.remove(tmp_outputfile)
    # Third, crop with sinusoidal window (to_projwin)
    print("Translating into target Ds")
    outDs = gdal.Translate(output,
                           cropped_sinu,
                           projWin=to_projwin,
                           height=to_nrows,
                           width=to_ncols,
                           outputBounds=to_projwin,
                           resampleAlg=resampleAlg
                           )

    del cropped_sinu
    # os.remove(tmp_outputfile2)
    return outDs


if __name__ == "__main__":
    a1 = np.array([[0, 2], [3, 4]])
    a2 = np.array([[100, 150], [50, 200]])
    b1 = np.array([[5, 6], [0, 8]])
    b2 = np.array([[200, 100], [500, 300]])
    c1 = np.array([[1, 0], [10, 0]])
    c2 = np.array([[100, 230], [60, 10]])
    d1 = np.array([[10, 300], [0, 5]])
    d2 = np.array([[150, 300], [40, 170]])
    a = np.stack([a1, a2])
    b = np.stack([b1, b2])
    c = np.stack([c1, c2])
    d = np.stack([d1, d2])
    values = np.stack([a1, b1, c1, d1])
    tocompare = np.stack([a2, b2, c2, d2])

    # mask_angles = np.ma.make_mask(tocompare == np.amin(tocompare, axis = 0))
    mask_angles = ((1/tocompare+0.01)*1000).astype(int)
    mask_values = np.ma.make_mask(values != 0)
    masks = np.stack([mask_values, mask_angles])

    # Getting the right format (each pixel in a row with the K dimensions in columns)
    maskVal = mask_values * 0b1
    maskVal[maskVal == 0] = 0b0

    # maskAng = mask_angles * 0b0010
    # maskAng[maskAng == 0] = 0b01

    # Useful later on
    # maskAng = np.reshape(maskAng, (-1,8,1))[0]
    # maskVal = np.reshape(maskVal, (-1,8,1))[0]
    # coso = np.concatenate([maskAng, maskVal], axis = 1)

    # getting values
    # coso = np.bitwise_or(maskAng, maskVal)
    coso = mask_angles * maskVal

    # number 6 is the bit with the best situation, so let's filter cases closer to it
    # coso2 = coso & 6
    coso2 = coso == np.amax(coso, axis=0)

    # Now we have two acceptable values 6 and 4.
    # 6 means the pixel is in the best situation possible. No clouds best view angle
    # 4 means the pixel is not the best option, but is the one we choose. No clouds band view angle.

    # The problem is a pixel can have both bit values. So the best way to deal with it is keeping the maximum value, which is the best situation possible.
    # final_filter = coso2 == np.amax(coso2, axis  = 0)

    # Now we apply the filter, getting the desire value
    # ole = values * final_filter
    ole = values * coso2

    # Reducing to 2d. I am summing them since all the not desired values are 0
    ole2 = np.max(ole, axis=0)

    # All right values
    # coso = coso & 4 # (0b0100 is ArCn & AnrCn. All clear conditions)


# =============================================================================
#     # k dimension into columns
# =============================================================================
    test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    test = np.reshape(test, (2, 2, 2))
    test = np.reshape(test, (-1, 2, 4))[0].transpose()

    test2 = np.reshape(
        np.array([[9, 10], [11, 12], [13, 14], [15, 16]]), (2, 2, 2))
    test2 = np.reshape(test2, (-1, 2, 4))[0].transpose()

    test3 = np.concatenate([test, test2], axis=1)
