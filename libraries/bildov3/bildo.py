#!/usr/bin/env python3

import sys
import os, shutil
import osr, gdal, ogr
import numpy as np
import pandas as pd


class OpenImageDataSourceError(ValueError):
    pass

class NotImplementedError(ValueError):
    pass

class BufferNotAssign(ValueError):
    pass

class SpacingNotAssign(ValueError):
    pass

class OutputFileNotSpecified(ValueError):
    pass

class bildo(object):
    # from osgeo import gdal, osr

    # import numpy as np
    # import pandas as pd

    def __init__(self, parallel = False):
        # attrs
        self.dataSource = None
        self.arrays = None
        self.path = None
        self.sensor = None
        self.extent = None
        self.crs = None
        self.geotransform = None
        self.dims = None
        self.format = None

        # options
        self.parallel = False
        if parallel:
            self.parallel = True
            self.par = parallelBildo(self)


    # Functions to write into disk
    def writeToDisk(self, output, data_type = None, **kwargs):
        from spatialFuntions import create_raster
        from osgeo import gdal
       
        in_ds = self.dataSource
        data = self.arrays

        if in_ds is None:
            raise RuntimeError("source data source is None")

        if data_type is None:
            data_type = gdal.GDT_Float32
           
        create_raster(in_ds, fn = output, data = data,
                      data_type = data_type, **kwargs)

    def readArrays(self):
        """
        Read all bands as numpy.narrays

        Returns
        -------
        Multidimensional np.array
        """
        from spatialFunctions import readArrays_

        if self.sensor == "MODIS" or self.format == "HDF":
            raise Warning("Sensor MODIS or format HDF can be demanding. I suggest to try another approach")
        self.arrays = readArrays_(self.dataSource)

    def getRasterExtent(self):
        """
        returns raster exent xmin, xmax, ymin, ymax

        Returns
        -------
        List with xmin, xmax, ymin and ymax values.

        """

        if self.dataSource == None:
            self.extent = None

        else:
            r = self.dataSource
            ulx, xres, xskew, uly, yskew, yres = r.GetGeoTransform()
            lrx = ulx + (r.RasterXSize * xres)
            rly = uly + (r.RasterYSize * yres)

            # xmin, xmax, ymin and ymax
            extent = [ulx, lrx, rly, uly]
            self.extent = extent

    def setDataSource(self, ds):
        self.dataSource = ds

    def setArrays(self, arrays):
        self.arrays = arrays

    def setPath(self, path):
        self.path = path

    def setSensor(self, sensor):
        self.sensor = sensor

    def setExtent(self, extent):
        self.extent = extent

    def setCRS(self, crs):
        self.crs = crs

    def setDims(self, dims):
        self.dims = dims

    def setFormat(self, format):
        self.format = format

    def setGeoTransform(self, geotransform):
        self.geotransform = geotransform

    def createParallelFramework(self, folderPath = "/tmp/parallelBildo"):
        self.parallel = True
        self.par = parallelBildo(self)



#####################################################
############ PARALLEL CHILD CLASS ###################
#####################################################
class parallelBildo(object):
    def __init__(self, parent):
        self.parent = parent
        self.folder = None
        self.tiles_path = None
        self.tiles_buffer_path = None
        self.tiles_buffer_folder = None
        self.tiles_folder = None

        # Buffers needed for parallel processing
        self.buffer = None
        self.spacing = None

        self.ncores = None


    def startParallel(self, folderPath ="/tmp/parallelBildo", ncores = None):
        """
        Creates folder for save parallel tiles

        Parameters
        ----------
        folderPath : str
            Folder name. The default is "/tmp/parallelBildo".

        Returns
        -------
        None.

        """
        import multiprocessing as mp

        # Checking parallel status of the parent class
        # if self.parent.parallel is False:
        #     self.parent.startParallel()

        # Creating and assignging
        os.makedirs(folderPath, exist_ok = True)
        self.folder = folderPath

        # Cheking cores
        if ncores is None:
            ncores = mp.cpu_count()-1

        self.ncores = ncores

    def stopParallel(self):
        import shutil

        shutil.rmtree(self.folder)

        self.folder = None
        self.ncores = None
        self.parent.parallel = False



    def createTiles(self, buffer = True):
        from spatialFunctions import create_layer

        if self.buffer is None and buffer == True:
            raise BufferNotAssign(self.buffer)

        if self.spacing is None:
            raise SpacingNotAssign(self.spacing)

        # Get xmin, xmax, ymin and ymax from extent
        # ensuring they are divisible by spacing
        # def divisibleBySpacing(e, s):
        #     out = np.ceil(np.ceil(e)/s)*s
        #     return out

        xmin, xmax, ymin, ymax = self.parent.extent
        # ytop = np.ceil(np.ceil(ymax) / self.spacing) * self.spacing
        # ybottom = np.floor(np.floor(ymin) / self.spacing) * self.spacing
        # xright = np.ceil(np.ceil(xmax) / self.spacing) * self.spacing
        # xleft = np.floor(np.floor(xmin) / self.spacing) * self.spacing
        ytop = (ymax/self.spacing)*self.spacing
        ybottom = (ymin/self.spacing)*self.spacing
        xright = (xmax/self.spacing)*self.spacing
        xleft = (xmin/self.spacing)*self.spacing

        # Defining number of rows and columns
        # rows = int((ytop - ybottom) / self.spacing)
        # cols = int((xright - xleft) / self.spacing)


        it = 0
        list_features = []
        if buffer: list_features_buffer = []
        for i in np.arange(xleft, xright, self.spacing):
            xleft = i
            xright = xleft + self.spacing
            ytop_backup = ytop
            for j in np.arange(ytop, ybottom, -self.spacing):
                # print(xleft, xright, ybottom, ytop)
                dict_fields = {}

                ytop = j
                ybottom = ytop - self.spacing

                # polygon = shp.geometry.Polygon([
                #     (xleft, ytop),
                #     (xright, ytop),
                #     (xright, ybottom),
                #     (xleft, ybottom)
                # ]
                # )
                # polygons.append(polygon)

                # Create ring
                ring = ogr.Geometry(ogr.wkbLinearRing)

                # Coordinates in clockwise order
                ring.AddPoint(xleft, ytop)
                ring.AddPoint(xright, ytop)
                ring.AddPoint(xright, ybottom)
                ring.AddPoint(xleft, ybottom)
                ring.AddPoint(xleft, ytop)

                # Transform ring to polygon
                polygon = ogr.Geometry(ogr.wkbPolygon)
                polygon.AddGeometry(ring)
                polygon_wkt = polygon.ExportToWkt()

                dict_fields["id"] = it
                list_features.append([polygon_wkt, [dict_fields]])

                if buffer:
                    polygon_geom = ogr.CreateGeometryFromWkt(polygon_wkt)
                    # print(polygon_geom)
                    polygon_buffer = polygon_geom.Buffer(self.buffer)
                    # print(polygon_buffer)
                    polygon_buffer_wkt = polygon_buffer.ExportToWkt()

                    list_features_buffer.append([polygon_buffer_wkt, [dict_fields]])

                it += 1
            ytop = ytop_backup

        output_tiles = f"{self.folder}/tiles.gpkg"
        if buffer:
            output_tiles_buffer = f"{self.folder}/tiles_buffer.gpkg"

        # print(list_features[0])
        create_layer(output_tiles, list_features, crs = self.parent.crs)
        self.tiles_layer_path = output_tiles
        self.tiles_folder = output_tiles.split(".")[0]
        os.makedirs(self.tiles_folder, exist_ok = True)

        if buffer:
            create_layer(output_tiles_buffer, list_features_buffer, crs = self.parent.crs)
            self.tiles_layer_buffer_path = output_tiles_buffer
            self.tiles_buffer_folder = output_tiles_buffer.split(".")[0]




    def cutTiles(self):
        """
        Create buffered tiles

        Returns
        -------
        None.

        """

        from spatialFunctions import masking_tiles, get_rasterExtent
        from spatialFunctions import layer_within_raster

        if self.tiles_buffer_folder is not None:
            output_folder = self.tiles_buffer_folder
            layer_tiles = self.tiles_layer_buffer_path

        else:
            output_folder = self.tiles_folder
            layer_tiles = self.tiles_layer_path

        masking_tiles(layer_tiles,
                      raster_path = self.parent.path,
                      output_folder = output_folder,
                      field = "id",
                      crs = self.parent.crs,
                      naming = True
                      )

        lista_tiles = os.listdir(output_folder)
        self.tiles = list(map(
            lambda x,y: f"{x}/{y}",
            [output_folder]*len(lista_tiles),
            lista_tiles
            ))


    def unbufferTiles(self, output_folder = None):
        """
        Create unbuffered tiles

        Returns
        -------
        None.

        """

        from spatialFunctions import masking_tiles, get_rasterExtent
        from spatialFunctions import layer_within_raster

        if output_folder is None:
            output_folder = f"{self.tiles_folder}/computed"

        else:
            output_folder = f"{self.tiles_folder}/{output_folder}"

        self.tiles_output_folder = output_folder
        layer_tiles = self.tiles_layer_path

        for tile in self.tiles_buffered_output:
            try:
                masking_tiles(layer_tiles,
                          raster_path = tile,
                          output_folder = output_folder,
                          field = "id",
                          crs = self.parent.crs,
                          extent = True,
                          naming = True
                          )
            except:
                pass

        lista_tiles = os.listdir(output_folder)
        self.tile_outputs = list(map(
            lambda x,y: f"{x}/{y}",
            [output_folder]*len(lista_tiles),
            lista_tiles
            ))



    def doParallel(self, func, input_folder = None, output_folder = None):
        from multiprocessing import Pool


        if output_folder is None:
            output_folder = "computed"

        if input_folder is None:
            inputs = self.tiles
        else:
            tmp_intiles = []
            for t in os.listdir(f"{self.tiles_buffer_folder}/{input_folder}"):
                if os.path.isdir(f"{self.tiles_buffer_folder}/{t}"):
                    pass
                else:
                    tmp_intiles.append(t)

            inputs = list(map(lambda x,y: f"{x}/{y}",
                              [f"{self.tiles_buffer_folder}/{input_folder}"]*len(tmp_intiles),
                              tmp_intiles
                              ))

        self.tiles_buffered_output_folder = f"{self.tiles_buffer_folder}/{output_folder}"
        os.makedirs(self.tiles_buffered_output_folder, exist_ok = True)

        def inout_files(a, b):
            import os

            a1 = a.split("/")
            a2 = a1[len(a1)-1]


            if os.path.isdir(a):
                return None
            else:
                return {a: f"{b}/{a2}"}

        # inputs = self.tiles
        ncores = self.ncores
        inouts = list(map(inout_files,
                           inputs,
                           [self.tiles_buffered_output_folder]*len(inputs)
                           )
                       )
        inouts = list(filter(lambda x: x is not None, inouts))

        # Do parallel
        with Pool(ncores) as p:
            p.map(func, inouts)


        lista_tiles = os.listdir(self.tiles_buffered_output_folder)
        self.tiles_buffered_output = list(map(
            lambda x,y: f"{x}/{y}",
            [self.tiles_buffered_output_folder]*len(lista_tiles),
            lista_tiles
            ))

    def mergeTiles(self, output = None, inNodata = 0, outNodata = 0, resampling = "bilinear"):

        if output is None:
            raise OutputFileNotSpecified(output)

        xmin, xmax, ymin, ymax = self.parent.extent

        # vrt optiosn need the ouput extent in a different order
        oextent = (xmin, ymin, xmax, ymax)
        vrt_options = gdal.BuildVRTOptions(addAlpha = False,
                                           # srcNodata = inNodata,
                                           # VRTNodata = outNodata,
                                           outputBounds = oextent,
                                           resampleAlg = resampling
                                           )
        vrt_output = f"{self.folder}/tmpVRT.vrt"
        ds_vrt = gdal.BuildVRT(vrt_output, self.tile_outputs, options = vrt_options)
        ds_newtif = gdal.Translate(output, ds_vrt)

        del ds_vrt
        del ds_newtif

        self.merged_output = output




#####################################################
################### FUNCTIONS #######################
#####################################################

def openBildo(path_or_ds, sensor = "Default", parallel = False):
    """
    Open an image from a path or a gdal.Dataset

    Parameters
    ----------
    gdal_ds: String path or gdal.Dataset
    sensor: Sensor/Platform used
    """

    from osgeo import gdal, osr

    def getFormat(path):
        tail = path.split(".")
        tail = tail[len(tail)-1]

        if "hdf" in tail:
            formato = "HDF"

        if "tiff" in tail:
            formato = "GTiff"

        else:
            formato = "Not Defined"

        return formato


    # Defining sensor
    if sensor in ["Default", "MODIS"]:
        sensor = sensor
       
    else:
        sensor = "Default"
        print("Sensor without not implemented yet. Default option triggered.")

    # Initiating class
    if sensor == "MODIS":
        from bildov2 import modisBildo

        if type(path_or_ds) is str:
            path = path_or_ds
        else:
            raise RuntimeError("ds is a data source. Are you sure?")
        image = modisBildo()
        image.setPath(path)
        image.setSensor(sensor)
        image.setFormat("hdf")
        image.getSelfTiles()
        image.getSelfProducts()
        image.getSelfTilesPath()
        image.getSelfProductsPath()
        image.getSelfImagesDict()

    else:
        if parallel:
            image = bildo(parallel = True)
        else:
            image = bildo()
        # Checking if it is path or ds
        if type(path_or_ds) is str:
            ds = gdal.Open(path_or_ds)
            path = path_or_ds
        elif type(path_or_ds) is gdal.Dataset:
            ds = path_or_ds
            path = ds.GetDescription()
        image.setDataSource(ds)
        image.readArrays()
        image.setPath(path)
        image.getRasterExtent()
        image.setCRS(ds.GetProjection())
        image.setGeoTransform(ds.GetGeoTransform())
        image.setSensor(sensor)
        image.setFormat(getFormat(path))
        image.setDims(image.arrays.shape)

    return image
