#!/usr/bin/env jupyter

from bildov2 import bildo

import sys
import os, shutil
import osr, gdal, ogr
import numpy as np
import pandas as pd

class modisBildo(bildo):
    """
    self.tiles,
    self.products,
    self.tiles_path,
    self.products_path,
    self.images
    self.images_masked
    """


    def __init__(self):

        # Reading bildo attrs
        super().__init__()
        self.tiles = None
        self.products = None
        self.tiles_path = None
        self.products_path = None
        self.images = None
        self.images_masked = None

        #self.parent = parent
        # self.getSelfTiles()
        # self.getSelfProducts()

        # self.getSelfTilesPath()
        # self.getSelfProductsPath()
        # self.getSelfImagesDict()

    def __repr__(self):
        return f'{self.images[["product", "tile", "date", "version"]]}'


    def getTiles(self, path):
        tiles = os.listdir(path)
        return tiles

    def getSelfTiles(self):
        tiles = self.getTiles(self.path)
        self.tiles = tiles


    def getProducts(self, tiles):
        listProducts = list()
        for tile in tiles:
            for product in os.listdir(f"{self.path}/{tile}"):
                if product not in listProducts:
                    listProducts.append(product)
        return listProducts


    def getSelfProducts(self):
        self.products = self.getProducts(self.tiles)


    def getTilesPath(self, path, tiles):
        tiles_path = list(map(lambda x,y: f"{x}/{y}",
                              [path]*len(tiles),
                              tiles))
        return tiles_path

    def getSelfTilesPath(self):
        self.tiles_path = self.getTilesPath(self.path, self.tiles)

    def getProductsPath(self, tiles_path):
        products_path = list()
        for tile_path in tiles_path:
            products = os.listdir(tile_path)
            for product in products:
                products_path.append(f"{tile_path}/{product}")

        return products_path

    def getSelfProductsPath(self):
        self.products_path = self.getProductsPath(self.tiles_path)


    def getSubDatasetsDict(self, file_path):
        """
        Getting subdatasets from files.
        """
        import re
        subdatasets = gdal.Open(file_path).GetSubDatasets()

        list_dicts = list()
        for subdata in subdatasets:
            band_path = subdata[0]
            original_name = subdata[1]

            band_name = re.findall("\](.*?)\(", original_name)[0].strip()
            name = band_name.split(" ")[0]
            size = re.findall("\[(.*?)\]", original_name)[0].split("x")
            data_type = re.findall("\((.*?)\)", original_name)[0]

            subdata_dict = {
                "band_name": band_name,
                "band_path": band_path,
                "original_name": original_name,
                "size": size,
                "data_type": data_type
                }
            list_dicts.append([name, subdata_dict])

        return dict(list_dicts)


    def getImagesDict(self, products_path_list, format, new_naming = False):

        # def getFullPath(files, products_path):
        #     def get_product(product):
        #         splitted = product.split("/")
        #         out = splitted[len(splitted)-1]
        #         return out

        #     listFullPaths = list()
        #     for product_path in products_path:
        #         product = get_product(product_path)
        #         for file in files:
        #             if product in file:
        #                 out = f"{product_path}/{file})"
        #                 listFullPaths.append(out)

        #     return listFullPaths


        imagesList = list(map(lambda x: os.listdir(x), products_path_list))


        imagesList = list(filter(lambda x: x.endswith(format),
                                  np.concatenate(imagesList).tolist()))

        # fullPaths = getFullPath(imagesList, self.products_path)
        # self.full = fullPaths



        def buildDict(file, new_naming):

            def getNameInfo(file, new_naming):
                if new_naming:
                    product, tile, jdate, datetime, bandname, operation = file.split("_")
                    jDay = jdate[4:]
                    year = jdate[:4]
                    jDate = f"{year}-{jDay}"
                    version = "NA"
                    listout = [product, year, jDay, jDate, tile, version]
                else:
                    product, julianDateA, tile, version, juliandDateP, formato = file.split(".")
                    julianDateA = julianDateA[1:]
                    yearA = julianDateA[:4]
                    julianDayA = julianDateA[4:]
                    jDate = f"{yearA}-{julianDayA}"
                    listout = [product, yearA, julianDayA, jDate, tile, version]
                return listout

            def getFullPath(file, products_path, new_naming):

                product, year, doy, jdate, tile, version = getNameInfo(file, new_naming = False)
                for product_path in products_path:
                    if tile in product_path:
                        if product in product_path:
                            out = f"{product_path}/{file}"

                return out

            def getTileExtent(file_path):
                metadata = gdal.Open(file_path).GetMetadata()
                lon_min = metadata["WESTBOUNDINGCOORDINATE"]
                lon_max = metadata["EASTBOUNDINGCOORDINATE"]
                lat_min = metadata["SOUTHBOUNDINGCOORDINATE"]
                lat_max = metadata["NORTHBOUNDINGCOORDINATE"]
                del metadata

                return [lon_min, lon_max, lat_min, lat_max]



            product, yearA, julianDayA, jDate, tile, version = getNameInfo(file, new_naming)
            file_path = getFullPath(file, products_path_list, product)

            if new_naming:
                fileInfo_dict = {
                    "file_path": file_path,
                    "year": yearA,
                    "doy": julianDayA,
                    "tile": tile,
                    "date": jDate,
                    "size":(gdal.Open(file_path).RasterYSize, gdal.Open(file_path).RasterXSize),
                    "product": product,
                    "version": version
                    # "tile_extent": getTileExtent(file_path),
                    # I should chan"band_path"ge this, only read subdatasets needed on the query.
                    # For better performance
                    # "subdatasets": getSubDatasetsDict(file_path)

                    }
            else:
                fileInfo_dict = {
                    "file_path": file_path,
                    "year": yearA,
                    "doy": julianDayA,
                    "tile": tile,
                    "date": jDate,
                    "product": product,
                    "version": version,
                    "tile_extent": getTileExtent(file_path)
                    # I should chan"band_path"ge this, only read subdatasets needed on the query.
                    # For better performance this should leave commented out.
                    # Reading of subdatasets have to be done AFTER data frame is created.
                    #"subdatasets": getSubDatasetsDict(file_path)

                    }
           
            return fileInfo_dict
        #return 1
        listImages = dict(list(map(lambda x: [x, buildDict(x, new_naming)], imagesList)))
        # self.images = listImages

        import pandas as pd
        df = pd.DataFrame.from_dict(listImages, "index")
        df.index.name = "file"
        df.date = pd.to_datetime(df.date, format = "%Y-%j")
        df = df.sort_values(by = ["date"])
        df.reset_index(inplace = True)
        return df

    def getSelfImagesDict(self):
        products_path_list = self.products_path
        format = self.format.lower()
        df = self.getImagesDict(products_path_list, format)
        self.images = df

    def subsetByTile(self, tile, band, column = "band_path"):
        from modisFunctions import getBandsInfo

        if type(tile) is not list:
            tile = [tile]
        subset = self.images.query(f"tile in {tile}").copy()

        # I'M TRUSTING LIST ORDER. POSSIBLE ERROR IN HERE IF SOMETHING FAILS
        file_paths = subset.copy().file_path.values.tolist()
        subset["subdatasets"] = list(map(lambda x: self.getSubDatasetsDict(x), file_paths))


        bandsList = list(subset.apply(getBandsInfo, args = (band, column,), axis = 1))
        return bandsList


    def subsetByDate(date):
        pass


    def subsetByProduct(self, product, band, column = "band_path"):
        from modisFunctions import getBandsInfo

        if type(product) is not list:
            product = [product]
        subset = self.images.query(f"product in {product}").copy()

        # I'M TRUSTING LIST ORDER. POSSIBLE ERROR IN HERE IF SOMETHING FAILS
        file_paths = subset.copy().file_path.values.tolist()
        subset["subdatasets"] = list(map(lambda x: self.getSubDatasetsDict(x), file_paths))

        bandsList = list(subset.apply(getBandsInfo, args = (band, column, ), axis = 1))
        return bandsList


    def subset(self, products, bands, datespan, tiles, columns = ["band_path"]):
        from modisFunctions import getBandsInfo

        if type(products) is not list:
            products = [products]
        if type(bands) is not list:
            bands = [bands]
        if type(datespan) is not list:
            print("datespan has to have start and end time in a list")
            return None
        if type(tiles) is not list:
            tiles = [tiles]
        if type(columns) is not list:
            columns = [columns]

        df = self.images
        subset = df.copy().query(f"tile in {tiles}")
        subset = subset.copy().query(f"product in {products}")

        start_date = pd.to_datetime(datespan[0])
        end_date = pd.to_datetime(datespan[1])
        start_filter =  subset.date >= start_date
        end_filter = subset.date <= end_date
        subset = subset[start_filter & end_filter]


        # I'M TRUSTING LIST ORDER. POSSIBLE ERROR IN HERE IF SOMETHING FAILS
        file_paths = subset.copy().file_path.values.tolist()
        subset["subdatasets"] = list(map(lambda x: self.getSubDatasetsDict(x), file_paths))

        listout = list(subset.apply(getBandsInfo, args = (bands, columns, ), axis = 1))
        return listout



    # IT CAN BE IMPROVED WITH A BETTER STRUCTURE OF LOOPS AND SUBSET
    def getBitmaskPairs(self, df, products, tile_bands, mask_bands, datespan, column = "band_path"):
        if type(products) is not list:
            # print("product has been tranformed to list")
            products = [products]
            print(products)

        if type(tile_bands) is not list:
            tile_bands = [tile_bands]

        if type(mask_bands) is not list:
            mask_bands = [mask_bands]

        lista = []
        for product in products:
            if product in ["MYD09GQ", "MOD09GQ"]:
                for tile_band in tile_bands:
                    for mask_band in mask_bands:
                        # list_product_bands = self.subsetByProduct(product, tile_band, column)
                        list_product_bands = self.subset(df, product, tile_band, datespan, column)

                        product_newName = product[:len(product)-1] + "A"
                        # list_state_masks = self.subsetByProduct(product_newName, mask_band, column)
                        list_state_masks = self.subset(self.images, product_newName, mask_band, datespan, column)
                        # print(list_product_bands)


                        for i in range(len(list_product_bands)):
                            lista.append([list_product_bands[i], list_state_masks[i]])

                        # lista = dict(list(map(lambda x,y: [x, y], list_product_bands, list_state_masks)))

            else:
                print(f"Product {product} not implemented yet")
                return None

        return lista


    def applyMask(self, bitmask_pairs, output_folder = "modis/tmp/masked"):
        from modisFunctions import createBitmask
        from spatialFunctions import create_raster
        # it = 0

        # Eliminate output folder if it exists
        if os.path.isdir(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        # Saving output folder for later
        masked_folder = output_folder

        # Getting pairs from list
        products_path_list = list()
        for pair in bitmask_pairs:

            # Band and mask format values
            # tile_nrows = int(pair[0]["size"][0])
            # tile_ncols = int(pair[0]["size"][1])
            tile_nrows, tile_ncols = [*map(int, pair[0]["size"])]
            # tile_band = pair[0]["band_path"]
            tile_band = pair[0]["band_path"]
            # mask_nrows = int(pair[1]["size"][0])
            # mask_ncols = int(pair[1]["size"][1])
            mask_nrows, mask_ncols = [*map(int, pair[1]["size"])]
            # mask_band = pair[1]["band_path"]
            mask_band = pair[1]["band_path"]

            height_factor = mask_nrows/tile_nrows
            width_factor = mask_ncols/tile_ncols

            if height_factor > 1 or width_factor > 1:
                print("Mask resolution bigger than tile's!! Another approach is needed")
                return None



            # getting the output name
            # tile band have the next format:
            # data/MODIS/h10v05/MOD09GA/MOD09GA.A2008010.h10v05.006.2015169021720.hdf'
            tmpName = tile_band.split(':')
            bandName = tmpName[len(tmpName)-1]
            bandName = bandName.split("_")[2]
            tmpName = tmpName[len(tmpName)-3].split("/")
            tmpName = tmpName[len(tmpName)-1]
            product, ajdate, tile, version, datetime, format = tmpName.split(".")

            output_folder_candidate = f"{output_folder}/{tile}/{product}"

            # Creating folder structure
            # Check if folder exists and it changed
            if "output_folder_final" not in locals():
                output_folder_final = f"{output_folder}/{tile}/{product}"
                products_path_list.append(output_folder_final)
            else:
                if output_folder_final != output_folder_candidate:
                    output_folder_final = output_folder_candidate
                    products_path_list.append(output_folder_final)

            os.makedirs(output_folder_final, exist_ok = True)


            # output_mask = f"{output_folder}/{product}_{tile}_{ajdate[1:]}_{datetime}_mask.tif"
            output_file = f"{output_folder_final}/{product}_{tile}_{ajdate[1:]}_{datetime}_{bandName}_masked.tif"
            print(output_file)

            # getting band path
            # mask_band = pair[1]["band_path"]

            # Creating virtual raster for masking
            vrtMaskDs = gdal.BuildVRT(f"{output_folder_final}/tmp_mask.vrt", mask_band)
            vrtBandDs = gdal.BuildVRT(f"{output_folder_final}/tmp_band.vrt", tile_band)
            band_array = vrtBandDs.ReadAsArray()

            # Reading mask array with the proper size
            mask_array = vrtMaskDs.GetRasterBand(1).ReadAsArray(0, 0,
                                                                mask_ncols,
                                                                mask_nrows,
                                                                tile_ncols,
                                                                tile_nrows)
            # Creating mask
            bitmask = createBitmask(mask_array, quality = "best",
                                    qc_band = "State_1km", na_value = 0)

            # Applying mask
            masked = bitmask * band_array

            # Creating raster
            create_raster(in_ds = vrtBandDs, fn = output_file, data = masked,
                          data_type = gdal.GDT_Float32, driver = "GTiff")

            # gdal.Translate(f"{output_folder}/tmp{it}.tif", vrtDs)
            # print(f"{output_folder}/tmp{it}.tif")
            del vrtMaskDs
            del vrtBandDs
            if os.path.exists(f"{output_folder_final}/tmp_mask.vrt"):
                os.remove(f"{output_folder_final}/tmp_mask.vrt")
            if os.path.exists(f"{output_folder_final}/tmp_band.vrt"):
                os.remove(f"{output_folder_final}/tmp_band.vrt")


        self.images_masked = self.getImagesDict(products_path_list, "tif", new_naming = True)



    def applyFunction_pairs(self, pairs, function, output_folder, *kwargs):

        # Eliminate output folder if it exists
        if os.path.isdir(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        for pair in pairs:
            function(pair, output_folder, *kwargs)

    def applyFunction_list(self, lista, function, output_folder, *kwargs):

        # Erase output folde if it exists
        if os.paths.isdir(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        for l in lista:
            function(lista, output_folder, *kwargs)

        return 0


    def readModisFolder(self, folder_path, formato = "tif"):
        tiles = self.getTiles(folder_path)
        tiles_path = self.getTilesPath(folder_path, tiles)
        products_path = self.getProductsPath(tiles_path)
        df = self.getImagesDict(products_path, formato, new_naming = True)
        return df

    def reReadMaskedFolder(self, folder_path, formato = "tif"):
        self.images_masked = self.readModisFolder(folder_path, formato)
