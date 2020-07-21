import osr
import ogr
import os

# Read the tiles
tiles = "tile_grid/tiles_sentinel2.gpkg"
driver = ogr.GetDriverByName("GPKG")
ds = driver.Open(tiles)
tiles_layer = ds.GetLayer()

#### Only for test if it gets the right tiles
# tilesDefinition = tiles_layer.GetLayerDefn()
# tilesFieldDefn = tilesDefinition.GetFieldDefn(0)

# driverout = ogr.GetDriverByName("ESRI Shapefile")
# outsrs = osr.SpatialReference()
# outsrs.ImportFromEPSG(4326)
#
# outgpkg = "tile_grid/test.shp"
# if os.path.exists(outgpkg):
#     driverout.DeleteDataSource(outgpkg)
#
# outDataSet = driverout.CreateDataSource(outgpkg)
# outlayer = outDataSet.CreateLayer("test", geom_type = ogr.wkbPolygon)
# outlayer.CreateField(tilesFieldDefn)
# outDefn = outlayer.GetLayerDefn()

# Read the spain provinces
provincias_spain = "tile_grid/provincias_pen.gpkg"
gpkg_driver = ogr.GetDriverByName("GPKG")
ds_pen = gpkg_driver.Open(provincias_spain)
pen_layer = ds_pen.GetLayer()

# For reproject into ETRS89 (just to be sure)
target_srs = osr.SpatialReference()
target_srs.ImportFromEPSG(4326)

listofgeometries = []
tilelist = []
for provincia in pen_layer:
    geom = provincia.GetGeometryRef()

    # Avoiding topologycal errors. It doesn't work at all.
    provincia_geom = geom.Buffer(0)
    geom = 0

    # Reprojecting into ETRS89
    provincia_srs = provincia_geom.GetSpatialReference()
    transformation_prov = osr.CoordinateTransformation(provincia_srs, target_srs)
    provincia_geom.Transform(transformation_prov)
    provincia_geom.ExportToWkt()

    for tiles_feature in tiles_layer:

        # Getting geometry
        geom = tiles_feature.GetGeometryRef()

        # Avoiding topological errors
        tiles_geom = geom.Buffer(0)
        geom = 0

        # Reprojecting into ETRS89
        tiles_srs = tiles_geom.GetSpatialReference()
        transformation_tiles = osr.CoordinateTransformation(tiles_srs, target_srs)
        tiles_geom.Transform(transformation_tiles)
        tiles_geom.ExportToWkt()

        # print(tiles_geom)

        # Getting tiles interescting with the provinces
        if provincia_geom.Intersects(tiles_geom):

            # Making sure only unique tiles are selected.
            if tiles_geom not in listofgeometries:

                listofgeometries.append(tiles_geom)

                # Appending tile indexes
                tilelist.append(tiles_feature.GetField(0))

                # outFeature = ogr.Feature(outDefn)
                # outFeature.SetGeometry(tiles_geom)
                # outFeature.SetField(outDefn.GetFieldDefn(0).GetNameRef(), tiles_feature.GetField(0))
                # outlayer.CreateFeature(outFeature)
                # outFeature = None

print(tilelist)

# outDataset = None
# ds_pen = None
# ds = None