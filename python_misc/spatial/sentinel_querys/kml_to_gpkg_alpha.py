import ogr, osr
import os

gpkg_driver = ogr.GetDriverByName("GPKG")
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)

if os.path.exists("sentinel_tiles.gpkg"):
    gpkg_driver.DeleteDataSource("sentinel_tiles.gpkg")
dsout = gpkg_driver.CreateDataSource("sentinel_tiles.gpkg")
layerout = dsout.CreateLayer("sentinel_tiles", srs, geom_type=ogr.wkbMultiPolygon)

nameField = ogr.FieldDefn("Tiles", ogr.OFTString)
layerout.CreateField(nameField)

dsin = ogr.Open("S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml")
layerin = dsin.GetLayer()

for feature in layerin:
    feature_outDefn = layerout.GetLayerDefn()
    feature_out = ogr.Feature(feature_outDefn)

    field = feature.GetField(0)
    collection = feature.GetGeometryRef()
    for geometry in collection:
        if geometry.GetGeometryName() == "POLYGON":
            geom = geometry

        feature_out.SetGeometry(geom.Clone())
        feature_out.SetField("Tiles", field)
        layerout.CreateFeature(feature_out)
    feature_out = None
#             print(geom)

dsout.FlushCache()
del dsin
del dsout