import sentinelsat
import collections

# https://sentinelsat.readthedocs.io/en/stable/api.html

api = sentinelsat.SentinelAPI(user = "morfeo",
                              password = "Thematrix",
                              api_url = 'https://scihub.copernicus.eu/dhus',
                              show_progressbars = True)

# Reading all the tiles
tiles = []
tiletxt = open("SpainPen_tiles.txt", mode = "r")
tilelines = tiletxt.read().splitlines()
tiletxt.close()
for t in tilelines:
    tiles.append(t)

# tiles = ["29SPB"]

#specify start and end date (here we use May 2017)
dateStart = "20180101"
dateEnd = "20181231"

query_kwargs = {
    'platformname': 'Sentinel-2',
    'producttype': ('S2MS2Ap', 'S2MSI1C'),
    'cloudcoverpercentage': (0, 15),
    'date': (dateStart, dateEnd)
}

products = collections.OrderedDict()
for tile in tiles:
    kw = query_kwargs.copy()
    kw['tileid'] = tile
    pp = api.query(**kw)
    products.update(pp)

# Define folder where download into
output_folder = "sentinel_images"
api.download_all(products, output_folder)