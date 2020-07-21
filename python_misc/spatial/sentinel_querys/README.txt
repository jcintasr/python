Hay dos scripts que deben ser convertidos en función.

tilaread.py
==================================================================
Lee las tiles de las imágenes sentinel (previamente transformados desde kml a Geopackage) y los reproyecta a ETRS89.
Se crea una LISTA con valores únicos del código de las tiles que intersectan con la España peninsular (provincias_pen). 
Aquí quizás sería más lógico usar el contorno de la península en lugar de todas las provincias, para que sea más rápido.

La idea es convertirlo en una función capaz de devolver las tiles necesarias si se le introduce una capa.
NOTA: Hay problemas con la reproyección de la capa de tiles desde WGS84 a ETRS89. (estaba solventando esto)

download_sentinel.py
==================================================================
Este script es completamente funcional. Sin embargo, la idea es que las LISTA de tiles creada en el anterior script, sea leida por éste.
Por ello se debería cambiar la primera parte, en el que se lee un archivo que ya no está (estaba basado en una versión antigua de las tiles).

La idea es que cuando sea convertido en funcion requiera:
	- Capa: Preparado para capas en formato GPKG. (tilaread.py)
	- Lista de tiles: Se suministra una lista con los tiles a descargar. (tilaread.py o manualmente).
	- Fecha de inicio.
	- Fecha del final.
	- Porcentaje máximo de nubes.


