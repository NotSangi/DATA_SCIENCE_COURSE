import geopandas as gpd
import geodatasets
import matplotlib.pyplot as plt

#STATIC MAP
path = geodatasets.get_path('geoda.chicago_commpop')
chicago = gpd.read_file(path)
print(chicago.head())

#Shows the map of chicago colored by population
chicago.plot(column='POP2010', legend=True,legend_kwds={'label': 'Population in 2010', 'orientation': 'horizontal'})
plt.show()
