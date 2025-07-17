import pandas as pd
import numpy as np

np.random.seed(42)  # reproducible

n = 5000  # number of trips

# Simulate trips spread across a bounding box (e.g., San Francisco)
df = pd.DataFrame({
    'origin_lat': np.random.uniform(37.6, 37.85, size=n),       # latitudes in SF
    'origin_lon': np.random.uniform(-122.55, -122.35, size=n),  # longitudes in SF
    'trips': np.random.poisson(3, size=n),                      # trips per event
    'price_usd': np.random.uniform(8, 30, size=n)               # trip price in USD
})

print(df.head())
print(f"Simulated trips: {len(df)}")

from h3 import h3
from shapely.geometry import Polygon
import geopandas as gpd
import matplotlib.pyplot as plt

H3_RES = 7  # adjust resolution as desired

# Add H3 cell index
df['h3_origin'] = df.apply(
    lambda row: h3.latlng_to_cell(row['origin_lat'], row['origin_lon'], H3_RES),
    axis=1
)

print(f"Unique H3 cells: {df['h3_origin'].nunique()}")

# Aggregate trips & prices per H3 cell
agg = df.groupby('h3_origin').agg(
    trips_sum=('trips', 'sum'),
    price_mean=('price_usd', 'mean')
).reset_index()

print(f"Aggregated rows: {len(agg)}")

# Convert H3 cell index to polygons
def h3_to_polygon(h3_index):
    boundary = h3.cell_to_boundary(h3_index)
    boundary_lonlat = [(lon, lat) for lat, lon in boundary]
    return Polygon(boundary_lonlat)

agg['geometry'] = agg['h3_origin'].apply(h3_to_polygon)

# Create GeoDataFrame properly
gdf = gpd.GeoDataFrame(agg, geometry="geometry")
gdf.set_crs(epsg=4326, inplace=True)

print(gdf.head())

# Save to GeoJSON
gdf.to_file("h3_cells.geojson", driver="GeoJSON")
print("✅ GeoJSON saved: h3_cells.geojson")

# Quick map
gdf.plot(column='trips_sum', cmap='OrRd', legend=True)
plt.title(f"Trips Aggregated by H3 Cell (res {H3_RES})")
plt.show()


