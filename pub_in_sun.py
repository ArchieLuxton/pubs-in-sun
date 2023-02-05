from utils import *
import numpy as np
import time
import pprint
import plotly.io as pio
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
np.set_printoptions(suppress=True)
pio.renderers.default = 'browser'
pp = pprint.PrettyPrinter(indent=4)

# Set the latitude and longitude of the center point
lat = 51.5074
lon = 0.1278
radius = 20

# Find all the buildings in the given search location
data = find_buildings(lat, lon, radius)

# Populate the dict 'buildings' with all the relevant data
buildings = populate_buildings(data)
pp.pprint(buildings)

# Convert the node numbers to latitude and longitude coordinates (slow!)
buildings = convert_node_to_coords(buildings, plot=True)

# Visualise the buildings that have been captured
#plotly_buildings(buildings)

# Populate the building_collection class with all the buildings that have been found
bN = [building(building_from_ind(buildings, i)[1]) for i in range(len(buildings))]
bN = building_collection(bN)

# Populate a DataFrame with all the relevant info from the buildings being plotted
tN = pd.concat([building_from_ind(buildings, i)[0] for i in range(len(buildings))])

# Calculate the sun's vector 
sunpath = calc_sunvector(tN, (2022, 12, 11, 14, 0, 0, 0))

# Costruct the array of rays
O, D = construct_rays(tN,
                      sunpath,
                      offset_lower=[0, 0],
                      offset_upper=[0, 0],
                      di=2,
                      dj=2,
                      z=0)

# Go through each ray and calculate intersects. 
# Populates the building_collection class with the right intersects 
#start = time.time()
#calculate_rays2(bN, O, D)
#end = time.time()
#print(f"Time taken for method 2: {end-start}")

start = time.time()
calculate_rays(bN, O, D)
end = time.time()
print(f"Time taken for method 1: {end-start}")
# Visualise the scene in 3D
plot_rays(bN, O, D, plot_ray=False, plot_shadow=True,s=50, figsize=(15,15), extent_offset=0)