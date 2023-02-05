import requests
import math
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import pprint
import xml.etree.ElementTree as ET
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pdb
from sympy import Point3D, Plane, Line3D
from scipy.spatial import ConvexHull
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union

np.set_printoptions(suppress=True)
pp = pprint.PrettyPrinter(indent=4)


def find_buildings(lat, lon, radius):
    # Convert the radius to latitude and longitude coordinates
    # using the haversine formula
    lat_range = radius / 111320.0
    lon_range = radius / (111320.0 * math.cos(lat))
    
    # Construct the Overpass query
    query = f'''
    [out:json][timeout:25];
    (
      node["building"]({lat-lat_range},{lon-lon_range},{lat+lat_range},{lon+lon_range});
      way["building"]({lat-lat_range},{lon-lon_range},{lat+lat_range},{lon+lon_range});
      relation["building"]({lat-lat_range},{lon-lon_range},{lat+lat_range},{lon+lon_range});
    );
    out body;
    >;
    out skel qt;
    '''
    
    # Send the request to the Overpass API
    response = requests.get(
        'https://overpass-api.de/api/interpreter', 
        params={'data': query}
    )
    
    print(response)
    
    # Load the response data into a GeoJSON object
    data = response.json()
    
    return data
    
    
def get_params(entry):
    """ Boring function that just tries to get all available info and doesn't trip up if it can't find it """
    try:
        city = entry["tags"]["addr:city"]
    except:
        city = "N/A"
    try:
        housenumber = entry["tags"]["addr:housenumber"]
    except:
        housenumber = "N/A"
    try:
        street = entry["tags"]["addr:street"]
    except:
        street = "N/A"
    try:
        levels = entry["tags"]["building:levels"]
    except:
        levels = -999
    return city, housenumber, street, levels


def populate_buildings(data):
    buildings = {}
    
    # First of all, go through all the "ways" (i.e. buildings) and populate a dict with all available info
    for entry in data["elements"]:
        if entry["type"] == "way":
            id = entry["id"]
            nodes = entry["nodes"]
            city, housenumber, street, levels = get_params(entry)
            buildings[id] = {"nodes":nodes, 
                             "levels":float(levels),
                             "address":(f"{housenumber}, {street}, {city}")}
            
    # Sometimes, location and height data is only included in "relations"
    # "relations" are groups of buildings that all share common attributes.
    # Loop through these separately once we've done all we can with the "ways" above.
    for entry in data["elements"]:
        if entry["type"] == "relation":
            city, housenumber, street, levels = get_params(entry)
            for m in entry["members"]:
                buildings[m["ref"]]["address"] = (f"Relation {entry['id']}: {housenumber}, {street}, {city}")
                buildings[m["ref"]]["levels"] = float(levels)
                
    print(f"Processed {len(buildings)} buildings")
    
    return buildings


def get_nodemap(buildings, b):
    """
    Ask the API for all nodes for a given way as coordinates.
    Then return a map that we can use to convert nodes to coordinates. 
    """
    node_map = {}
    
    query = f"""[out:json][timeout:25];
    way({b});
    (._;>>;);
    out;"""

    response = requests.get(
        'https://overpass-api.de/api/interpreter', 
        params={'data': query}
    )

    # Load the response data into a GeoJSON object
    data = response.json()

    for element in data["elements"]:
        if element["type"] == "node":
            node_map[element["id"]] = (element["lat"], element["lon"])
    
    return node_map

def latlon_from_way(buildings, b):
    """
    For a given way (i.e. building), ask the API to return all nodes as lat/lon coordinates.
    Then, map each node to its respective lat/lon coordinates and then convert the node list 
    (which is ordered and includes a final node that loops back to origin) to coordinates.
    """
    node_map = get_nodemap(buildings, b)
    latlon = list(map(node_map.get, buildings[b]["nodes"]))
    return latlon

def latlon_from_nodes(buildings, b):
    """ Go through every node and call API to convert to lat and lon """
    
    base_url = 'https://www.openstreetmap.org/api/0.6/'
    
    nodes = buildings[b]["nodes"]
    latlon = [0]*len(nodes)
    for i, node in enumerate(nodes):
        node_response = requests.get(base_url + 'node/' + str(node))
        node_data = node_response.content

        # Parse the XML data
        node = ET.fromstring(node_data)[0]
        lat = node.attrib['lat']
        lon = node.attrib['lon']

        latlon[i] = (float(lat), float(lon))
    return latlon


def convert_node_to_coords(buildings, plot=True):
    if plot:
        # Set the projection for the map
        projection = ccrs.PlateCarree()
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=projection))
    
    #timings = np.zeros((len(buildings),3))
    
    for i, b in enumerate(buildings):
        print(f'Processing building {i}/{len(buildings)}', end='\r')
        
        ## TODO: Why does this give JSONDecodeError in get_nodemap? 
        #if len(buildings[b]['nodes']) > 9:
        #    # This is faster if the number of nodes is high
        #    latlon = latlon_from_way(buildings, b)
        #else:
        #    # This is faster if number of nodes is small
        #    latlon = latlon_from_nodes(buildings, b)
        
        latlon = latlon_from_nodes(buildings, b)
        
        # Split the array into separate arrays for the latitude and longitude coordinates
        buildings[b]["nodes_ll"] = latlon
    
        # Draw the building
        lats = [latlon[i][1] for i in range(len(latlon))]
        lons = [latlon[i][0] for i in range(len(latlon))]
        
        if plot:
            ax.plot(lats, lons, color='red', linewidth=2,transform=ccrs.PlateCarree())
    
    if plot:
        # Show the plot
        plt.show()
        
    return buildings

def plotly_buildings(buildings, shadows=None, zoom=18):
    traces = []
    colormap = {-999:"black",
               1:"yellow",
               2:"orange",
               3:"red",
               4:"green"}
    
    if shadows is not None:
        for s in shadows:
            traces.append(go.Scattermapbox(
                name=f"Way: XX",
                fill = "toself",
                lon = s["lon"],
                lat = s["lat"],
                marker = { 'size': 0, 'color': "grey"},
                hovertemplate =f"""
                <b>Address</b>:XX
                <br>
                <b>Levels</b>:YYY"""))
        
    for i, b in enumerate(buildings):
        t = pd.DataFrame(buildings[b]["nodes_ll"], columns=["lat", "lon"])
        traces.append(go.Scattermapbox(
            name=f"Way: {b}",
            fill = "toself",
            lon = t["lon"], lat = t["lat"],
            marker = { 'size': 1, 'color': colormap[buildings[b]["levels"]] },
        hovertemplate =f"""
        <b>Address</b>:{buildings[b]["address"]}
        <br>
        <b>Levels</b>:{buildings[b]["levels"]}"""))
        

        
    fig = go.Figure(data=traces, layout=dict())
    fig.update_layout(
        mapbox = {
            'style': "stamen-terrain",
            'center': {'lon': t["lon"].iloc[0],
                       'lat': t["lat"].iloc[0]},
            'zoom': zoom},
        showlegend = False)
    
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
    
class building():
    def __init__(self, roof_verts):
        self.faces = []
        self.roof_verts = roof_verts.copy()
        self.N = len(roof_verts)-1   # Number of unique vertices (and faces)
        self.add_faces()
        self.collision_coords = []     # The coordinates on each face where a ray hits it
        self.shadow_coords = []        # The coordinates where each face casts a shadow (if any)
        self.origin_coords = []
        self.sunny_coords = []
        self.intersects = []
        self.midpoints = []
    def add_faces(self):
        self.faces.append(self.roof_verts)
        for i in range(self.N):
            p1 = self.roof_verts[i]          # E.g. Top left 
            p2 = self.roof_verts[i+1]        # E.g. Top righ
            p3 = self.roof_verts.copy()[i]   # E.g. Bottom left
            p3[2] = 0                        # Set floor level to 0
            p4 = self.roof_verts.copy()[i+1] # E.g. Bottom right
            p4[2] = 0                        # Set floor level to 0
            self.faces.append(np.stack([p1, p2, p4, p3, p1]))
    def add_face(self, vertices):
        self.faces.append(vertices)
        
class building_collection():
    def __init__(self, buildings):
        self.buildings = buildings
        self.collision_coords = []     # The coordinates on each face where a ray hits it
        self.faces = []
        self.shadow_coords = []        # The coordinates where each face casts a shadow (if any)
        self.origin_coords = []
        self.sunny_coords = []
        self.intersects = []
        self.compile_faces()
        self.unique_faces = None
    def compile_faces(self):
        for building in self.buildings:
            for face in building.faces:
                self.faces.append(face)

def sunpos(when, location, refraction):
    # Thanks https://levelup.gitconnected.com/python-sun-position-for-solar-energy-and-research-7a4ead801777
    # Extract the passed data
    year, month, day, hour, minute, second, timezone = when
    latitude, longitude = location
    # Math typing shortcuts
    rad, deg = math.radians, math.degrees
    sin, cos, tan = math.sin, math.cos, math.tan
    asin, atan2 = math.asin, math.atan2
    # Convert latitude and longitude to radians
    rlat = rad(latitude)
    rlon = rad(longitude)
    # Decimal hour of the day at Greenwich
    greenwichtime = hour - timezone + minute / 60 + second / 3600
    # Days from J2000, accurate from 1901 to 2099
    daynum = (
        367 * year
        - 7 * (year + (month + 9) // 12) // 4
        + 275 * month // 9
        + day
        - 730531.5
        + greenwichtime / 24
    )
    # Mean longitude of the sun
    mean_long = daynum * 0.01720279239 + 4.894967873
    # Mean anomaly of the Sun
    mean_anom = daynum * 0.01720197034 + 6.240040768
    # Ecliptic longitude of the sun
    eclip_long = (
        mean_long
        + 0.03342305518 * sin(mean_anom)
        + 0.0003490658504 * sin(2 * mean_anom)
    )
    # Obliquity of the ecliptic
    obliquity = 0.4090877234 - 0.000000006981317008 * daynum
    # Right ascension of the sun
    rasc = atan2(cos(obliquity) * sin(eclip_long), cos(eclip_long))
    # Declination of the sun
    decl = asin(sin(obliquity) * sin(eclip_long))
    # Local sidereal time
    sidereal = 4.894961213 + 6.300388099 * daynum + rlon
    # Hour angle of the sun
    hour_ang = sidereal - rasc
    # Local elevation of the sun
    elevation = asin(sin(decl) * sin(rlat) + cos(decl) * cos(rlat) * cos(hour_ang))
    # Local azimuth of the sun
    azimuth = atan2(
        -cos(decl) * cos(rlat) * sin(hour_ang),
        sin(decl) - sin(rlat) * sin(elevation),
    )
    # Convert azimuth and elevation to degrees
    azimuth = into_range(deg(azimuth), 0, 360)
    elevation = into_range(deg(elevation), -180, 180)
    # Refraction correction (optional)
    if refraction:
        targ = rad((elevation + (10.3 / (elevation + 5.11))))
        elevation += (1.02 / tan(targ)) / 60
    # Return azimuth and elevation in degrees
    return (round(azimuth, 2), round(elevation, 2))

def into_range(x, range_min, range_max):
    shiftedx = x - range_min
    delta = range_max - range_min
    return (((shiftedx % delta) + delta) % delta) + range_min

def calc_sunvector(building_df, date):
    lat_av = building_df["lat (deg)"].mean()
    lon_av = building_df["lon (deg)"].mean()
    azimuth, elevation = sunpos(date, (lat_av, lon_av), True)
    
    a = 1  # Metres directly north
    b = a/np.cos(np.radians(azimuth))
    c = b/np.cos(np.radians(elevation))
    d = np.sqrt(b**2 - a**2)
    e = np.sqrt(c**2 - b**2)

    sunpath = np.array([d, a, e]) # Directly north. i, j, k, where i is east, j is north, k is up

    return sunpath

def building_from_ind(buildings, ind):
    b = list(buildings.keys())[ind]
    t = pd.DataFrame(buildings[b]["nodes_ll"], columns=["lat (m)", "lon (m)"])*111139
    t["lat (deg)"] = [buildings[b]["nodes_ll"][i][0] for i in range(len(buildings[b]["nodes_ll"]))]
    t["lon (deg)"] = [buildings[b]["nodes_ll"][i][1] for i in range(len(buildings[b]["nodes_ll"]))]
    t["height (m)"] = buildings[b]["levels"]*3
    verts = np.array([[t["lat (m)"].iloc[i], t["lon (m)"].iloc[i], t["height (m)"].iloc[i]] for i in range(len(t))])
    poly = Polygon(verts)
    return t, verts, poly

def construct_rays(t, sunpath, offset_lower=[20,20], offset_upper=[20,20], di=5, dj=5, z=0):
    # Height off the ground of the destination of the rays
    z = 0

    # Centrepoint of the ray array 
    centre = [t["lat (m)"].mean(),t["lon (m)"].mean()]

    # Construct the min and max latitude and longitudes (i.e. bottom left and top right corner of the destination of the array)
    x0y0 = [t["lat (m)"].min()-offset_lower[0], t["lon (m)"].min()-offset_upper[1]]  # Min x and y coordinates of the ray destination
    x1y1 = [t["lat (m)"].max()+offset_upper[0], t["lon (m)"].max()+offset_upper[1]]  # Max y and y coordinates of the ray destination

    # Construct the meshgrid (i.e. array) of the ray destination (i.e. at ground level)
    X = np.arange(x0y0[0],x1y1[0],di)
    Y = np.arange(x0y0[1],x1y1[1],dj)
    XY = np.meshgrid(X,Y)
    XY = np.array([np.array(XY)[0].flatten(),np.array(XY)[1].flatten()]).T
    Z = np.full((len(XY)),z)
    D = np.column_stack((XY,Z))       # Where the sun lands

    # Construct the origin array for the rays
    # Ray is 1000m in the direction of D from O
    O = D + 1000*sunpath              # Origin - i.e. where is the sun in the sky?

    print(f"Number of rays: {len(D)}")
    
    return O, D

def find_intersect_sympy(O, D, verts):
    #plane Points
    a1 = Point3D(tuple(verts[0]))
    a2 = Point3D(tuple(verts[1]))
    a3 = Point3D(tuple(verts[2]))
    #line Points
    p0 = Point3D (tuple(O)) #point in line
    v0 = D-O #line direction as vector
    
    #create plane and line
    plane = Plane(a1,a2,a3)
    
    line = Line3D(p0,direction_ratio=v0)

    intr = plane.intersection(line)
    
    I = np.array(intr[0],dtype=float)
    return I

def find_intersect(O_full, D_full, verts):
    assert len(O_full) == len(D_full), "Please ensure O and D are the right shape!"
    
    def do_calcs(O, D, verts):
        # Compute vectors AB and AC
        A = verts[0]
        B = verts[1]
        C = verts[2]
        D = D-O
        AB = B - A
        AC = C - A

        # Compute the normal vector of the plane
        N = np.cross(AB, AC)

        # Compute the scalar d
        d = -np.dot(N, A)

        # Compute the numerator of t
        t_num = np.dot(N, A - O)

        # Compute the denominator of t
        t_den = np.dot(N, D)

        # If the denominator is zero, the vector is parallel to the plane and does not intersect it
        if abs(t_den) < 1e-6:
            pdb.set_trace()
            return None

        # Compute the value of t
        t = t_num / t_den

        # If t is negative, the vector intersects the plane behind O
        #if t < 0:
        #    pdb.set_trace()
        #    return None

        # Compute the intersection point
        P = O + t * D

        return P

    if O_full.ndim == 2:
        I = []
        for i in range(O_full.shape[0]):
            O = O_full[i]
            D = D_full[i]
            intrsct_point = do_calcs(O, D, verts)
            if intrsct_point is not None:
                intrsct_point = intrsct_point.reshape(-1,1)
            I.append(intrsct_point)
            return I
    else:
        res = do_calcs(O_full, D_full, verts)
        if res is not None:
            res = res.reshape(-1,1)
        return res

def check_intersect_all_rays(verts, I_full):
    if I_full.ndim == 2:
        res = []
        for i in range(I_full.shape[0]):
            I = I_full[i]         
            res.append(check_intersect(verts, I))
    else:
        res = check_intersect(verts, I_full)
        
    return res 

def check_intersect(verts, I):
    # Define two vectors on our plane to form a new basis.
    # These are guaranteed to be linearly independent, but not orthogonal
    v1 = verts[1]-verts[0]
    v2 = verts[2]-verts[1]
    v3 = np.cross(v1, v2)
    
    A = np.array([v1, v2, v3])
    verts_ = np.array([np.linalg.solve(A, v) for v in verts])
    I_ = np.linalg.solve(A, I.astype(float))
    
    # Remove superfolous dimension.. Whichever that may be.
    # How does this scale to non-orthogonal planes? Hopefully not much modification needed
    # But either way, will need much higher resolution data than this to use polys properly
    for i in range(verts.shape[1]):
        if (verts_[:,i]==verts_[0,i]).all() == True:
            verts_ = np.delete(verts_, i,1)
    
    point = Point(I_)
    polygon = Polygon([(verts_[i][0], verts_[i][1]) for i in range(verts_.shape[0])])
    
    return polygon.contains(point)

def check_intersect3(verts, I_full):
    def do_calcs(verts, I):
        # Check to see if the shape is 2D and will break convex_hull
        redundant_dim = np.where(np.all(verts == verts[0,:], axis=0)==True)[0]
        if len(redundant_dim) > 0:
            # Then one component can be totally removed from this vert - otherwise will break convex hull 
            verts = np.delete(verts, redundant_dim, axis=1)

def calculate_rays(building, O_full, D_full):   
    def check_ray(O, D, building):
        for i, face in enumerate(building.faces):
            
            
            
            I = find_intersect(O, D, face).squeeze()
            maxim = face.max(axis=0)
            minim = face.min(axis=0)
            
            # Optimisation - Only check the ones that are likely to be intersects 
            if (I[2] >= 0) and \
               (I[2] <= maxim[2]) and \
               (I[0] >= minim[0]) and \
               (I[0] <= maxim[0]) and \
               (I[1] >= minim[1]) and \
               (I[1] <= maxim[1]):
                   
                    # Incredibly, this step doesn't really add any time at all  
                    intersect = np.array(check_intersect_all_rays(face, I))
                    if intersect is False: 
                        print(f"I thought it was going to intersect but it didn't!")
            else:
                intersect = False
            if intersect:
                # If intersect, save the data to the object and return - i.e. move to the next ray

                building.collision_coords.append(I)
                building.shadow_coords.append(D)
                building.origin_coords.append(O)
                building.sunny_coords.append([])
                return True

        building.collision_coords.append([])
        building.shadow_coords.append([])
        building.origin_coords.append(O)
        building.sunny_coords.append(D)
        return True
    
    building.collision_coords = []
    building.shadow_coords = []
    building.origin_coords = []
    building.sunny_coords = []   
    
    # Loop through each ray and see if it hits anything    
    for i in range(len(O_full)):
        check_ray(O_full[i], D_full[i], building)
        
        if i%500 == 0:
            print(f"Ray {i}/{len(O_full)}", end="\r")
  

def plot_rays(building,
              O,
              D, 
              plot_ray=False,
              plot_shadow=True,
              s=50,
              extent_offset=20,
              figsize=(15,15)):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    for i, face in enumerate(building.faces):
        vertices = [[tuple(v) for v in face]]
        poly = Poly3DCollection(vertices, alpha=0.8)
        ax.add_collection3d(poly)

    for i in range(len(O)):
        if building.shadow_coords[i] != []:
            if plot_ray:
                ax.plot((building.origin_coords[i][0], building.collision_coords[i][0]),
                        (building.origin_coords[i][1], building.collision_coords[i][1]),
                        (building.origin_coords[i][2], building.collision_coords[i][2]), color="red", alpha=0.5, linewidth=0.5)
            if plot_shadow:
                ax.scatter(building.shadow_coords[i][0],
                           building.shadow_coords[i][1],
                           building.shadow_coords[i][2], marker="o", c="black", s=s)
        else:
            if plot_ray:
                ax.plot((building.origin_coords[i][0], building.sunny_coords[i][0]),
                        (building.origin_coords[i][1], building.sunny_coords[i][1]),
                        (building.origin_coords[i][2], building.sunny_coords[i][2]), color="orange", alpha=0.5, linewidth=0.5)
            if plot_shadow:
                ax.scatter(building.sunny_coords[i][0],
                           building.sunny_coords[i][1],
                           building.sunny_coords[i][2], marker="o", c="yellow", s=s)


    #roof_verts = building.faces[0]
    mins = np.concatenate(np.array(building.faces, dtype=object)).min(axis=0)
    maxs = np.concatenate(np.array(building.faces, dtype=object)).max(axis=0)
    
    ax.set_xlim(mins[0]-extent_offset,maxs[0]+extent_offset)
    ax.set_ylim(mins[1]-extent_offset,maxs[1]+extent_offset)
    ax.set_zlim(mins[2]-extent_offset,maxs[2]+extent_offset) 
    
    
def combine_shadow_polys(b):
    # Create Polygons for each face that's projected onto the floor
    polys = [Polygon([tuple(point) for point in s]) for s in b.shadow_polys]

    # Merge the Polygons together
    mergedPolys = unary_union(polys)

    # Extract the outline of the new Polygon
    xx, yy = mergedPolys.exterior.coords.xy
    xx = xx.tolist()
    yy = yy.tolist()

    #gpd.GeoSeries([mergedPolys]).boundary.plot()
    #plt.show()
    #mergedPolys = unary_union(polys)
    
    return np.array([xx, yy]).reshape(2,-1).T
    
def plot_rays_vertices(building,
              s=50,
              extent_offset=20,
              figsize=(15,15)):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    def plot_polys(building):
        shadow_poly = Poly3DCollection([[tuple(point) for point in building.overall_shadow_poly]], alpha=0.2, color="black")
        ax.add_collection3d(shadow_poly)

        for i, face in enumerate(building.faces):
            vertices = [[tuple(v) for v in face]]
            poly = Poly3DCollection(vertices, alpha=0.8)
            ax.add_collection3d(poly)

    if isinstance(building, list):
        for b in building:
            plot_polys(b)
        all_faces = np.concatenate(np.array([np.concatenate(np.array(b.faces), dtype=object) for b in building]))
    else:
        plot_polys(building)
        all_faces = np.concatenate(np.array(b.faces), dtype=object)
    
    mins = all_faces.min(axis=0)
    maxs = all_faces.max(axis=0)
    
    ax.set_xlim(mins[0]-extent_offset,maxs[0]+extent_offset)
    ax.set_ylim(mins[1]-extent_offset,maxs[1]+extent_offset)
    ax.set_zlim(mins[2]-extent_offset,maxs[2]+extent_offset) 
    
    
def calculate_shadows(bN, sunpath):
    for b in bN:
        b.collision_coords = []
        b.shadow_coords = []
        b.origin_coords = []
        b.sunny_coords = []  
        b.shadow_con_hull = []
        b.shadow_polys = []
        b.overall_shadow_poly = []

        for face in b.faces:
            shadow_coords = []
            if (face[:,2] >= 0).all():
                for i, O in enumerate(face):
                    ground_level = 0
                    L = (ground_level-O[2])/sunpath[2]  # How many sunpaths do we need to move before we hit the ground?
                    D = O + L*sunpath
                    b.collision_coords.append(O)
                    shadow_coords.append(D)
                    b.origin_coords.append(O)
                    b.sunny_coords.append([])

                shadow = np.array(shadow_coords)
                b.shadow_polys.append(shadow)

                b.overall_shadow_poly = combine_shadow_polys(b)

                removed_coord = np.full(len(b.overall_shadow_poly),0)
                b.overall_shadow_poly = np.column_stack((b.overall_shadow_poly, removed_coord))
