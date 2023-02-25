from utils import latlon_to_m, calculate_bounding_box, geqc, leqc, spherical_to_cartesian
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd
from shapely.affinity import translate
import geopandas
import shapely
import math
from plots import plot_vector_and_raster
from convertbng.util import convert_bng, convert_lonlat
from utils import MinimumBoundingBox

def construct_building_walls(casting_buildings, floor_height=0):
    # Go through each resulting building and construct all the polys 
    ## TODO: CAN WE JUST CONSTRUCT A SINGLE POLY NORMAL TO THE DIRECTION OF THE SUN? OLD SCHOOL SPRITE STYLE. WOULD THIS SAVE ANY COMPUTATION?
    ## TODO: THINK ABOUT HOW TO IMPLEMENT FLOOR - PROBABLY NEED TO USE THE DSM INSTEAD OF CHM AT FIRST
    # TODO: WE ONLY NEED TO DO THIS ONCE FOR ALL BUILDINGS AND SAVE THE RESULT; LOAD FROM A DATABASE
    faces = []

    for i, geom in enumerate(casting_buildings["geometry"]):
        if isinstance(geom, shapely.geometry.polygon.Polygon):
            roof = np.array(geom.exterior.coords)[:,:2]
        elif isinstance(geom, shapely.geometry.multipolygon.MultiPolygon):
            roof = np.array(geom.convex_hull.exterior.coords)[:,:2]
        # Read the footprint geometry and convert to lat lon
        roof_latlon = np.array(convert_lonlat(roof[:,0], roof[:,1])).T
        roof_latlon[:, [1, 0]] = roof_latlon[:, [0, 1]]

        # Convert to metres
        roof_m = latlon_to_m(roof_latlon[:,0], roof_latlon[:,1]).T

        # Add the height of the roof
        roof_m = np.column_stack([roof_m, np.full(roof_m.shape[0], casting_buildings.iloc[i]["height"])])

        faces.append(roof_m)

        # Go through each line and create a new face between that and the floor
        for i in range(0, len(roof_m)-1):
            p1 = roof_m[i]          # E.g. Top left 
            p2 = roof_m[i+1]        # E.g. Top righ
            p3 = roof_m.copy()[i]   # E.g. Bottom left
            p3[2] = floor_height               # Set floor level to 0
            p4 = roof_m.copy()[i+1] # E.g. Bottom right
            p4[2] = floor_height               # Set floor level to 0
            faces.append(np.stack([p1, p2, p4, p3, p1]))
            
    return faces

def into_range(x, range_min, range_max):
    shiftedx = x - range_min
    delta = range_max - range_min
    return (((shiftedx % delta) + delta) % delta) + range_min

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


def get_casting_buildings(date, centre, shapes, bounding_box_root, cast_distance=5000, plot_bounding_box=False, plot_casting_buildings=False):
    """ OPTIMISATION: From the main array of buildings, return a dataframe of only buildings that can cast a shadow on our garden """ 
    azimuth, elevation = sunpos(date, centre, True)

    # Construct the second 
    x_offset = cast_distance * math.sin(math.radians(azimuth))
    y_offset = cast_distance * math.cos(math.radians(azimuth))

    # translate the polygon
    moved_poly = translate(bounding_box_root, xoff=x_offset, yoff=y_offset)

    projected_prism = construct_prism(bounding_box_root, moved_poly)

    if plot_bounding_box:
        shapes_ = shapes.copy()
        shapes_["geometry"].iloc[-1] = projected_prism
        plot_vector_and_raster(shapes_)

    casting_buildings = shapes[shapes.geometry.intersects(projected_prism)]
    
    if plot_casting_buildings:
        plot_vector_and_raster(casting_buildings)
        
            
    return casting_buildings


def construct_prism(root, projection):
    """
    Based on a root polygon (a square), and a projection polygon (also a square), 
    Project a prism that is a hull around the two squares
    """
    # Extract the coordinates of the vertices of the squares
    coords1 = root.exterior.coords
    coords2 = projection.exterior.coords

    # Find the centroids of the squares
    centroid1 = root.centroid
    centroid2 = projection.centroid

    # Calculate the bearing between the two squares
    dx = centroid2.x - centroid1.x
    dy = centroid2.y - centroid1.y
    bearing = 90 - np.degrees(np.arctan2(dy, dx))
    if bearing < 0:
        bearing += 360
    
    #rint(f"Bearing = {bearing}")
    # Determine which vertices to connect based on the bearing
    if bearing >= 0 and bearing <= 90:
        prism = Polygon([coords1[3], coords1[0], coords1[1], coords2[1], coords2[2], coords2[3], coords1[3]])
    elif bearing > 90 and bearing <= 180:
        prism = Polygon([coords1[0], coords1[3], coords1[2], coords2[2], coords2[1], coords2[0], coords1[0]])
    elif bearing > 180 and bearing <= 270:
        prism = Polygon([coords1[1], coords1[2], coords1[3], coords2[3], coords2[0], coords2[1], coords1[1]])
    else:
        prism = Polygon([coords1[0], coords1[1], coords1[2], coords2[2], coords2[3], coords2[0], coords1[0]])
    return prism



    
    
def generate_points(polygon, resolution):   
    """
    Generate some points within a Polygon object.
    At this point, not guaranteed to sit within the Polygon's bounds.
    Add a central point too so it never fails to find any
    """
    minx, miny, maxx, maxy = polygon.bounds
    x = np.arange(minx, maxx, 1/resolution)
    y = np.arange(miny, maxy, 1/resolution)
    XY = np.meshgrid(x, y)
    
    #centre = polygon.centroid
    
    return XY[0].ravel(), XY[1].ravel()

def get_points_in_poly(polygon, resolution, plot=False):
    """
    Generate lots of points in the region of a Polygon, then 
    filter out all the points that sit outside the perimiter. 
    Returns the geopandas dataframe of points that sit within the perimeter.
    """
    ## Calculate a grid of points that sit within this shape
    gdf_poly = gpd.GeoDataFrame(index=["myPoly"], geometry=[polygon])
    x,y = generate_points(polygon, resolution)
    df = pd.DataFrame()
    df['points'] = list(zip(x,y))
    df['points'] = df['points'].apply(Point)
    gdf_points = gpd.GeoDataFrame(df, geometry='points')
    Sjoin = gpd.tools.sjoin(gdf_points, gdf_poly, how='left')

    # Keep points in "myPoly"
    pnts_in_poly = gdf_points[Sjoin.index_right=='myPoly']

    # Plot result
    if plot:
        base = gdf_poly.boundary.plot(linewidth=1, edgecolor="black")
        pnts_in_poly.plot(ax=base, linewidth=1, color="red", markersize=8)
        plt.show()
    return pnts_in_poly

def intersection_first_check(P, face):
    bb = calculate_bounding_box(face)
    if geqc(P[0], bb[0][0]) & \
       leqc(P[0], bb[1][0]) & \
       geqc(P[1], bb[0][1]) & \
       leqc(P[1], bb[1][1]) & \
       geqc(P[2], bb[0][2]) & \
       leqc(P[2], bb[1][2]):
        res = is_point_in_polygon(face, P)   # ChatGPT
        return res
    else:
        return False




def point_in_2d_polygon(point, polygon):
    """
    winding number method

    """
    # Unpack the point into x and y coordinates
    x, y = point
    
    # Get the number of vertices in the polygon
    n = polygon.shape[0] - 1
    
    # Keep track of whether the point is inside the polygon
    inside = False
    
    # Get the first vertex of the polygon
    p1x, p1y = polygon[0]
    
    # Iterate over each vertex in the polygon
    for i in range(n+1):
        # Get the current vertex
        p2x, p2y = polygon[i % n]
        
        # Check if the point is in the bounding box of the edge
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                # Check if the point is to the left of the edge
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        # Calculate the intersection point of the edge with the horizontal line
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        # Toggle the inside status
                        inside = not inside
        # Set the current vertex as the previous vertex
        p1x, p1y = p2x, p2y
        
    # Return the inside status
    return inside


def is_point_in_polygon(polygon, point):
    """
    ChatGPT implementation. Faster.
    """
    # Check to see if it's a roof 
    if (polygon[:,2] == polygon[0,2]).all():
        return point_in_2d_polygon((point[0], point[1]), 
                                   polygon[:, :2])
    
        ## Alternative version, use Shapely. Maybe a tiny bit slower
        # return Polygon(polygon).contains(Point(point))
    else:
        polygon = np.array(polygon)
        # Calculate the normal vector of the plane
        A = polygon[0,:]
        B = polygon[1,:]
        C = polygon[2,:]
        normal = np.cross(B - A, C - A)
        normal /= np.linalg.norm(normal)
    
        # Calculate a change of basis matrix to map the polygon and point to 2D
        z_axis = normal
        x_axis = polygon[1,:] - polygon[0,:]
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        change_of_basis_matrix = np.array([x_axis, y_axis, z_axis]).T
    
        # Transform the polygon and point to 2D
        polygon_2d = np.dot(polygon, change_of_basis_matrix)
        point_2d = np.dot(point, change_of_basis_matrix)
    
        return point_in_2d_polygon((point_2d[0], point_2d[1]), 
                                   polygon_2d[:, :2])

        ## Alternative version: Use Shapely to check whether the point is within the polygon
        # polygon = Polygon(polygon_2d[:,:2])
        # point = Point(point_2d[0], point_2d[1])
        # return polygon.contains(point)
        
def calculate_intersection(faces, O, OD):
    intersect = False
    for i, face in enumerate(faces):
        ## TODO: CAN I VECTORISE THIS? CHATGPT SEEMS TO THINK SO, BUT CAN'T GET THE DIMENSIONS RIGHT
        # Define a plane with 3 coordinates 
        A = face[0]
        B = face[1]
        C = face[2]
        normal = np.cross(B - A, C - A)

        # calculate the distance between the ray and the plane
        numerator = np.dot(normal, A - O)
        denominator = np.dot(normal, OD)
        if denominator == 0:
            pass
            #print(f"Parallel to plane, no intersection")

        t = numerator / denominator
        P = O + t * OD
        intersect = intersection_first_check(P, face)

        if intersect:
            return [True, i]

    return [False]

def raycast_time(heatmap, faces, garden_samples, sunvector):
    # Define the ray 
    sun_dist = 10000
    for i in range(len(garden_samples)):
        D = garden_samples[i]      # D = destination (garden)
        O = D + sun_dist*sunvector # O = origin (sun)
        OD = D - O
    
        res = calculate_intersection(faces, O, OD)
    
        if len(res) == 2:
            # Intersection was found, shift the colliding building to the top of the queue so it might be found faster next time
            heatmap[i]  -= 1
            faces.insert(0, faces.pop(res[1]))
        elif len(res)==1:
            # No intersection found for this ray
            pass
        
    return heatmap, faces

# def raycast_day(heatmap, shapes, garden_latlon, garden_samples, date, bounding_box_root, floor_height, minute_steps=2, plot_faces=False):
#     centre = (garden_latlon[:,0].mean(), garden_latlon[:,1].mean())
#     print(f"Minute steps = {minute_steps}")
#     for hour in range(0,24):
#         for minute in minute_steps:
#             print(f"{hour}h {minute}m...")
#             datetime  = (date[0], date[1], date[2], hour, minute, 0, 0)
#             azimuth, elevation = sunpos(datetime, centre, True)
#             if elevation >= 0:
#                 casting_buildings = get_casting_buildings(datetime, centre, shapes, bounding_box_root, cast_distance=5000, plot_bounding_box=False, plot_casting_buildings=False)
#                 faces = construct_building_walls(casting_buildings, floor_height=floor_height)
#                 if plot_faces:
#                     plot_faces(faces)
#                 sunvector = spherical_to_cartesian(azimuth, elevation)
#                 heatmap, faces = raycast_time(heatmap, faces, garden_samples, sunvector)
#             else:
#                 heatmap -= 1
            
#     return heatmap

def calculate_bounding_box_root(garden_latlon, o=0, method=1):
    if method==1:
        ## Use minimum bounding box to get the tightest box possible
        # Need to check this, as it doesn't seem to be working properly
        garden_bng = np.array(convert_bng(garden_latlon[:,1], garden_latlon[:,0])).T
        garden_poly = tuple([tuple(t) for t in garden_bng])
        bounding_box_root = Polygon(np.array(list(MinimumBoundingBox(garden_poly).corner_points)))
    elif method==2:
        ## Create box aligned with cartesian coordinates. Guaranteed to be bigger.
        ## TODO: Are these both giving the correct result? Should only be an optimisation
        garden_bng = np.array(convert_bng(garden_latlon[:,1], garden_latlon[:,0])).T
        garden_poly = Polygon([tuple(t) for t in garden_bng])
        minx, miny, maxx, maxy = garden_poly.boundary.bounds
        bounding_box_root = Polygon([(minx-o, miny-o), (maxx+o, miny-o), (maxx+o, maxy+o), (minx-o, maxy+o)])
    return bounding_box_root

# def run_raycasting(garden_samples, garden_latlon, shapes, floor_height, date=(2022, 2, 8), minute_resolution=15):
#     print("running!")
    
#     minute_steps = np.arange(0, 60, minute_resolution).astype(int)#np.linspace(0,60,4).astype(int)[:-1]##
    
#     heatmap = np.full(len(garden_samples), 24*len(minute_steps)).astype(float)
#     heatmap = raycast_day(heatmap=heatmap,
#                           shapes=shapes, 
#                           garden_latlon = garden_latlon,
#                           garden_samples=garden_samples,
#                           date=date,
#                           bounding_box_root=calculate_bounding_box_root(garden_latlon, o=0),
#                           floor_height=floor_height,
#                           minute_steps=minute_steps,
#                           plot_faces=False) / len(minute_steps)
    
#     return heatmap