from utils import latlon_to_m
from shapely.geometry import Point, Polygon
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import geopandas as gpd
import matplotlib.pyplot as plt
from utils import latlon_to_m, spherical_to_cartesian, m_to_lat, m_to_lon
from convertbng.util import convert_bng, convert_lonlat
from plots import plot_scatter_heatmap_plotly, plot_faces, plot_projected_shadows_plotly, plot_vector_and_raster
from raycaster import sunpos, construct_prism, get_casting_buildings, construct_building_walls, calculate_bounding_box_root, raycast_time
import time 
import math
from shapely.affinity import translate
#run_raycasting, raycast_day, 

import plotly.io as pio
pio.renderers.default='browser'

class sunnygarden():
    def __init__(self, border_ll):
        self.border_ll = border_ll
        self.mean_lat = np.mean(border_ll[:,0])
        self.mean_lon = np.mean(border_ll[:,1])
        self.mean_bng = convert_bng(self.mean_lon, self.mean_lat)
        self.border_m = latlon_to_m(border_ll[:,0], border_ll[:,1]).T
        self.border_m_poly = Polygon([tuple(t) for t in self.border_m])
        self.sunlatlon = None
        self.casting_buildings = None
        
    def fetch_buildings(self, shape_path, search_radius):
        leftextent = self.mean_bng[0][0]-search_radius
        rightextent = self.mean_bng[0][0]+search_radius
        bottomextent = self.mean_bng[1][0]-search_radius
        topextent = self.mean_bng[1][0]+search_radius
        shapes = gpd.read_file(shape_path)
        shapes_mini = shapes.cx[leftextent-search_radius:rightextent+search_radius,
                                bottomextent-search_radius:topextent+search_radius]
        self.shapes = shapes_mini
        
    def generate_points(self, polygon, resolution):   
        """
        Generate some points within a Polygon object.
        At this point, not guaranteed to sit within the Polygon's bounds.
        Add a central point too so it never fails to find any
        """
        minx, miny, maxx, maxy = polygon.bounds
        x = np.arange(minx, maxx, 1/resolution)
        y = np.arange(miny, maxy, 1/resolution)
        XY = np.meshgrid(x, y)
        
        return XY[0].ravel(), XY[1].ravel()

    def     get_points_in_poly(self, resolution, floor_height=0, plot=False):
        """
        Generate lots of points in the region of a Polygon, then 
        filter out all the points that sit outside the perimiter. 
        Returns the geopandas dataframe of points that sit within the perimeter.
        """
        ## Calculate a grid of points that sit within this shape
        gdf_poly = gpd.GeoDataFrame(index=["myPoly"], geometry=[self.border_m_poly])
        x,y = self.generate_points(self.border_m_poly, resolution)
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
            
        self.garden_samples_gpd = pnts_in_poly
        self.garden_samples = np.column_stack([np.concatenate(self.garden_samples_gpd['points'].apply(lambda x: np.array([x.x, x.y])).values).reshape(-1,2), np.full(len(self.garden_samples_gpd), floor_height)])
    def raycast_instant(self, datetime, timer=False, floor_height=0, plot_3d=False):
        """
        Perform raycasting for one time step only.

        Parameters
        ----------
        datetime : TYPE
            DESCRIPTION.
        timer : TYPE, optional
            DESCRIPTION. The default is False.
        floor_height : TYPE, optional
            DESCRIPTION. The default is 0.
        plot_3d : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        heatmap : np.array
            The heatmap showing whether or not each sample is in the sun 
            at that instant.

        """
        self.centre = (self.mean_lat, self.mean_lon)
        azimuth, elevation = sunpos(datetime, self.centre, True)

        # Draw a box around the garden
        # This then gets projected towards the sun (later), then any buildings
        # within the resulting prism will be considered for ray casting
        self.bounding_box_root = calculate_bounding_box_root(self.border_ll, o=0, method=1)
        
        # Initialise an empty heatmap with value of 1
        heatmap = np.full(len(self.garden_samples), 1).astype(float)
        if elevation >= 0:
            # Find which buildings could cast a shadow on the garden
            casting_buildings = get_casting_buildings(datetime,
                                                      self.centre,
                                                      self.shapes,
                                                      self.bounding_box_root,
                                                      cast_distance=5000,
                                                      plot_bounding_box=False, 
                                                      plot_casting_buildings=False)
            self.casting_buildings = casting_buildings
            
            # Based on the footprint geometries, build up all the different
            # faces of all the buildings into a list
            faces = construct_building_walls(casting_buildings, 
                                             floor_height=floor_height)
            
            # Optional: plot all the faces in 3D (matplotlib)
            if plot_3d:
                plot_faces(faces)
                
            # Calculate the direction of the sun's vector
            sunvector = spherical_to_cartesian(azimuth, elevation)
            
            # Perform raycasting for the given time
            self.heatmap, faces = raycast_time(heatmap,
                                               faces,
                                               self.garden_samples,
                                               sunvector)
            
            
            # Calculate the position of the sun 
            # We use this to plot in the plotly map
            # (after converting to lat/lon)
            sunm = -300*sunvector[:2]
            sun_lat = m_to_lat(sunm[0]) + self.centre[0]
            sun_lon = m_to_lon(sunm[1], lat=sun_lat) + self.centre[1]
            self.sunlatlon = [sun_lat, sun_lon]
            
            return self.heatmap
        else:
            print(f"Sun is below the horizon!")
            
    def raycast_day(self,
                    date,
                    minute_resolution=60,
                    timer=False,
                    floor_height=0,
                    plot_casting_buildings=False, 
                    plot_3d=False):
        """
        Perform raycasting for a whole day and return a heatmap.
        The heatmap shows how many hours of sun there are for that spot.

        Parameters
        ----------
        date: tuple
            Should be of format (yyyy, mm, dd), e.g. (2022, 6, 25).
        minute_resolution : int, optional
            How many minutes to increment in every hour.
            E.g. if '15', then we move forward by 15 minutes in every time step. 
            The default is 60.
        timer : bool, optional
            Whether or not to time the raycasting. If True, it'll print to console
            once finished. The default is False.
        floor_height : float, optional
            Height of the floor. TODO: implement terrain!.
            The default is 0.

        Returns
        -------
        heatmap : np.array
            The heatmap showing how many hours of sun there are for each 
            sample in the garden. E.g.: 0 = no sun, 24 = 24hrs of sun.
            Note: only counts the valid daylight hours (when elevation >= 0)

        """
        self.casting_buildings = None
        self.sunlatlon = None
        self.centre = (self.mean_lat, self.mean_lon)
        
        if timer:
            start = time.time()
        
        # Get an array of all the minutes we want to scan over in every hour
        minute_steps = np.arange(0, 60, minute_resolution).astype(int)
        
        # Initialise the heat map, starting with the full amount of sun possible
        self.heatmap = np.full(len(self.garden_samples), 24*len(minute_steps)).astype(float)
        
        for hour in range(0,24):
            for minute in minute_steps:
                print(f"{hour}h {minute}m...")
                datetime  = (date[0], date[1], date[2], hour, minute, 0, 0)
                azimuth, elevation = sunpos(datetime, self.centre, True)
                if elevation >= 0:
                    # Get the bounding box of the garden
                    self.bounding_box_root=calculate_bounding_box_root(self.border_ll, o=0, method=1)
                    
                    # Based on the bounding box, get a gpd of buildings that *could* cast on our garden
                    casting_buildings = get_casting_buildings(datetime, self.centre, self.shapes, self.bounding_box_root, cast_distance=5000)
                    
                    # Construct all the faces based on the footprints
                    faces = construct_building_walls(casting_buildings, floor_height=floor_height)
                    
                    if plot_3d:
                        plot_faces(faces)
                        
                    if plot_casting_buildings:
                        fig, ax = plt.subplots()
                        self.shapes.plot(color="gray", ax=ax)
                        casting_buildings.plot(color="red", ax=ax)
                        ax.set_title(f"Hour {hour}, minute {minute}")
                    
                    sunvector = spherical_to_cartesian(azimuth, elevation)
                    self.heatmap, faces = raycast_time(self.heatmap, faces, self.garden_samples, sunvector)
                else:
                    self.heatmap -= 1

        if timer:
            end = time.time()
            print(f"Time taken to raycast: {end-start}s.")
        return self.heatmap
   
    
    # def project_vertices(self, vertices, s):
    #     # Calculate the normal vector of the plane as the z-axis
    #     normal = np.array([0, 0, 1])
    
    #     # Calculate the projection of the vertices onto the plane
    #     projected_vertices = vertices - ((vertices - s) @ normal)[:, np.newaxis] * normal
    
    #     # Translate the vertices along the plane normal to ensure they lie exactly on the plane
    #     t = -projected_vertices[:, 2][:, np.newaxis]
    #     projected_vertices += t * normal
    
    #     return projected_vertices
       
    # def project_shadows(self, datetime, floor_height=0):
    #     self.centre = (self.mean_lat, self.mean_lon)
    #     azimuth, elevation = sunpos(datetime, self.centre, True)
    #     sunvector = spherical_to_cartesian(azimuth, elevation)
    #     bounding_box_root = calculate_bounding_box_root(self.border_ll, o=0, method=1)
    #     if elevation >= 0:
    #         self.casting_buildings_projected = get_casting_buildings(datetime,
    #                                                   self.centre,
    #                                                   self.shapes,
    #                                                   bounding_box_root,
    #                                                   cast_distance=5000,
    #                                                   plot_bounding_box=False, 
    #                                                   plot_casting_buildings=False)
    #         faces = construct_building_walls(self.casting_buildings_projected, 
    #                                          floor_height=floor_height)   
        
    #         all_shadows = []
    #         for face in faces:
    #             projected_face = self.project_vertices(face, sunvector)
                
    #             # Convert back to lat, lon
    #             lat = m_to_lat(projected_face[:, 0])
    #             lon = m_to_lon(projected_face[:, 1], lat)
    #             latlon = np.stack([lat, lon]).T
                
    #             shadow = Polygon(latlon)
    #             all_shadows.append(shadow)
            
    #         self.projected_shadows = all_shadows
            
    #     elif elevation < 0:
    #         print("SUN IS BELOW THE HORIZON!")
            
            
    # def plot_projected_shadows(self):
    #     plot_projected_shadows_plotly(self.casting_buildings_projected["geometry"], self.projected_shadows, self.centre, self.border_ll, zoom=18)
        
    def plot(self, zoom=19, markersize=8, xsplits=None, ysplits=None):
        """
        Plot the heatmap in Plotly.
        If we've chosen to plot a single time only, then the buildings and 
        the sun will also be drawn
        
        ysplits and xsplits does interpolation but it doesn't really work
        """
        # If we've passed the casting buildings
        if self.casting_buildings is not None:
            buildings_to_plot = self.casting_buildings["geometry"]
        else:
            buildings_to_plot = None
       
        df = plot_scatter_heatmap_plotly(self.garden_samples,
                            self.heatmap,
                            self.sunlatlon,
                            buildings=buildings_to_plot,
                            zoom=zoom,
                            markersize=markersize,
                            xsplits=xsplits,
                            ysplits=ysplits)


# Define the border of the garden
garden_latlon = np.array([[51.46741877305855, -0.023404599059418246],
                            [51.46719487087323, -0.02357089601567079],
                            [51.4672199346053, -0.02364465676239571],
                            [51.46731601211739, -0.023578942642586233],
                            [51.46735026579013, -0.023696959837346106],
                            [51.46741877305855, -0.023404599059418246]])


#garden_latlon = np.array([[51.46749512258817, -0.023820286057519548],
#                         [51.46740825069726, -0.02253889204326329],
#                         [51.46672388749404, -0.02335855036458685], 
#                         [51.46678180297227, -0.02454542800945235],
#                         [51.46749512258817, -0.023820286057519548]])

# Instantiate sunnygarden class
garden = sunnygarden(garden_latlon)

# Fetch the building geometries in a given area
garden.fetch_buildings(shape_path="../geospatial_data/processed/building_database_SE_England.shp", search_radius=150)

# Create samples in the garden to use for raycasting
garden.get_points_in_poly(resolution=1,
                          floor_height=0,
                          plot=False)

# Perform raycasting for a whole day to get the heatmap
heatmap = garden.raycast_day(date=(2022, 2, 8),       # (yyyy, mm, dd)
                             minute_resolution=60,
                             timer=True, 
                             plot_casting_buildings=False, # Show which buildings are being considered at every time step, 2D
                             plot_3d=False)           # Plot which buildings are being considered, 3D

# Instead, do raycasting for a single timestep
#heatmap = garden.raycast_instant(datetime=(2022, 2, 16, 15, 18, 0, 0), plot_3d=True)

# Plot the results in an interactive plotly map
# Result here changes depending on whether we did a whole day or just an instant
garden.plot()