from utils import m_to_lat, m_to_lon, interp_df
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from scipy.interpolate import griddata
from convertbng.util import convert_bng, convert_lonlat
import rasterio
import matplotlib as mpl
import plotly.graph_objects as go
import plotly.io as pio
import shapely
import plotly.express as px
pio.renderers.default='browser'

def plot_vector_and_raster(df, raster_path, bottomextent, topextent, leftextent, rightextent):
    # Load the LIDAR data with rasterio
    with rasterio.open(raster_path) as src:
        height_data = src.read(1)
        crs = src.crs
        transform = src.transform

    # Get the bounds of the height data
    bounds = src.bounds

    # Plot the LIDAR data using matplotlib
    fig, ax = plt.subplots()
    im = ax.imshow(height_data, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], cmap='viridis',vmin=0,vmax=20)

    # Plot the building footprints on top of the LIDAR data using geopandas
    df.plot(ax=ax, facecolor='none', edgecolor='red')

    plt.ylim((bottomextent, topextent))
    plt.xlim((leftextent, rightextent))

    # Add a title and show the plot
    plt.title('Building Footprints and LIDAR Height Data')
    plt.show()
    
    
def plot_faces(sides, extent_offset=5):
    #%matplotlib notebook
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')


    for i, face in enumerate(sides):
        vertices = [[tuple(v) for v in face]]
        poly = Poly3DCollection(vertices, alpha=0.8)
        ax.add_collection3d(poly)

    all_faces = np.concatenate(np.array(sides))

    mins = all_faces.min(axis=0)
    maxs = all_faces.max(axis=0)

    ax.set_xlim(mins[0]-extent_offset,maxs[0]+extent_offset)
    ax.set_ylim(mins[1]-extent_offset,maxs[1]+extent_offset)
    ax.set_zlim(mins[2]-extent_offset,maxs[2]+extent_offset) 

def plot_heatmap_matplotlib(garden_m_poly,garden_samples_gpd, sample_points, heatmap):
    fig, ax = plt.subplots()
    ax.plot(*garden_m_poly.exterior.xy)
    sample_points = np.array([[list(garden_samples_gpd.points)[i].x, list(garden_samples_gpd.points)[i].y] for i in range(len(garden_samples_gpd))])
    plot = ax.scatter(sample_points[:,0], sample_points[:,1], c=heatmap, cmap='copper', s=28)
    plt.colorbar(plot)


def plot_scatter_heatmap_plotly(garden_samples, heatmap, sunlatlon, buildings=None, zoom=19, markersize=5, xsplits=100, ysplits=100):
    
    def get_colormap(hours, start=0, end=np.max(heatmap)):
        cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["black", "yellow"])
        norm = mpl.colors.Normalize(vmin=start, vmax=end)
        color_values = cmap(norm(hours))
        return color_values
    
    shadow_df = pd.DataFrame([m_to_lat(garden_samples[:,0]), m_to_lon(garden_samples[:,1],m_to_lat(garden_samples[:,0])), heatmap]).T
    shadow_df.columns = ["lat","lon","hours"]
    
    if (xsplits is not None) & (ysplits is not None):
        shadow_df = interp_df(shadow_df, xsplits=xsplits, ysplits=ysplits)


    traces = []
    
    colours = get_colormap(shadow_df["hours"].values)
    
    # Plot the heatmap
    if np.max(heatmap) == 1:
        colours_rgba = []
        for i in range(len(heatmap)):
            if heatmap[i] == 0:
                colours_rgba.append("rgba(0.0, 0.0, 0.0, 1.0)")
            elif heatmap[i] == 1:
                colours_rgba.append("rgba(255, 255, 0.0, 1.0)")
    else:
        colours_rgba = [f"rgba{tuple(c)}" for c in colours]
    
    # Plot our gardeb
    traces.append(go.Scattermapbox(lat=shadow_df["lat"],
                                   lon=shadow_df["lon"],
                                   mode='markers',
                                   marker=go.scattermapbox.Marker(size=markersize, color=colours_rgba)))#, symbol="square")))
    
    # Plot the sun (if we want to)
    if sunlatlon is not None:
        print(sunlatlon)
        traces.append(go.Scattermapbox(lat=[sunlatlon[0]],
                                       lon=[sunlatlon[1]],
                                       mode='markers',
                                       marker=go.scattermapbox.Marker(size=20, color="yellow")))#, symbol="square")))
        
        
        
    if buildings is not None:
        for i, b in enumerate(buildings):
            if isinstance(b, shapely.geometry.polygon.Polygon):
                bng = np.array(b.exterior.coords.xy).T
            elif isinstance(b, shapely.geometry.multipolygon.MultiPolygon):
                bng = np.array(b.convex_hull.exterior.coords.xy).T
                
            lonlat = np.array(convert_lonlat(bng[:,0], bng[:,1])).T
            traces.append(go.Scattermapbox(
                name="Way: {b}",
                fill = "toself",
                lon = lonlat[:,0], lat = lonlat[:,1],
                marker = { 'size': 1, 'color': "red" },
            hovertemplate ="""
            <b>Address</b>:{buildings[b]["address"]}
            <br>
            <b>Levels</b>:{buildings[b]["levels"]}"""))
            
    fig = go.Figure(data=traces, layout=dict()) 
    fig.update_layout(
        mapbox = {
            'style': "stamen-terrain",
            'center': {'lon': shadow_df["lon"].iloc[0],
                       'lat': shadow_df["lat"].iloc[0]},
            'zoom': zoom},
        showlegend = False)
    if (xsplits is None) | (ysplits is None):
        fig.update_traces(hovertemplate=[f"{shadow_df.hours[i]:.2f} Hours<br><b>Lat: {shadow_df.lat[i]}<br>Lon: {shadow_df.lon[i]}</b>" for i in range(len(shadow_df))])

    #fig.update_traces(hovertemplate=[f"{heatmap[i]} Hours" for i in range(len(heatmap))])
    
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
    
    return shadow_df

def plot_projected_shadows_plotly(buildings, shadows, centre, garden_ll, zoom=18):
    traces = []
    colormap = {-999:"black",
               1:"yellow",
               2:"orange",
               3:"red",
               4:"green"}
    

    for s in shadows:
        # Convert to a numpy array of shape (N,2)
        if isinstance(s, shapely.geometry.polygon.Polygon):
            latlon = np.array(s.exterior.coords.xy).T
        elif isinstance(s, shapely.geometry.multipolygon.MultiPolygon):
            latlon = np.array(s.convex_hull.exterior.coords.xy).T
        
        #lonlat = np.array(convert_lonlat(bng[:,1], bng[:,0])).T
        traces.append(go.Scattermapbox(
            name=f"Way: XX",
            fill = "toself",
            lon = latlon[:,1],
            lat = latlon[:,0],
            marker = { 'size': 0, 'color': "grey"},
            hovertemplate =f"""
            <b>Address</b>:XX
            <br>
            <b>Levels</b>:YYY"""))
        
    for i, b in enumerate(buildings):
        if isinstance(b, shapely.geometry.polygon.Polygon):
            bng = np.array(b.exterior.coords.xy).T
        elif isinstance(b, shapely.geometry.multipolygon.MultiPolygon):
            bng = np.array(b.convex_hull.exterior.coords.xy).T
            
        lonlat = np.array(convert_lonlat(bng[:,0], bng[:,1])).T
        traces.append(go.Scattermapbox(
            name="Way: {b}",
            fill = "toself",
            lon = lonlat[:,0], lat = lonlat[:,1],
            marker = { 'size': 1, 'color': "red" },
        hovertemplate ="""
        <b>Address</b>:{buildings[b]["address"]}
        <br>
        <b>Levels</b>:{buildings[b]["levels"]}"""))
        
    traces.append(go.Scattermapbox(
        name="Our garden",
        fill = "toself",
        lon = garden_ll[:,1], lat = garden_ll[:,0],
        marker = { 'size': 1, 'color': "green" },
    hovertemplate ="""
    <b>Address</b>:{buildings[b]["address"]}
    <br>
    <b>Levels</b>:{buildings[b]["levels"]}"""))
        
    fig = go.Figure(data=traces, layout=dict())
    fig.update_layout(
        mapbox = {
            'style': "stamen-terrain",
            'center': {'lon': centre[1],
                       'lat': centre[0]},
            'zoom': zoom},
        showlegend = False)
    
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
    
def plot_heatmap_plotly(garden_samples, heatmap, zoom=19, markersize=5, xsplits=100, ysplits=100):
    
    def get_colormap(hours, start=1, end=np.max(heatmap)):
        cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["black", "yellow"])
        norm = mpl.colors.Normalize(vmin=start, vmax=end)
        color_values = cmap(norm(hours))
        return color_values
    
    shadow_df = pd.DataFrame([m_to_lat(garden_samples[:,0]), m_to_lon(garden_samples[:,1],m_to_lat(garden_samples[:,0])), heatmap]).T
    shadow_df.columns = ["lat","lon","hours"]
    
    if (xsplits is not None) & (ysplits is not None):
        shadow_df = interp_df(shadow_df, xsplits=100, ysplits=100)
    
    traces = []
    
    colours = get_colormap(shadow_df["hours"].values)
    
    colours_rgba = [f"rgba{tuple(c)}" for c in colours]
    
    
    fig = px.density_mapbox(shadow_df, lat='lat', lon='lon', z='hours', radius=10,
                            center=dict(lat=0, lon=180), zoom=0,
                            mapbox_style="stamen-terrain")
    
    #fig = go.Figure(data=traces, layout=dict())
    fig.update_layout(
        mapbox = {
            'style': "stamen-terrain",
            'center': {'lon': shadow_df["lon"].iloc[0],
                       'lat': shadow_df["lat"].iloc[0]},
            'zoom': zoom},
        showlegend = False)
    if (xsplits is None) | (ysplits is None):
        fig.update_traces(hovertemplate=[f"{shadow_df.hours[i]:.2f} Hours<br><b>Lat: {shadow_df.lat[i]}<br>Lon: {shadow_df.lon[i]}</b>" for i in range(len(shadow_df))])

    #fig.update_traces(hovertemplate=[f"{heatmap[i]} Hours" for i in range(len(heatmap))])
    
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
    
    
