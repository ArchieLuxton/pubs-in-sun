from utils import latlon_to_m
from plots import plot_vector_and_raster
import numpy as np
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
import rasterio
import geopandas
from convertbng.util import convert_bng, convert_lonlat
import shapely
from rasterio.windows import Window
from utils import crop_tiff


def combine_DTM_DSM(dsmpath, dtmpath, chmpath):
    """ Combine the DTM and DSM shapefiles to get the CHM. Save as GeoTif """
    with rasterio.open(dsmpath) as src:
        DSM = src.read(1, masked=True)
        dsm_meta = src.profile
    DTM = rasterio.open(dtmpath)
    CHM_data = DSM-DTM.read(1) 
    # export chm as a new geotiff to use or share with colleagues
    try:
        with rasterio.open(chmpath, 'w', **dsm_meta) as ff:
            ff.write(CHM_data,1)
    except:
        print(f"Failed to overwrite the CHM file! Have you got it open?")
        
    return CHM_data


def read_or_merge_shapefiles(savepath, CHM, path1, path2, path3):
    """
    Read the multiple shapefiles that make up all the different buildings. 
    Merge them together, and save the merged one. 
    If the merged version already exists, just open it up.
    Only saves the GeoTiff in the same region as the CHM covers!
    """
    try:
        shapes = gpd.read_file(savepath)
    except:
        print("Can't find combined shape file! Constructing it from the raw data...")
        TQ_buildings = gpd.read_file(path1)
        TQ_buildings_cropped = TQ_buildings.cx[CHM.bounds.left:CHM.bounds.right, CHM.bounds.bottom:CHM.bounds.top]
        #TQ_buildings_cropped.to_file(f"TQ_buildings_cropped.shp")
        print("Finished main building set")
    
        TQ_important = gpd.read_file(path2)
        TQ_important_cropped = TQ_important.cx[CHM.bounds.left:CHM.bounds.right, CHM.bounds.bottom:CHM.bounds.top]
        #TQ_important_cropped.to_file(f"TQ_important_buildings_cropped.shp")
        print("Finished important buildings set")
    
        TQ_glasshouse = gpd.read_file(path3)
        TQ_glasshouse_cropped = TQ_glasshouse.cx[CHM.bounds.left:CHM.bounds.right, CHM.bounds.bottom:CHM.bounds.top]
        #try:
        #    TQ_glasshouse_cropped.to_file(f"TQ_glasshouse_cropped.shp")
        #except:
        #    pass
        print("Finished glass house set")
    
        shapes = geopandas.overlay(TQ_buildings_cropped, TQ_important_cropped, how='union')
        shapes.to_file(savepath)
    
    return shapes
    





#def request_garden_latlon():
#    return garden_latlon


#def create_shapefile(dsmpath, dtmpath, chmpath, )
if __name__ == "__main__":
    """ 
    This script:
        1) Opens up the DTM and DSM LIDAR images to create the CHM (Canopy Height Model), then saving the result.
        2) Opens up the new CHM image
        3) Combines the shape files for the different building types (from OS) and saves to one shapefile
        4) Based on the coordinates of our garden (defined globally), crops the shape file to a relevant area 
        5) Calculates the mean heights for all these buildings in our area of interest
        6) Saves this cropped shape file (with mean heights) to file
    """
    
    ## DEFINE THE GARDEN WE WANT TO LOOK AT
    # Note: Defined clockwise, point [-1] should == point [0]
    ## TALBOT
    garden_latlon = np.array([[51.46741877305855, -0.023404599059418246],
                                [51.46719487087323, -0.02357089601567079],
                                [51.4672199346053, -0.02364465676239571],
                                [51.46731601211739, -0.023578942642586233],
                                [51.46735026579013, -0.023696959837346106],
                                [51.46741877305855, -0.023404599059418246]])

    ## DOG AND BELL
    #garden_latlon = np.array([[51.483384154708666, -0.026016725385063718], 
    #                           [51.483448908309114, -0.025964734596783293], 
    #                           [51.48347468980964, -0.025595022324566948], 
    #                           [51.483384154708666, -0.025643161943345115],
    #                           [51.483384154708666, -0.026016725385063718]])
    
    crop= True
    
    # Where to save the full CHM (Canopy Height Model) LIDAR image
    full_CHM_savepath = "../geospatial_data/processed/chm_TQ37ne_SE_LDN.tiff"
    # Where to save the full merged shapefile (just building perimeters)
    full_shapefile_savepath = "../geospatial_data/processed/SE_England_buildings_merged_raw.shp"
    # Where to save the full 'building database' - the shapefile with mean heights included 
    full_building_DB_savepath = "../geospatial_data/processed/building_database_SE_England.shp"
    
    
    # Where to save the cropped CHM (Canopy Height Model) LIDAR image - based on inputted garden
    cropped_CHM_savepath = "../geospatial_data/processed/chm_near_talbot.tiff"
    # Where to save the cropped 'building database' - the shapefile with the mean heights included
    cropped_building_DB_savepath = "../geospatial_data/processed/building_database_near_talbot.shp"
    


    ## COMBINE DTM (DIGITAL TERRAIN MODEL) AND DSM (DIGITAL SURFACE MODEL) TO GET CHM (CANOPY HEIGHT MODEL)
    # Return the array of CHM data
    combine_DTM_DSM(dsmpath = "../geospatial_data/raw/LIDAR-COMPOSITE-FIRST-RETURN-DSM/TQ37ne_FZ_DSM_1m.tif",
                    dtmpath = "../geospatial_data/raw/LIDAR-COMPOSITE-DTM/TQ37ne_DTM_1m.tif",
                    chmpath = full_CHM_savepath)
    
    
    ## OPEN UP THE CANOPY HEIGHT MODEL (I.E. BUILDING HEIGHTS FROM LIDAR)
    # Return the CHM object
    CHM = rasterio.open(full_CHM_savepath)


    ## GET THE FULL LIST OF SHAPES (BUILDING FOOTPRINTS) FROM OUR SHAPEFILE (FROM OS MAPS).
    # Merge all files together if merged file can't be found
    shapes = read_or_merge_shapefiles(savepath = full_shapefile_savepath,
                                      CHM=CHM,
                                      path1 = "../geospatial_data/raw/OS-MAPS-SE/OS_OpenMap_Local_ESRI_Shape_File_TQ/data/TQ_Building.shp",
                                      path2 = "../geospatial_data/raw/OS-MAPS-SE/OS_OpenMap_Local_ESRI_Shape_File_TQ/data/TQ_ImportantBuilding.shp",
                                      path3 = "../geospatial_data/raw/OS-MAPS-SE/OS_OpenMap_Local_ESRI_Shape_File_TQ/data/TQ_Glasshouse.shp")
    shapes["centroid_x"] = shapes["geometry"].centroid.x
    shapes["centroid_y"] = shapes["geometry"].centroid.y
    
    

    ## FILTER THE SHAPEFILE TO THE LIMITS OF OUR LIDAR DATA (CHM)
    shapes = shapes.cx[CHM.bounds.left:CHM.bounds.right,
                       CHM.bounds.bottom:CHM.bounds.top]
    #shapes = shapes[(shapes["centroid_x"] < CHM.bounds.right) & \
    #                     (shapes["centroid_x"] > CHM.bounds.left) & \
    #                     (shapes["centroid_y"] < CHM.bounds.top) & \
    #                     (shapes["centroid_y"] > CHM.bounds.bottom)]
        

    
    if crop:
    
        ## CROP THE SHAPE FILE 
        garden_lat = np.mean(garden_latlon[:,0])
        garden_lon = np.mean(garden_latlon[:,1])
        garden_bng = convert_bng(garden_lon, garden_lat)
        garden_ind = CHM.index(garden_bng[0][0], garden_bng[1][0])
        print(f"In BNG: {garden_bng}")  # Easting (lon), Northing (lat) 
        print(f"As index: {garden_ind}")  
    
            
        ## GET SMALL SUBSET OF GEOMETRY TO TEST ON
        offset = 250   # Grid units (BNG), possibly metres?
        leftextent = garden_bng[0][0]-offset
        rightextent = garden_bng[0][0]+offset
        bottomextent = garden_bng[1][0]-offset
        topextent = garden_bng[1][0]+offset

        # More efficient than normal Pandas syntax
        #shapes_mini = shapes.cx[mini_leftextent:mini_rightextent,
        #                        mini_bottomextent:mini_topextent]
        shapes= shapes[(shapes["centroid_x"] < rightextent) & \
                             (shapes["centroid_x"] > leftextent) & \
                             (shapes["centroid_y"] < topextent) & \
                             (shapes["centroid_y"] > bottomextent)]
            
        shapes_savename = cropped_building_DB_savepath
        CHM_path = cropped_CHM_savepath
        
        ## CROP THE CHM GEOTIFF:
        crop_tiff(openpath=full_CHM_savepath,
                  savepath=CHM_path,
                  top=topextent,
                  bottom=bottomextent,
                  left=leftextent,
                  right=rightextent)
        
        
        
        
    else:
        leftextent = CHM.bounds.left
        rightextent = CHM.bounds.right
        bottomextent = CHM.bounds.bottom
        topextent = CHM.bounds.top
        
        shapes_savename = full_building_DB_savepath
        CHM_path = full_CHM_savepath
       
    ## CALCULATE THE HEIGHTS OF BUILDINGS IN SUBSET AND DROP ONES THAT ARE TOO SHORT
    mean_heights = pd.DataFrame(
        zonal_stats(vectors=shapes['geometry'], 
                    raster=CHM_path, 
                    stats='mean'))['mean']
    shapes["height"] = mean_heights.values
    min_height = 1  # metres
    shapes[shapes["height"]<min_height] = None
    shapes= shapes[shapes["height"].notna()]
    
    
    shapes.to_file(shapes_savename)
    

    ## PLOT THE SMALL SUBSET 
    plot_vector_and_raster(df=shapes,
                           bottomextent = bottomextent, 
                           topextent = topextent, 
                           leftextent = leftextent, 
                           rightextent = rightextent,
                           raster_path = CHM_path)
    

    