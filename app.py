import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import logging
from geopy.geocoders import Nominatim
from shapely.geometry import Point
from shapely.ops import nearest_points
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CENTER = [39.0458, -76.6413]  # Default map center (Baltimore)
CRS_WGS84 = "EPSG:4326"  # WGS84 coordinate system for geocoding
CRS_PROJECTED = "EPSG:26985"  # Projected CRS for Maryland (NAD83 / UTM zone 18N)
BUFFER_DISTANCE_FEET = 500
BUFFER_DISTANCE_METERS = BUFFER_DISTANCE_FEET * 0.3048  # Convert feet to meters


# Data source URLs
DATA_SOURCES = {
    "property_data": "https://geodata.md.gov/imap/services/PlanningCadastre/MD_PropertyData/MapServer/WFSServer?request=GetCapabilities&service=WFS",
    "roads": "https://services.arcgis.com/njFNhDsUCentVYJW/arcgis/rest/services/MDOT_Know_Your_Roads/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson",
    "wetlands": "https://geodata.md.gov/imap/services/Hydrology/MD_Wetlands/MapServer/WFSServer?request=GetCapabilities&service=WFS",
    "floodplain": "https://geodata.md.gov/imap/rest/services/Hydrology/MD_Floodplain/FeatureServer/1/query?outFields=*&where=1%3D1&f=geojson"
}

# Risk Calculation Parameters
WETLAND_DECAY_CONSTANT = 0.005
ROAD_DECAY_CONSTANT = 0.003
FLOODPLAIN_RISK_SCORE = 0.8
WEIGHTS = {
    "wetlands": 0.4,
    "roads": 0.2,
    "size": 0.1,
    "floodplain": 0.3,
}


@st.cache_data
def load_data(source, data_type="geojson"):
    try:
        if data_type == "geojson":
            gdf = gpd.read_file(source)
            return gdf
    except Exception as e:
        logger.error(f"Error loading {source}: {e}")
        st.error(f"Failed to load {source}.")
        return gpd.GeoDataFrame()


def create_map(center_point=None, property_data=gpd.GeoDataFrame(), wetlands_gdf=gpd.GeoDataFrame(),
               roads=gpd.GeoDataFrame(), floodplain_gdf=gpd.GeoDataFrame(), highlighted_parcel=None,
               buffer_gdf=None):  # Add buffer_gdf
    m = folium.Map(
        location=center_point or DEFAULT_CENTER,
        zoom_start=12,
        tiles='CartoDB positron'
    )

    if not property_data.empty:
        folium.GeoJson(property_data, name="Property Data").add_to(m)
    if not wetlands_gdf.empty:
        folium.GeoJson(wetlands_gdf, name="Wetlands").add_to(m)
    if not roads.empty:
        folium.GeoJson(roads, name="Roads").add_to(m)
    if not floodplain_gdf.empty:
        folium.GeoJson(floodplain_gdf, name="Floodplain").add_to(m)

    if highlighted_parcel is not None:
        folium.GeoJson(
            highlighted_parcel.to_crs(CRS_WGS84),
            style_function=lambda x: {'fillColor': 'red', 'color': 'red'}
        ).add_to(m)

    if buffer_gdf is not None:  # Add buffer to map
        folium.GeoJson(
            buffer_gdf.to_crs(CRS_WGS84),
            style_function=lambda x: {'color': 'blue', 'fillOpacity': 0.1}
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def calculate_risk(parcel_row, wetlands, roads, floodplain, max_area):
    """Calculates mosquito risk, considering a buffer around the parcel."""

    if wetlands.empty or roads.empty or floodplain.empty:
        return 0.0

    parcel = parcel_row.geometry
    parcel_proj = parcel.to_crs(CRS_PROJECTED)
    parcel_centroid_proj = parcel_proj.centroid
    parcel_buffer_proj = parcel_centroid_proj.buffer(BUFFER_DISTANCE_METERS)

    # --- Wetlands Risk ---
    wetlands_proj = wetlands.to_crs(CRS_PROJECTED)
    # Intersect wetlands with the buffer
    wetlands_in_buffer = wetlands_proj[wetlands_proj.intersects(parcel_buffer_proj)]
    if not wetlands_in_buffer.empty:
        nearest_wetland = nearest_points(parcel_centroid_proj, wetlands_in_buffer.unary_union)[1]
        distance_to_wetland = parcel_centroid_proj.distance(nearest_wetland)
        risk_wetlands = np.exp(-WETLAND_DECAY_CONSTANT * distance_to_wetland)
    else:
        risk_wetlands = 0  # No wetlands within buffer

    # --- Roads Risk ---
    roads_proj = roads.to_crs(CRS_PROJECTED)
    roads_in_buffer = roads_proj[roads_proj.intersects(parcel_buffer_proj)]
    if not roads_in_buffer.empty:
        nearest_road = nearest_points(parcel_centroid_proj, roads_in_buffer.unary_union)[1]
        distance_to_road = parcel_centroid_proj.distance(nearest_road)
        risk_roads = np.exp(-ROAD_DECAY_CONSTANT * distance_to_road)
    else:
        risk_roads = 0

    # --- Parcel Size Risk ---
    parcel_area = parcel_proj.area  # Use original parcel area
    risk_size = parcel_area / max_area if max_area > 0 else 0

    # --- Floodplain Risk ---
    floodplain_proj = floodplain.to_crs(CRS_PROJECTED)
    # Check if the *buffer* intersects the floodplain
    if floodplain_proj.intersects(parcel_buffer_proj).any():
        risk_floodplain = FLOODPLAIN_RISK_SCORE
    else:
        risk_floodplain = 0

    # --- Combined Risk ---
    total_risk = (
        WEIGHTS["wetlands"] * risk_wetlands +
        WEIGHTS["roads"] * risk_roads +
        WEIGHTS["size"] * risk_size +
        WEIGHTS["floodplain"] * risk_floodplain
    )

    return total_risk


def geocode_address(address, property_data):
    """Geocodes an address using the property data."""
    if property_data.empty:
        return None

    full_address_series = (
        property_data['Address'].fillna('') + " " +
        property_data['Street Address Number'].fillna('') + " " +
        property_data['Street Address Name'].fillna('') + " " +
        property_data['Street Address Type'].fillna('') + " " +
        property_data['City'].fillna('') + " " +
        property_data['Zip Code'].fillna('')
    )

    match = full_address_series.str.contains(address, case=False, na=False)

    if match.any():
        matched_parcel = property_data[match].iloc[0]
        centroid = matched_parcel.geometry.centroid
        return [centroid.y, centroid.x]
    else:
        return None


def find_nearest_parcel(point, parcels):
    """Finds the nearest parcel to a given point."""
    if parcels.empty:
        return gpd.GeoDataFrame()

    parcels_proj = parcels.to_crs(CRS_PROJECTED)
    point_proj = gpd.GeoSeries([point], crs=CRS_WGS84).to_crs(CRS_PROJECTED)[0]
    nearest = nearest_points(point_proj, parcels_proj.unary_union)
    nearest_parcel = parcels_proj[parcels_proj.geometry == nearest[1]]
    return nearest_parcel.to_crs(CRS_WGS84)


# --- Streamlit App ---
st.title("Mosquito Risk Assessment")

with st.spinner("Loading data..."):
    property_data = load_data(DATA_SOURCES["property_data"])
    roads = load_data(DATA_SOURCES["roads"])
    wetlands = load_data(DATA_SOURCES["wetlands"])
    floodplain = load_data(DATA_SOURCES["floodplain"])

address = st.text_input("Enter an address:", placeholder="Search for an address")

if address:
    with st.spinner("Searching..."):
        center_point = geocode_address(address, property_data)
        if center_point:
            point = Point(center_point[1], center_point[0])
            nearest_parcel = find_nearest_parcel(point, property_data)

            if not nearest_parcel.empty:
                if wetlands is not None and roads is not None and floodplain is not None:
                    max_area = property_data.to_crs(CRS_PROJECTED).area.max() if not property_data.empty() else 0

                    # Create the buffer GeoDataFrame
                    parcel_centroid_proj = nearest_parcel.to_crs(CRS_PROJECTED).geometry.centroid.iloc[0]
                    buffer_proj = parcel_centroid_proj.buffer(BUFFER_DISTANCE_METERS)
                    buffer_gdf = gpd.GeoDataFrame({'geometry': [buffer_proj]}, crs=CRS_PROJECTED)

                    risk_score = calculate_risk(nearest_parcel.iloc[0], wetlands, roads, floodplain, max_area)
                    st.metric("Mosquito Risk Score", f"{risk_score:.2f}")

                    main_map = create_map(center_point=center_point, property_data=property_data,
                                          wetlands_gdf=wetlands, roads=roads, floodplain_gdf=floodplain,
                                          highlighted_parcel=nearest_parcel, buffer_gdf=buffer_gdf)  # Pass buffer
                    folium_static(main_map)
                else:
                    st.warning("Risk calculation unavailable due to missing data.")

            else:
                st.error("No parcel found near this address.")
        else:
            st.error("Address not found. Please try a different address.")
else:
    st.write("Enter an address to search and calculate risk.")
    main_map = create_map(property_data=property_data, wetlands_gdf=wetlands, roads=roads, floodplain_gdf=floodplain)
    folium_static(main_map)
