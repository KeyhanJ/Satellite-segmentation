"""
Streamlit webapp for satellite segmentation using SAM3.

Workflow:
1. Draw ROI on interactive map
2. Choose text prompt or box prompt
3. Generate segmentation masks
4. Convert to GeoJSON and simplify
5. Display and edit results in custom map component
"""

from __future__ import annotations

import os
import time
import json
from math import cos, radians

# Fix OpenMP duplicate library warning
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from typing import Tuple, Optional

import geopandas as gpd
import leafmap.foliumap as leafmap
import pandas as pd
import streamlit as st
import folium
from rdp import rdp
from samgeo.common import tms_to_geotiff, raster_to_gpkg
from shapely.geometry import MultiPolygon, Polygon
import visvalingamwyatt as vw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
from io import BytesIO

# Import SAM3
try:
    from samgeo import SamGeo3
    SAM3_AVAILABLE = True
except ImportError:
    try:
        from segment_geospatial import SamGeo3
        SAM3_AVAILABLE = True
    except ImportError:
        SamGeo3 = None
        SAM3_AVAILABLE = False

# Import custom map component
try:
    from map_component import GeoJSONMapEditor
    MAP_COMPONENT_AVAILABLE = True
except ImportError:
    GeoJSONMapEditor = None
    MAP_COMPONENT_AVAILABLE = False

try:
    from streamlit_folium import st_folium
except ImportError:
    st_folium = None

# Configuration - files will be created in the same directory as this script
SCRIPT_DIR = Path(__file__).parent.absolute()
ASSETS = {
    "image": SCRIPT_DIR / "satellite.tif",
    "mask": SCRIPT_DIR / "segment.tif",
    "vector": SCRIPT_DIR / "segment.gpkg",
    "classified": SCRIPT_DIR / "classified_segments.geojson",
    "final": SCRIPT_DIR / "final_simplified.geojson",
    "preview": SCRIPT_DIR / "preview_class.geojson",  # Temporary preview file
    "temp_preview": SCRIPT_DIR / "temp_preview_masks.geojson",  # Temporary preview GeoJSON
    "temp_preview_vector": SCRIPT_DIR / "temp_preview_vector.gpkg",  # Temporary vector for conversion
}

# Color mapping for land use classes (similar to SWMM land use maps)
CLASS_COLORS = {
    "building": "#8B0000",  # Dark red
    "flat roof building": "#8B0000",
    "pitched roof building": "#8B0000",
    "Paved Roads Asphalt": "#2F2F2F",  # Dark gray
    "Paved Local Street": "#696969",  # Medium gray
    "Parking Lots Paved": "#A9A9A9",  # Light gray
    "Squares Plazas Paved": "#D3D3D3",  # Very light gray
    "Sidewalks Footpaths Paved": "#E0E0E0",  # Very light gray
    "Courtyards Hardscape": "#B8860B",  # Dark goldenrod
    "Water Bodies Permanent": "#00008B",  # Dark blue
    "Streams Rivers Channel": "#0000CD",  # Medium blue
    "Commercial Industrial Ground": "#8B008B",  # Dark magenta
    "Compacted Gravel Surface": "#CD853F",  # Peru
    "Residential Lawns Garden": "#90EE90",  # Light green
    "Unpaved Paths Trail": "#DEB887",  # Burlywood
    "Institutional Ground": "#FF8C00",  # Dark orange
    "Railway Track": "#654321",  # Dark brown
    "Parks Maintained Grass": "#32CD32",  # Lime green
    "Sports Fields Turf": "#00FF00",  # Green
    "Natural Grassland Meadow": "#98FB98",  # Pale green
    "Forest Woodland": "#228B22",  # Forest green
    "Agricultural Cultivated": "#F5DEB3",  # Wheat
    "Bare Soil Disturbed": "#FFA500",  # Orange
    "Cemetery Ground": "#006400",  # Dark green
}

# Predefined prompts for dropdown selection
PREDEFINED_PROMPTS = [
    "building",
    "flat roof building",
    "pitched roof building",
    "Paved Roads Asphalt",
    "Paved Local Street",
    "Parking Lots Paved",
    "Squares Plazas Paved",
    "Sidewalks Footpaths Paved",
    "Courtyards Hardscape",
    "Water Bodies Permanent",
    "Streams Rivers Channel",
    "Commercial Industrial Ground",
    "Compacted Gravel Surface",
    "Residential Lawns Garden",
    "Unpaved Paths Trail",
    "Institutional Ground",
    "Railway Track",
    "Parks Maintained Grass",
    "Sports Fields Turf",
    "Natural Grassland Meadow",
    "Forest Woodland",
    "Agricultural Cultivated",
    "Bare Soil Disturbed",
    "Cemetery Ground",
]


def check_huggingface_auth(force_check: bool = False) -> bool:
    """Check if user is authenticated with HuggingFace."""
    if not force_check and "hf_authenticated" in st.session_state:
        return st.session_state["hf_authenticated"]
    
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        st.session_state["hf_authenticated"] = True
        if "name" in user_info:
            st.session_state["hf_username"] = user_info["name"]
        return True
    except Exception:
        st.session_state["hf_authenticated"] = False
        return False


def login_huggingface(token: str) -> bool:
    """Login to HuggingFace with a token."""
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        st.session_state["hf_authenticated"] = True
        st.session_state["hf_token"] = token
        return True
    except Exception as e:
        st.error(f"Login failed: {e}")
        st.session_state["hf_authenticated"] = False
        return False


def initialize_sam3(image_path: Path) -> SamGeo3:
    """Initialize SAM3 and set the image."""
    if not SAM3_AVAILABLE or SamGeo3 is None:
        raise ImportError(
            "SAM3 is not available. Please install it with:\n"
            "pip install 'segment-geospatial[samgeo3]'\n"
            "Then restart Streamlit."
        )
    
    if not check_huggingface_auth():
        raise RuntimeError(
            "HuggingFace authentication required. Please log in using the HuggingFace Login section."
        )
    
    try:
        sam3 = SamGeo3(
            backend="transformers",
            device=None,
            checkpoint_path=None,
            load_from_HF=True,
        )
        sam3.set_image(str(image_path))
        return sam3
    except Exception as e:
        raise RuntimeError(f"Failed to initialize SAM3: {e}") from e


def download_imagery(bbox: Tuple[float, float, float, float], output: Path) -> Path:
    """Download satellite imagery for the given bounding box."""
    bounds = list(bbox)
    if len(bounds) != 4:
        raise ValueError(f"bbox must have 4 numbers [xmin, ymin, xmax, ymax], got {bounds}")
    
    # Try to delete existing file if it exists (for clean overwrite)
    # If deletion fails (file locked), tms_to_geotiff will still try to overwrite
    if output.exists():
        try:
            output.unlink()
        except (PermissionError, OSError):
            # File is locked - tms_to_geotiff with overwrite=True should still overwrite it
            # Some libraries can overwrite without deleting first
            pass
    
    # Download with overwrite=True - this will overwrite the file even if delete failed
    tms_to_geotiff(output=str(output), bbox=bounds, zoom=20, source="Satellite", overwrite=True)
    return output


def simplify_douglas(geom: Polygon | MultiPolygon, tolerance: float) -> Polygon | MultiPolygon | None:
    """Simplify geometry using Douglas-Peucker algorithm."""
    if geom is None:
        return None
    if geom.geom_type == "Polygon":
        coords = list(geom.exterior.coords)
        simplified = rdp(coords, epsilon=tolerance)
        return Polygon(simplified) if len(simplified) >= 4 else None
    if geom.geom_type == "MultiPolygon":
        polys = []
        for poly in geom.geoms:
            coords = list(poly.exterior.coords)
            simplified = rdp(coords, epsilon=tolerance)
            if len(simplified) >= 4:
                polys.append(Polygon(simplified))
        return MultiPolygon(polys) if polys else None
    return geom


def simplify_visvalingam(geom: Polygon | MultiPolygon, threshold: float) -> Polygon | MultiPolygon | None:
    """Simplify geometry using Visvalingam-Whyatt algorithm."""
    if geom is None:
        return None
    if geom.geom_type == "Polygon":
        coords = list(geom.exterior.coords)
        simplified = vw.simplify(coords, threshold=threshold)
        return Polygon(simplified) if len(simplified) >= 4 else None
    if geom.geom_type == "MultiPolygon":
        polys = []
        for poly in geom.geoms:
            coords = list(poly.exterior.coords)
            simplified = vw.simplify(coords, threshold=threshold)
            if len(simplified) >= 4:
                polys.append(Polygon(simplified))
        return MultiPolygon(polys) if polys else None
    return geom


def simplify_grid_snapping(geom: Polygon | MultiPolygon, grid_size: float) -> Polygon | MultiPolygon | None:
    """Simplify geometry using grid snapping."""
    if geom is None:
        return None

    def snap_point(pt: Tuple[float, float]) -> Tuple[float, float]:
        return (round(pt[0] / grid_size) * grid_size, round(pt[1] / grid_size) * grid_size)

    if geom.geom_type == "Polygon":
        snapped = [snap_point(pt) for pt in geom.exterior.coords]
        return Polygon(snapped) if len(snapped) >= 4 else None
    if geom.geom_type == "MultiPolygon":
        polys = []
        for poly in geom.geoms:
            snapped = [snap_point(pt) for pt in poly.exterior.coords]
            if len(snapped) >= 4:
                polys.append(Polygon(snapped))
        return MultiPolygon(polys) if polys else None
    return geom


def simplify_polygons(
    input_path: Path,
    output_path: Path,
    method: str = "douglas-peucker",
    douglas_tolerance: float = 1.0,
    visvalingam_threshold: float = 1.0,
    grid_size: float = 1.0,
) -> Path:
    """Simplify polygons from GeoJSON and save as GeoJSON."""
    gdf = gpd.read_file(input_path)
    
    if method == "douglas-peucker":
        gdf["geometry"] = gdf.geometry.apply(lambda geom: simplify_douglas(geom, douglas_tolerance))
    elif method == "visvalingam":
        gdf["geometry"] = gdf.geometry.apply(lambda geom: simplify_visvalingam(geom, visvalingam_threshold))
    elif method == "grid-snapping":
        gdf["geometry"] = gdf.geometry.apply(lambda geom: simplify_grid_snapping(geom, grid_size))
    else:
        raise ValueError(f"Unknown simplification method: {method}")

    gdf = gdf[gdf.geometry.notna()]
    gdf.to_file(output_path, driver="GeoJSON")
    return output_path


def reset_workflow():
    """Reset the workflow for a new ROI segmentation."""
    # Generate a new session ID for this workflow
    st.session_state["session_id"] = f"session_{int(time.time())}"
    
    # Clear all workflow-related session state except authentication
    keys_to_keep = ["hf_authenticated", "hf_username", "hf_token", "session_id"]
    keys_to_remove = [key for key in st.session_state.keys() if key not in keys_to_keep]
    for key in keys_to_remove:
        del st.session_state[key]
    
    # Explicitly clear multiclass tracking and preview
    if "classes_added" in st.session_state:
        del st.session_state["classes_added"]
    if "adding_another_class" in st.session_state:
        del st.session_state["adding_another_class"]
    if "preview_mode" in st.session_state:
        del st.session_state["preview_mode"]
    
    # Clean up preview file if it exists
    if ASSETS["preview"].exists():
        try:
            ASSETS["preview"].unlink()
        except Exception:
            pass


def calculate_roi_area(roi_geojson: dict, roi_crs: str = "EPSG:4326") -> tuple[float, float]:
    """
    Calculate ROI polygon area in m² and hectares.
    
    Args:
        roi_geojson: GeoJSON polygon geometry
        roi_crs: CRS of the ROI polygon (default: EPSG:4326)
    
    Returns:
        tuple: (area_m2, area_ha)
    """
    try:
        from shapely.geometry import shape
        roi_shape = shape(roi_geojson)
        roi_gdf = gpd.GeoDataFrame(geometry=[roi_shape], crs=roi_crs)
        
        # Convert to WGS84 first to get centroid for UTM zone calculation
        roi_wgs84 = roi_gdf.to_crs("EPSG:4326")
        centroid = roi_wgs84.geometry.iloc[0].centroid
        lon, lat = centroid.x, centroid.y
        
        # Determine appropriate UTM zone
        utm_zone = int((lon + 180) / 6) + 1
        # Use UTM for accurate area calculation
        if lat >= 0:
            area_crs = f"EPSG:{32600 + utm_zone}"  # UTM North
        else:
            area_crs = f"EPSG:{32700 + utm_zone}"  # UTM South
        
        # Convert to UTM for area calculation
        roi_utm = roi_gdf.to_crs(area_crs)
        area_m2 = roi_utm.geometry.iloc[0].area
        area_ha = area_m2 / 10000  # Convert to hectares
        
        return area_m2, area_ha
    except Exception as e:
        # Fallback: if conversion fails, return approximate area using WGS84
        # This is less accurate but better than nothing
        try:
            from shapely.geometry import shape
            roi_shape = shape(roi_geojson)
            # Approximate area calculation (less accurate for large areas)
            # Using equirectangular projection approximation
            roi_wgs84 = gpd.GeoDataFrame(geometry=[roi_shape], crs="EPSG:4326")
            bounds = roi_wgs84.total_bounds
            # Rough approximation: assumes small area
            lat_center = (bounds[1] + bounds[3]) / 2
            lon_range = bounds[2] - bounds[0]
            lat_range = bounds[3] - bounds[1]
            # Convert degrees to meters (approximate)
            lat_m = lat_range * 111320  # meters per degree latitude
            lon_m = lon_range * 111320 * abs(cos(radians(lat_center)))  # meters per degree longitude (varies by latitude)
            area_m2 = lat_m * lon_m
            area_ha = area_m2 / 10000
            return area_m2, area_ha
        except Exception:
            return 0.0, 0.0


def detect_roi_change(new_bbox: Optional[Tuple[float, float, float, float]]) -> bool:
    """Detect if ROI has changed from the previous session."""
    if new_bbox is None:
        return False
    
    old_bbox = st.session_state.get("bbox")
    if old_bbox is None:
        return True
    
    # Check if bbox is significantly different (tolerance for floating point differences)
    tolerance = 0.0001
    return any(abs(new_bbox[i] - old_bbox[i]) > tolerance for i in range(4))


def capture_roi() -> Optional[Tuple[float, float, float, float]]:
    """Capture ROI from user-drawn polygon on map."""
    st.subheader("📍 Step 1: Draw ROI (Region of Interest)")
    st.write("Draw a polygon or rectangle on the map below to define your area of interest.")
    
    # Basemap selector (default Satellite)
    basemap_choice = st.selectbox(
        "Basemap",
        options=["SATELLITE", "OpenStreetMap", "Terrain"],
        index=0,
        help="Choose a basemap for ROI drawing (default is Satellite)."
    )
    
    if st_folium is None:
        st.error("streamlit-folium is required. Install via 'pip install streamlit-folium'.")
        return None

    roi_map = leafmap.Map(
        center=[0, 0],
        zoom=2,
        draw_control=True,
        draw_export=False,
        locate_control=True,
        measure_control=True,
    )
    roi_map.add_basemap(basemap_choice)
    
    map_state = st_folium(roi_map, height=500, width=None, key="roi_map")
    
    if not map_state:
        return st.session_state.get("bbox")

    last_shape = map_state.get("last_active_drawing") or (map_state.get("all_drawings") or [{}])[-1]
    geometry = last_shape.get("geometry")
    
    if geometry:
        from shapely.geometry import shape
        polygon = shape(geometry)
        minx, miny, maxx, maxy = polygon.bounds
        bbox = (float(minx), float(miny), float(maxx), float(maxy))
        
        # Check if ROI has changed
        if detect_roi_change(bbox):
            # New ROI detected - reset workflow
            reset_workflow()
            st.info("🔄 New ROI detected. Starting fresh workflow...")
        
        st.session_state["bbox"] = bbox
        st.session_state["roi_geojson"] = geometry
        st.session_state["roi_crs"] = "EPSG:4326"  # Folium/Leaflet maps use WGS84 by default
        
        # Calculate and display ROI area
        try:
            area_m2, area_ha = calculate_roi_area(geometry, "EPSG:4326")
            st.success(f"✓ ROI captured! **Area:** {area_ha:.4f} ha ({area_m2:,.2f} m²)")
        except Exception:
            st.success("✓ ROI captured!")
        
        return bbox
    
    return st.session_state.get("bbox")


def parse_polygon_coordinates(file_content: str) -> tuple[Optional[list], Optional[str]]:
    """
    Parse polygon coordinates from various text file formats.
    
    Supports:
    - Python-like format: polygon_coordinates = [(x1, y1), (x2, y2), ...]
    - CSV format: x,y (one coordinate per line)
    - Simple text: x y (one coordinate per line, space or comma separated)
    
    Returns:
        tuple: (coordinates list, CRS string) or (None, None) if parsing fails
    """
    import re
    
    coordinates = None
    crs = None
    
    # Try to extract CRS from file (look for EPSG patterns)
    crs_match = re.search(r"EPSG[:\s]*(\d+)", file_content, re.IGNORECASE)
    if crs_match:
        crs = f"EPSG:{crs_match.group(1)}"
    
    # Try Python-like format: polygon_coordinates = [(x1, y1), (x2, y2), ...]
    python_pattern = r"polygon_coordinates\s*=\s*\[(.*?)\]"
    python_match = re.search(python_pattern, file_content, re.DOTALL | re.IGNORECASE)
    
    if python_match:
        coords_str = python_match.group(1)
        # Extract all (x, y) tuples
        coord_tuples = re.findall(r"\(([^)]+)\)", coords_str)
        if coord_tuples:
            coordinates = []
            for coord_str in coord_tuples:
                parts = [p.strip() for p in re.split(r"[,;]", coord_str)]
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        coordinates.append((x, y))
                    except ValueError:
                        continue
    else:
        # Try CSV or simple text format
        lines = file_content.strip().split('\n')
        coordinates = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            
            # Try comma-separated
            if ',' in line:
                parts = [p.strip() for p in line.split(',')]
            else:
                # Try space-separated
                parts = line.split()
            
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    coordinates.append((x, y))
                except ValueError:
                    continue
    
    if coordinates and len(coordinates) >= 3:
        # Ensure polygon is closed (first point == last point)
        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])
        return coordinates, crs
    
    return None, None


def capture_roi_from_coordinates() -> Optional[Tuple[float, float, float, float]]:
    """Capture ROI from manually entered polygon coordinates."""
    st.subheader("📍 Step 1: Enter Polygon Coordinates")
    st.write("Enter polygon coordinates manually. One coordinate per line. Supports numbered list format (e.g., `1. (x, y)`), Python tuples `(x, y)`, or simple format `x, y` or `x y`")
    
    # CRS input
    crs_input = st.text_input(
        "CRS (Coordinate Reference System)",
        value="EPSG:4326",
        help="Enter CRS in format EPSG:XXXX (e.g., EPSG:4326 for WGS84, EPSG:32633 for UTM Zone 33N). Default is EPSG:4326."
    )
    
    # Coordinates text area
    coords_text = st.text_area(
        "Polygon Coordinates",
        height=200,
        help="Enter coordinates, one per line. Supported formats:\n- Numbered list: 1. (508299.46, 5396933.51)\n- Python tuple: (508299.46, 5396933.51)\n- Simple: 508299.46, 5396933.51 or 508299.46 5396933.51",
        key="manual_coordinates_input"
    )
    
    if st.button("Load Polygon", key="load_manual_polygon"):
        if not coords_text.strip():
            st.error("❌ Please enter coordinates.")
            return None
        
        try:
            import re
            # Parse coordinates from text
            coordinates = []
            lines = coords_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Try to extract coordinates from parentheses: (x, y) or numbered: 1. (x, y)
                paren_match = re.search(r"\(([^)]+)\)", line)
                if paren_match:
                    # Found coordinates in parentheses
                    coord_str = paren_match.group(1)
                    parts = [p.strip() for p in re.split(r"[,;]", coord_str)]
                    if len(parts) >= 2:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            coordinates.append((x, y))
                            continue
                        except ValueError:
                            pass
                
                # If no parentheses, try simple format: x, y or x y
                if ',' in line:
                    parts = [p.strip() for p in line.split(',')]
                else:
                    # Try space-separated
                    parts = line.split()
                
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        coordinates.append((x, y))
                    except ValueError:
                        st.warning(f"⚠️ Skipped invalid line: {line}")
                        continue
            
            if len(coordinates) < 3:
                st.error("❌ Polygon must have at least 3 points.")
                return None
            
            # Parse CRS
            crs = crs_input.strip() if crs_input.strip() else "EPSG:4326"
            if not crs.upper().startswith("EPSG:"):
                # Try to add EPSG: prefix if missing
                if crs.isdigit():
                    crs = f"EPSG:{crs}"
                else:
                    st.error(f"❌ Invalid CRS format: {crs}. Use format EPSG:XXXX")
                    return None
            
            # Create polygon geometry
            from shapely.geometry import Polygon
            polygon = Polygon(coordinates)
            
            # Convert to GeoDataFrame with original CRS
            roi_gdf_original = gpd.GeoDataFrame(geometry=[polygon], crs=crs)
            
            # Convert to WGS84 (EPSG:4326) - Leaflet default
            roi_gdf = roi_gdf_original.to_crs("EPSG:4326")
            
            if crs != "EPSG:4326":
                st.info(f"✓ Converted polygon from {crs} to EPSG:4326 (WGS84)")
            
            # Calculate bounding box (already in WGS84)
            bounds = roi_gdf.total_bounds
            bbox = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
            
            # Store polygon geometry in WGS84 (EPSG:4326) - same as manually drawn polygons
            roi_geom_wgs84 = roi_gdf.geometry.iloc[0]
            roi_geojson = {
                "type": "Polygon",
                "coordinates": [[[float(x), float(y)] for x, y in roi_geom_wgs84.exterior.coords]]
            }
            
            # Check if ROI has changed
            if detect_roi_change(bbox):
                reset_workflow()
                st.info("🔄 New ROI detected. Starting fresh workflow...")
            
            st.session_state["bbox"] = bbox
            st.session_state["roi_geojson"] = roi_geojson
            st.session_state["roi_crs"] = "EPSG:4326"  # Always store as WGS84 (Leaflet default)
            
            # Calculate and display ROI area
            try:
                area_m2, area_ha = calculate_roi_area(roi_geojson, "EPSG:4326")
                st.success(f"✓ Polygon loaded: {len(coordinates)} vertices, CRS: EPSG:4326 (WGS84)")
                st.info(f"**ROI Area:** {area_ha:.4f} ha ({area_m2:,.2f} m²)")
                st.info(f"Bounding box: [{bbox[0]:.6f}, {bbox[1]:.6f}, {bbox[2]:.6f}, {bbox[3]:.6f}]")
            except Exception as e:
                st.success(f"✓ Polygon loaded: {len(coordinates)} vertices, CRS: EPSG:4326 (WGS84)")
                st.info(f"Bounding box: [{bbox[0]:.6f}, {bbox[1]:.6f}, {bbox[2]:.6f}, {bbox[3]:.6f}]")
            
            st.info("⬇️ Scroll down to download satellite imagery for this ROI.")
            
            return bbox
            
        except Exception as e:
            st.error(f"❌ Failed to process coordinates: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None
    
    return st.session_state.get("bbox")


def show_mask_preview(mask_path: Path, image_path: Path) -> None:
    """Display a simple preview map showing masks as GeoJSON polygons overlaid on satellite imagery."""
    if st_folium is None:
        st.error("streamlit-folium is required for preview.")
        return
    
    # Convert mask to temporary GeoJSON for preview
    temp_preview_geojson = ASSETS["final"].parent / "temp_preview_masks.geojson"
    temp_vector = ASSETS["final"].parent / "temp_preview_vector.gpkg"
    
    try:
        # Convert raster to vector (GeoPackage first)
        raster_to_gpkg(str(mask_path), str(temp_vector), simplify_tolerance=None)
        
        # Load and ensure CRS is WGS84 (EPSG:4326) for web maps
        gdf = gpd.read_file(temp_vector)
        
        # Check and convert CRS if needed
        if gdf.crs is None:
            # If no CRS, assume it's in the same CRS as the image
            try:
                import rasterio
                with rasterio.open(image_path) as src:
                    if src.crs:
                        gdf.set_crs(src.crs, inplace=True)
            except Exception:
                # Default to WGS84 if can't determine
                gdf.set_crs('EPSG:4326', inplace=True)
        
        # Convert to WGS84 if not already
        if gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        
        # Save as GeoJSON
        gdf.to_file(temp_preview_geojson, driver="GeoJSON")
        
        # Get bounds from GeoJSON
        center = [0, 0]
        zoom = 10
        
        if not gdf.empty:
            bounds = gdf.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
            center = [center_lat, center_lon]
            lat_range = bounds[3] - bounds[1]
            if lat_range < 0.01:
                zoom = 15
            elif lat_range < 0.1:
                zoom = 12
            else:
                zoom = 10
        else:
            # Fallback to image bounds if no polygons
            try:
                import rasterio
                with rasterio.open(image_path) as src:
                    bounds = src.bounds
                    center_lat = (bounds.bottom + bounds.top) / 2
                    center_lon = (bounds.left + bounds.right) / 2
                    center = [center_lat, center_lon]
                    lat_range = bounds.top - bounds.bottom
                    if lat_range < 0.01:
                        zoom = 15
                    elif lat_range < 0.1:
                        zoom = 12
                    else:
                        zoom = 10
            except Exception:
                pass
        
        # Create simple preview map with satellite basemap
        preview_map = leafmap.Map(
            center=center,
            zoom=zoom,
            draw_control=False,  # No editing needed
            locate_control=False,
            measure_control=False,
        )
        
        # Add satellite basemap (Esri World Imagery)
        preview_map.add_basemap("SATELLITE")
        
        # Add GeoJSON polygons overlay using folium directly
        if temp_preview_geojson.exists():
            # Read GeoJSON and ensure CRS is correct
            gdf_preview = gpd.read_file(temp_preview_geojson)
            if not gdf_preview.empty:
                # Ensure CRS is WGS84 for web display
                if gdf_preview.crs is None:
                    gdf_preview.set_crs('EPSG:4326', inplace=True)
                elif gdf_preview.crs != 'EPSG:4326':
                    gdf_preview = gdf_preview.to_crs('EPSG:4326')
                
                # Convert to GeoJSON format (folium expects GeoJSON in WGS84)
                geojson_data = gdf_preview.to_json()
                folium.GeoJson(
                    geojson_data,
                    style_function=lambda feature: {
                        'fillColor': '#3388ff',
                        'color': '#3388ff',
                        'weight': 2,
                        'fillOpacity': 0.5,
                    },
                    name="Segmentation Masks"
                ).add_to(preview_map)
        
        # Display map
        st_folium(preview_map, height=500, width=None, key="mask_preview_map")
        
        # Clean up temporary files
        if temp_vector.exists():
            try:
                temp_vector.unlink()
            except Exception:
                pass
        
    except Exception as e:
        st.error(f"Failed to create preview: {e}")
        # Fallback: try to show raster if GeoJSON conversion fails
        try:
            preview_map = leafmap.Map(center=[0, 0], zoom=2)
            preview_map.add_basemap("SATELLITE")
            if mask_path.exists():
                preview_map.add_raster(str(mask_path), layer_name="Segmentation Masks", palette="Set1", opacity=0.6)
            st_folium(preview_map, height=500, width=None, key="mask_preview_map_fallback")
        except Exception:
            pass
    
    # Clean up temporary GeoJSON after display (will be cleaned up on next run)
    if temp_preview_geojson.exists():
        try:
            # Don't delete immediately - let it be cleaned up on next preview generation
            pass
        except Exception:
            pass


def capture_box_prompts() -> Optional[list]:
    """Capture bounding boxes from user-drawn rectangles on map."""
    st.subheader("📦 Draw Bounding Boxes for Segmentation")
    st.write("Draw rectangles on the map to define areas for segmentation. You can draw multiple boxes.")
    
    if st_folium is None:
        st.error("streamlit-folium is required.")
        return None
    
    # Show the satellite image
    # First, determine center and zoom if we have the image
    center = [0, 0]
    zoom = 2
    
    if ASSETS["image"].exists():
        # Get raster bounds to set map center and zoom
        try:
            import rasterio
            with rasterio.open(ASSETS["image"]) as src:
                bounds = src.bounds
                # Calculate center from bounds
                center_lat = (bounds.bottom + bounds.top) / 2
                center_lon = (bounds.left + bounds.right) / 2
                center = [center_lat, center_lon]
                # Estimate zoom level based on bounds size
                lat_range = bounds.top - bounds.bottom
                if lat_range < 0.01:
                    zoom = 15
                elif lat_range < 0.1:
                    zoom = 12
                else:
                    zoom = 10
        except Exception:
            # If rasterio fails, use default view
            pass
    
    # Create map with draw controls enabled (same as ROI map)
    box_map = leafmap.Map(
        center=center,
        zoom=zoom,
        draw_control=True,
        draw_export=False,
        locate_control=True,
        measure_control=True,
    )
    
    if ASSETS["image"].exists():
        box_map.add_raster(str(ASSETS["image"]), layer_name="ROI image")
    else:
        box_map.add_basemap("SATELLITE")
    
    map_state = st_folium(box_map, height=600, width=None, key="box_prompt_map")
    
    if not map_state:
        return None
    
    all_drawings = map_state.get("all_drawings", [])
    if not all_drawings:
        return None
    
    # Convert polygons to bounding boxes
    boxes = []
    from shapely.geometry import shape
    
    for drawing in all_drawings:
        geom = drawing.get("geometry")
        if geom and geom.get("type") == "Polygon":
            try:
                polygon = shape(geom)
                minx, miny, maxx, maxy = polygon.bounds
                boxes.append([float(minx), float(miny), float(maxx), float(maxy)])
            except Exception:
                continue
    
    if boxes:
        st.success(f"✓ Captured {len(boxes)} bounding box(es)")
        return boxes
    
    return None


def generate_land_use_map_image(
    final_gdf: gpd.GeoDataFrame,
    roi_gdf: gpd.GeoDataFrame,
    class_stats: list,
    roi_area_m2: float,
    output_path: Optional[Path] = None
) -> BytesIO:
    """
    Generate a land use map visualization with legend.
    
    Args:
        final_gdf: GeoDataFrame with classified segments
        roi_gdf: GeoDataFrame with ROI polygon
        class_stats: List of dicts with class statistics
        roi_area_m2: Total ROI area in m²
        output_path: Optional path to save the image
    
    Returns:
        BytesIO object containing the image
    """
    # Ensure both are in the same CRS (use the final_gdf CRS)
    if final_gdf.crs is None:
        final_gdf.set_crs("EPSG:4326", inplace=True)
    if roi_gdf.crs is None:
        roi_gdf.set_crs("EPSG:4326", inplace=True)
    
    if final_gdf.crs != roi_gdf.crs:
        roi_gdf = roi_gdf.to_crs(final_gdf.crs)
    
    # Get unique classes and assign colors
    unique_classes = final_gdf["class"].unique() if "class" in final_gdf.columns else []
    
    if len(unique_classes) == 0:
        raise ValueError("No classes found in GeoDataFrame. Cannot generate visualization.")
    
    # Create color mapping - use predefined colors or generate if not available
    color_map = {}
    # Default colors for unknown classes
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, class_name in enumerate(unique_classes):
        if class_name in CLASS_COLORS:
            color_map[class_name] = CLASS_COLORS[class_name]
        else:
            # Use default color palette for unknown classes
            color_map[class_name] = default_colors[i % len(default_colors)]
    
    # Add unclassified color if needed
    unclassified_color = "#FFFFFF"  # White for unclassified
    
    # Create figure with two subplots: map on left, legend on right
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], hspace=0.3, wspace=0.3)
    
    # Left subplot: Map
    ax_map = fig.add_subplot(gs[0, 0])
    
    # Plot ROI boundary
    roi_gdf.boundary.plot(ax=ax_map, color='black', linewidth=2, label='ROI Boundary')
    
    # Plot each class with its color
    if "class" in final_gdf.columns:
        for class_name in unique_classes:
            class_gdf = final_gdf[final_gdf["class"] == class_name]
            if not class_gdf.empty:
                color = color_map.get(class_name, "#808080")
                class_gdf.plot(ax=ax_map, color=color, edgecolor='none', alpha=0.7)
    
    # Set title and labels
    ax_map.set_title("SWMM-Focused Land Use Map", fontsize=16, fontweight='bold', pad=20)
    
    # Get bounds and set equal aspect
    bounds = roi_gdf.total_bounds
    ax_map.set_xlim(bounds[0], bounds[2])
    ax_map.set_ylim(bounds[1], bounds[3])
    ax_map.set_aspect('equal')
    
    # Add grid
    ax_map.grid(True, alpha=0.3, linestyle='--')
    
    # Add axis labels with CRS info
    crs_name = final_gdf.crs.name if final_gdf.crs else "Unknown"
    ax_map.set_xlabel(f"Easting (m) - {crs_name}", fontsize=10)
    ax_map.set_ylabel(f"Northing (m) - {crs_name}", fontsize=10)
    
    # Right subplot: Legend
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.axis('off')
    
    # Title for legend
    ax_legend.text(0.5, 0.98, "Land Use Categories", 
                   ha='center', va='top', fontsize=14, fontweight='bold',
                   transform=ax_legend.transAxes)
    
    # Calculate total classified area and unclassified
    total_classified_area = sum(s["Area (m²)"] for s in class_stats)
    unclassified_area = roi_area_m2 - total_classified_area
    unclassified_percentage = (unclassified_area / roi_area_m2) * 100 if roi_area_m2 > 0 else 0
    
    # Sort class_stats by area (descending) for legend
    sorted_stats = sorted(class_stats, key=lambda x: x["Area (m²)"], reverse=True)
    
    # Create legend entries
    y_pos = 0.95
    line_height = 0.025
    fontsize = 9
    
    for stat in sorted_stats:
        class_name = stat["Class"]
        percentage = stat["Percentage (%)"]
        color = color_map.get(class_name, "#808080")
        
        # Draw color box
        rect = Rectangle((0.02, y_pos - line_height/2), 0.08, line_height,
                        transform=ax_legend.transAxes, facecolor=color, edgecolor='black', linewidth=0.5)
        ax_legend.add_patch(rect)
        
        # Add text: Class name, percentage
        text = f"{class_name} ({percentage:.2f}%)"
        ax_legend.text(0.12, y_pos, text, transform=ax_legend.transAxes,
                      fontsize=fontsize, va='center', ha='left')
        
        y_pos -= line_height * 1.5
    
    # Add unclassified if exists
    if unclassified_area > 0:
        rect = Rectangle((0.02, y_pos - line_height/2), 0.08, line_height,
                       transform=ax_legend.transAxes, facecolor=unclassified_color, 
                       edgecolor='black', linewidth=0.5)
        ax_legend.add_patch(rect)
        text = f"Unclassified ({unclassified_percentage:.2f}%)"
        ax_legend.text(0.12, y_pos, text, transform=ax_legend.transAxes,
                      fontsize=fontsize, va='center', ha='left')
        y_pos -= line_height * 1.5
    
    # Add summary statistics at the bottom
    y_pos -= line_height * 2
    ax_legend.text(0.5, y_pos, f"Total Area: {roi_area_m2/10000:.2f} ha",
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   transform=ax_legend.transAxes)
    
    # Calculate weighted imperviousness (simplified - would need actual imperviousness values)
    # For now, just show total classified percentage
    classified_percentage = (total_classified_area / roi_area_m2) * 100 if roi_area_m2 > 0 else 0
    y_pos -= line_height * 1.5
    ax_legend.text(0.5, y_pos, f"Classified: {classified_percentage:.1f}%",
                   ha='center', va='top', fontsize=9,
                   transform=ax_legend.transAxes)
    
    # Save to BytesIO
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    img_buffer.seek(0)
    
    # Also save to file if path provided
    if output_path:
        plt.savefig(output_path, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    
    plt.close()
    
    return img_buffer


def main() -> None:
    """Main Streamlit app."""
    st.set_page_config(page_title="Satellite Segmentation (SAM3)", layout="wide")
    st.title("🛰️ Satellite Segmentation with SAM3")
    st.write("Segment objects in satellite imagery using text or box prompts.")
    
    # Add "Start New Segmentation" button at the top
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🆕 Start New Segmentation", type="secondary", help="Clear current session and start fresh"):
            reset_workflow()
            st.rerun()
    
    # HuggingFace Authentication Section
    st.sidebar.header("🔐 HuggingFace Authentication")
    is_authenticated = check_huggingface_auth()
    
    if is_authenticated:
        if "hf_username" in st.session_state:
            st.sidebar.success(f"✓ Authenticated as: {st.session_state['hf_username']}")
        else:
            try:
                from huggingface_hub import whoami
                user_info = whoami()
                username = user_info.get('name', 'Unknown user')
                st.session_state["hf_username"] = username
                st.sidebar.success(f"✓ Authenticated as: {username}")
            except Exception:
                st.sidebar.success("✓ Authenticated with HuggingFace")
    else:
        st.sidebar.warning("⚠️ HuggingFace authentication required")
        st.sidebar.write(
            "SAM3 is a gated model. You need to:\n"
            "1. Request access at: https://huggingface.co/facebook/sam3\n"
            "2. Get your token from: https://huggingface.co/settings/tokens\n"
            "3. Enter your token below"
        )
        
        hf_token = st.sidebar.text_input(
            "HuggingFace Token",
            type="password",
            help="Enter your HuggingFace access token (starts with 'hf_...')",
            key="hf_token_input"
        )
        
        if st.sidebar.button("Login to HuggingFace", key="hf_login_btn"):
            if hf_token:
                with st.spinner("Logging in..."):
                    if login_huggingface(hf_token):
                        st.sidebar.success("✓ Successfully logged in!")
                        st.rerun()
            else:
                st.sidebar.error("Please enter your HuggingFace token.")
    
    # Initialize session ID if not exists
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = f"session_{int(time.time())}"
    
    # Step 1: Choose imagery source (ROI download, upload GeoTIFF, or enter polygon coordinates)
    st.subheader("🗺️ Step 1: Choose Imagery Source")
    imagery_source = st.radio(
        "Select imagery source:",
        ["Draw ROI & download imagery", "Upload GeoTIFF", "Enter polygon coordinates"],
        key="imagery_source",
    )

    bbox = None
    if imagery_source == "Draw ROI & download imagery":
        bbox = capture_roi()
    elif imagery_source == "Enter polygon coordinates":
        bbox = capture_roi_from_coordinates()
    else:
        st.info("Upload a GeoTIFF to use as SAM input (skips ROI download).")
        uploaded_tif = st.file_uploader(
            "Upload GeoTIFF",
            type=["tif", "tiff"],
            key="uploaded_geotiff",
            accept_multiple_files=False,
            help="Provide a georeferenced GeoTIFF. If it has nodata outside your area, SAM will still process the rectangular extent but nodata areas will be ignored.",
        )
        if uploaded_tif is not None:
            try:
                ASSETS["image"].write_bytes(uploaded_tif.getbuffer())
                st.session_state["imagery_downloaded"] = True
                st.session_state["image_path"] = str(ASSETS["image"])
                # Reset downstream state for the new image
                st.session_state["sam3_initialized"] = None
                st.session_state["masks_generated"] = False
                st.session_state["geojson_processed"] = False
                st.session_state["show_mask_preview"] = False
                st.success(f"✓ Uploaded GeoTIFF saved as {ASSETS['image'].name}")
            except Exception as e:
                st.error(f"Failed to save uploaded GeoTIFF: {e}")

    # Show workflow status
    if bbox or st.session_state.get("imagery_downloaded"):
        st.sidebar.divider()
        st.sidebar.subheader("📊 Workflow Status")
        if st.session_state.get("imagery_downloaded"):
            st.sidebar.success("✓ Imagery ready")
        if st.session_state.get("sam3_initialized"):
            st.sidebar.success("✓ SAM3 initialized")
        if st.session_state.get("masks_generated"):
            st.sidebar.success("✓ Masks generated")
        if st.session_state.get("geojson_processed"):
            st.sidebar.success("✓ GeoJSON processed")
    
    # Step 2: Download imagery (if using ROI) and initialize SAM3
    imagery_ready = st.session_state.get("imagery_downloaded", False)
    if is_authenticated and (imagery_source == "Upload GeoTIFF" or bbox):
        # Show download button for both "Draw ROI" and "Upload polygon coordinates"
        needs_download = (imagery_source == "Draw ROI & download imagery" or imagery_source == "Enter polygon coordinates")
        if needs_download and bbox and not imagery_ready:
            st.subheader("📥 Step 2: Download Satellite Imagery")
            st.write("Click the button below to download satellite imagery for your ROI.")
            if st.button("📥 Download Satellite Imagery", type="primary", key="download_btn"):
                with st.spinner("Downloading satellite imagery..."):
                    try:
                        download_imagery(bbox, ASSETS["image"])
                        
                        # After downloading, convert ROI to image CRS (single conversion)
                        # tms_to_geotiff typically outputs Web Mercator (EPSG:3857)
                        roi_geom = st.session_state.get("roi_geojson")
                        if roi_geom:
                            try:
                                import rasterio
                                with rasterio.open(ASSETS["image"]) as src:
                                    image_crs = src.crs if src.crs else "EPSG:3857"  # Default to Web Mercator
                                
                                # Convert ROI from original CRS to image CRS (single conversion)
                                from shapely.geometry import shape
                                roi_shape = shape(roi_geom)
                                roi_crs_original = st.session_state.get("roi_crs", "EPSG:4326")
                                roi_gdf_original = gpd.GeoDataFrame(geometry=[roi_shape], crs=roi_crs_original)
                                
                                if str(image_crs) != roi_crs_original:
                                    roi_gdf_image_crs = roi_gdf_original.to_crs(image_crs)
                                    roi_geom_image_crs = roi_gdf_image_crs.geometry.iloc[0]
                                    roi_geojson_image_crs = {
                                        "type": "Polygon",
                                        "coordinates": [[[float(x), float(y)] for x, y in roi_geom_image_crs.exterior.coords]]
                                    }
                                    st.session_state["roi_geojson"] = roi_geojson_image_crs
                                    st.session_state["roi_crs"] = str(image_crs)
                            except Exception as e:
                                # If conversion fails, keep original (will convert during clipping)
                                st.warning(f"Could not convert ROI to image CRS: {e}. Will convert during clipping.")
                        
                        st.session_state["imagery_downloaded"] = True
                        st.session_state["image_path"] = str(ASSETS["image"])
                        st.success("✓ Imagery downloaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to download imagery: {e}")
                        st.stop()
        # Step 3: Choose prompt type
        if st.session_state.get("imagery_downloaded"):
            st.divider()
            is_adding_class = st.session_state.get("adding_another_class", False)
            if is_adding_class:
                st.subheader("➕ Add Another Class: Choose Segmentation Method")
                st.info("💡 You're adding another class to your multiclass GeoJSON. Generate masks for the new class.")
            else:
                st.subheader("🎯 Step 2: Choose Segmentation Method")
            
            prompt_type = st.radio(
                "Select segmentation method:",
                ["Text Prompt", "Box Prompt"],
                key="prompt_type"
            )
            
            # Initialize SAM3
            if st.session_state.get("sam3_initialized") is None:
                try:
                    with st.spinner("Initializing SAM3..."):
                        sam3 = initialize_sam3(ASSETS["image"])
                        st.session_state["sam3"] = sam3
                        st.session_state["sam3_initialized"] = True
                        st.success("✓ SAM3 initialized successfully!")
                except Exception as e:
                    st.error(f"Failed to initialize SAM3: {e}")
                    st.stop()
            
            # Text Prompt
            if prompt_type == "Text Prompt":
                st.subheader("📝 Text Prompt Segmentation")
                
                # Track previous dropdown selection to detect changes
                if "prev_dropdown_selection" not in st.session_state:
                    st.session_state["prev_dropdown_selection"] = "-- Custom Prompt --"
                
                # Dropdown for predefined prompts
                st.write("**Select a predefined prompt or enter a custom one:**")
                selected_prompt = st.selectbox(
                    "Choose from predefined prompts:",
                    options=["-- Custom Prompt --"] + PREDEFINED_PROMPTS,
                    key="predefined_prompt_dropdown",
                    help="Select a prompt from the list to auto-fill the text input, or choose 'Custom Prompt' to type your own"
                )
                
                # Auto-fill text prompt when dropdown selection changes to a predefined prompt
                if selected_prompt != "-- Custom Prompt --":
                    # If dropdown changed to a predefined prompt, update text input
                    if st.session_state["prev_dropdown_selection"] != selected_prompt:
                        st.session_state["text_prompt"] = selected_prompt
                    st.session_state["prev_dropdown_selection"] = selected_prompt
                    
                    text_prompt = st.text_input(
                        "Text prompt (auto-filled from selection, can be edited):",
                        value=st.session_state.get("text_prompt", selected_prompt),
                        key="text_prompt",
                        help="The prompt is auto-filled from your selection above. You can edit it if needed."
                    )
                else:
                    # Update tracking when switching to custom (only if changed)
                    if st.session_state["prev_dropdown_selection"] != "-- Custom Prompt --":
                        st.session_state["prev_dropdown_selection"] = "-- Custom Prompt --"
                    # Custom prompt - keep existing value or empty
                    text_prompt = st.text_input(
                        "Enter text prompt (e.g., 'road', 'building', 'tree', 'water'):",
                        value=st.session_state.get("text_prompt", ""),
                        key="text_prompt",
                        help="Describe what you want to segment"
                    )
                
                # Show confidence threshold option (available for all runs)
                confidence_threshold = st.slider(
                    "SAM Confidence Threshold (optional)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    key="confidence_threshold",
                    help="Higher values = more confident predictions only. Lower values = include less confident predictions. Default: 0.5"
                )
                
                if st.button("🚀 Generate Masks", type="primary", key="generate_text"):
                    if text_prompt:
                        with st.spinner(f"Generating masks for '{text_prompt}'..."):
                            try:
                                sam3 = st.session_state["sam3"]
                                # Set confidence threshold (always available now)
                                sam3.set_confidence_threshold(confidence_threshold)
                                st.session_state["last_confidence_threshold"] = confidence_threshold
                                
                                # Generate masks
                                sam3.generate_masks(prompt=text_prompt)
                                
                                # Try to save masks - check if any were generated
                                try:
                                    sam3.save_masks(str(ASSETS["mask"]), unique=True)
                                except ValueError as ve:
                                    if "No masks found" in str(ve):
                                        st.warning(
                                            f"No masks were generated for the prompt '{text_prompt}'. "
                                            "This could happen if:\n"
                                            "- The prompt doesn't match any features in the image\n"
                                            "- The confidence threshold is too high (try lowering it)\n"
                                            "- The ROI area is too small or doesn't contain the requested features\n\n"
                                            "Please try:\n"
                                            "- A different text prompt\n"
                                            "- Lowering the confidence threshold (try 0.3 or lower)\n"
                                            "- Checking if your ROI contains the features you're looking for"
                                        )
                                        st.session_state["masks_generated"] = False
                                    else:
                                        raise
                                else:
                                    # Masks were saved successfully
                                    st.session_state["masks_generated"] = True
                                    st.session_state["geojson_processed"] = False  # Reset processing flag
                                    
                                    # Show preview immediately (for both first run and adding classes)
                                    st.session_state["show_mask_preview"] = True
                                    st.session_state["preview_prompt"] = text_prompt
                                    
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Segmentation failed: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                    else:
                        st.warning("Please enter a text prompt.")
            
            # Box Prompt
            else:
                st.subheader("📦 Box Prompt Segmentation")
                boxes = capture_box_prompts()
                
                # Show confidence threshold option (available for all runs)
                confidence_threshold = st.slider(
                    "SAM Confidence Threshold (optional)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    key="confidence_threshold_box",
                    help="Higher values = more confident predictions only. Lower values = include less confident predictions. Default: 0.5"
                )
                
                if st.button("🚀 Generate Masks from Boxes", type="primary", key="generate_boxes"):
                    if boxes:
                        with st.spinner(f"Generating masks for {len(boxes)} box(es)..."):
                            try:
                                sam3 = st.session_state["sam3"]
                                # Set confidence threshold (always available now)
                                sam3.set_confidence_threshold(confidence_threshold)
                                st.session_state["last_confidence_threshold"] = confidence_threshold
                                
                                box_labels = [True] * len(boxes)
                                # Generate masks
                                sam3.generate_masks_by_boxes(
                                    boxes=boxes,
                                    box_labels=box_labels,
                                    box_crs="EPSG:4326",
                                )
                                
                                # Try to save masks - check if any were generated
                                try:
                                    sam3.save_masks(str(ASSETS["mask"]), unique=True)
                                except ValueError as ve:
                                    if "No masks found" in str(ve):
                                        st.warning(
                                            f"No masks were generated for the {len(boxes)} box(es). "
                                            "This could happen if:\n"
                                            "- The boxes don't contain any segmentable features\n"
                                            "- The confidence threshold is too high (try lowering it)\n"
                                            "- The boxes are too small or in empty areas\n\n"
                                            "Please try:\n"
                                            "- Drawing boxes around more prominent features\n"
                                            "- Lowering the confidence threshold (try 0.3 or lower)\n"
                                            "- Making sure your boxes are within the ROI area"
                                        )
                                        st.session_state["masks_generated"] = False
                                    else:
                                        raise
                                else:
                                    # Masks were saved successfully
                                    st.session_state["masks_generated"] = True
                                    st.session_state["geojson_processed"] = False  # Reset processing flag
                                    
                                    # Show preview immediately (for both first run and adding classes)
                                    st.session_state["show_mask_preview"] = True
                                    st.session_state["preview_prompt"] = f"{len(boxes)} box(es)"
                                    
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Segmentation failed: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                    else:
                        st.warning("Please draw at least one rectangle on the map.")
            
            # Step 3.5: Preview masks (only when adding another class)
            if st.session_state.get("masks_generated") and st.session_state.get("show_mask_preview", False):
                st.divider()
                st.subheader("👁️ Preview: Review Segmentation Results")
                st.info("Review the segmentation masks below. Click **Accept** to proceed with class assignment, or **Discard** to re-run with different settings.")
                
                # Show preview map
                if ASSETS["mask"].exists() and ASSETS["image"].exists():
                    preview_prompt = st.session_state.get("preview_prompt", "segmentation")
                    st.write(f"**Prompt**: `{preview_prompt}`")
                    
                    # Display simple preview map
                    show_mask_preview(ASSETS["mask"], ASSETS["image"])
                    
                    # Accept/Discard buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Accept & Continue to Class Assignment", type="primary", key="accept_mask_preview"):
                            # Clear preview flag and proceed to class assignment
                            st.session_state["show_mask_preview"] = False
                            st.session_state["mask_preview_accepted"] = True
                            st.rerun()
                    
                    with col2:
                        if st.button("❌ Discard & Re-run Segmentation", type="secondary", key="discard_mask_preview"):
                            # Reset masks and allow re-segmentation
                            st.session_state["show_mask_preview"] = False
                            st.session_state["masks_generated"] = False
                            st.session_state["mask_preview_accepted"] = False
                            st.info("💡 Masks discarded. You can now re-run segmentation with a different confidence threshold if needed.")
                            st.rerun()
                else:
                    st.warning("Mask or image file not found. Cannot display preview.")
                    st.session_state["show_mask_preview"] = False
            
            # Step 4: Process masks to GeoJSON and assign class
            if st.session_state.get("masks_generated") and not st.session_state.get("show_mask_preview", False):
                st.divider()
                is_adding_class = st.session_state.get("adding_another_class", False)
                if is_adding_class:
                    st.subheader("⚙️ Add Class: Process Results & Assign Class Name")
                else:
                    st.subheader("⚙️ Step 3: Process Results & Assign Class")
                
                # Check if already processed
                if not st.session_state.get("geojson_processed"):
                    is_adding_class = st.session_state.get("adding_another_class", False)
                    
                    # Normal processing mode (preview is now shown before this step)
                    st.info("💡 Masks have been generated! Assign a class name and convert to GeoJSON.")
                    
                    # Class assignment input
                    class_name = st.text_input(
                            "Assign Class",
                            key="class_name_input",
                            placeholder="e.g., building, road, water, tree",
                            help="Enter a class name for the segmented polygons (e.g., 'building', 'road', 'water')"
                    )
                    
                    # Simplification method + parameter input
                    simplification_method_label_to_value = {
                        "Douglas-Peucker": "douglas-peucker",
                        "Visvalingam": "visvalingam",
                        "Grid Snapping": "grid-snapping",
                    }
                    simplification_method_label = st.selectbox(
                        "Simplification Method",
                        options=list(simplification_method_label_to_value.keys()),
                        index=0,
                        help=(
                            "Choose how polygons should be simplified:\n"
                            "- Douglas-Peucker: classic line simplification (good general-purpose)\n"
                            "- Visvalingam: preserves visual shape better for some geometries\n"
                            "- Grid Snapping: snaps vertices to a grid (useful for very noisy data)"
                        ),
                        key="simplification_method",
                    )
                    simplification_method = simplification_method_label_to_value[simplification_method_label]

                    # Method-specific parameter
                    if simplification_method == "douglas-peucker":
                        simplification_value = st.slider(
                            "Simplification Threshold (Douglas-Peucker)",
                            min_value=0.1,
                            max_value=10.0,
                            value=1.0,
                            step=0.1,
                            key="simplification_threshold_douglas",
                            help="Higher values = more simplification (fewer vertices). Lower values = less simplification (more detail).",
                        )
                        simplification_kwargs = {
                            "douglas_tolerance": simplification_value,
                            "visvalingam_threshold": 1.0,
                            "grid_size": 1.0,
                        }
                    elif simplification_method == "visvalingam":
                        simplification_value = st.slider(
                            "Visvalingam Threshold",
                            min_value=0.1,
                            max_value=10.0,
                            value=1.0,
                            step=0.1,
                            key="simplification_threshold_visvalingam",
                            help="Higher values = more simplification (fewer vertices). Lower values = less simplification (more detail).",
                        )
                        simplification_kwargs = {
                            "douglas_tolerance": 1.0,
                            "visvalingam_threshold": simplification_value,
                            "grid_size": 1.0,
                        }
                    else:  # grid-snapping
                        simplification_value = st.slider(
                            "Grid Size (for snapping)",
                            min_value=0.1,
                            max_value=50.0,
                            value=1.0,
                            step=0.1,
                            key="simplification_threshold_grid",
                            help="Larger grid size = stronger snapping to a coarse grid. Use smaller values to preserve more detail.",
                        )
                        simplification_kwargs = {
                            "douglas_tolerance": 1.0,
                            "visvalingam_threshold": 1.0,
                            "grid_size": simplification_value,
                        }
                    
                    if st.button("🔄 Convert Masks to GeoJSON & Assign Class", type="primary", key="process_masks"):
                            if not class_name or not class_name.strip():
                                st.warning("⚠️ Please enter a class name before processing.")
                            else:
                                with st.spinner("Processing masks and assigning class..."):
                                    try:
                                        # Convert raster to vector
                                        raster_to_gpkg(str(ASSETS["mask"]), str(ASSETS["vector"]), simplify_tolerance=None)
                                        
                                        # Load and assign class column
                                        gdf = gpd.read_file(ASSETS["vector"])
                                        class_name_clean = class_name.strip()
                                        gdf["class"] = class_name_clean
                                        
                                        # Clip to ROI polygon if available
                                        roi_geom = st.session_state.get("roi_geojson")
                                        roi_crs = st.session_state.get("roi_crs", "EPSG:4326")
                                        if roi_geom:
                                            from shapely.geometry import shape
                                            roi_shape = shape(roi_geom)
                                            # ROI should already be in image CRS (converted after download)
                                            roi_gdf = gpd.GeoDataFrame(geometry=[roi_shape], crs=roi_crs)
                                            
                                            # Ensure vector has CRS (should match image CRS)
                                            if gdf.crs is None:
                                                try:
                                                    import rasterio
                                                    with rasterio.open(ASSETS["image"]) as src:
                                                        if src.crs:
                                                            gdf.set_crs(src.crs, inplace=True)
                                                        else:
                                                            gdf.set_crs(roi_crs, inplace=True)
                                                except Exception:
                                                    gdf.set_crs(roi_crs, inplace=True)
                                            
                                            # Convert only if CRS differs (shouldn't happen if converted after download)
                                            if gdf.crs != roi_gdf.crs:
                                                roi_gdf = roi_gdf.to_crs(gdf.crs)
                                            
                                            gdf = gpd.clip(gdf, roi_gdf)
                                            gdf = gdf[gdf.geometry.notna()]
                                        
                                        gdf.to_file(ASSETS["classified"], driver="GeoJSON")
                                        
                                        if is_adding_class and ASSETS["final"].exists():
                                            # Adding another class - append to existing GeoJSON
                                            # Read existing GeoJSON FIRST before overwriting anything
                                            existing_gdf = gpd.read_file(ASSETS["final"])
                                            
                                            # Simplify new polygons to temporary file first
                                            temp_new_file = ASSETS["final"].parent / "temp_new_class.geojson"
                                            simplify_polygons(
                                                ASSETS["classified"],
                                                temp_new_file,
                                                method=simplification_method,
                                                douglas_tolerance=simplification_kwargs["douglas_tolerance"],
                                                visvalingam_threshold=simplification_kwargs["visvalingam_threshold"],
                                                grid_size=simplification_kwargs["grid_size"],
                                            )
                                            
                                            # Load the new simplified polygons
                                            new_gdf = gpd.read_file(temp_new_file)
                                            
                                            # Combine both GeoDataFrames
                                            combined_gdf = gpd.GeoDataFrame(
                                                pd.concat([existing_gdf, new_gdf], ignore_index=True),
                                                crs=existing_gdf.crs
                                            )
                                            
                                            # Save combined GeoJSON
                                            combined_gdf.to_file(ASSETS["final"], driver="GeoJSON")
                                            
                                            # Clean up temp file
                                            if temp_new_file.exists():
                                                temp_new_file.unlink()
                                            
                                            # Update classes list
                                            if "classes_added" not in st.session_state:
                                                st.session_state["classes_added"] = []
                                            if class_name_clean not in st.session_state["classes_added"]:
                                                st.session_state["classes_added"].append(class_name_clean)
                                            
                                            st.session_state["final_geojson"] = str(ASSETS["final"])
                                            st.session_state["geojson_processed"] = True
                                            st.session_state["adding_another_class"] = False
                                            st.session_state["show_mask_preview"] = False  # Clear preview flag
                                            
                                            # Store success message in session state so it persists after rerun
                                            st.session_state["last_success_message"] = f"✓ Class '{class_name_clean}' added! {len(new_gdf)} polygons appended. Total: {len(combined_gdf)} polygons across {len(st.session_state['classes_added'])} classes."
                                            st.rerun()
                                        else:
                                            # First class - simplify directly to final file
                                            simplify_polygons(
                                                ASSETS["classified"],
                                                ASSETS["final"],
                                                method=simplification_method,
                                                douglas_tolerance=simplification_kwargs["douglas_tolerance"],
                                                visvalingam_threshold=simplification_kwargs["visvalingam_threshold"],
                                                grid_size=simplification_kwargs["grid_size"],
                                            )
                                            
                                            # Update classes list
                                            if "classes_added" not in st.session_state:
                                                st.session_state["classes_added"] = []
                                            if class_name_clean not in st.session_state["classes_added"]:
                                                st.session_state["classes_added"].append(class_name_clean)
                                            
                                            st.session_state["final_geojson"] = str(ASSETS["final"])
                                            st.session_state["geojson_processed"] = True
                                            st.success(f"✓ Class '{class_name_clean}' assigned! {len(gdf)} polygons saved.")
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"Processing failed: {e}")
                                        import traceback
                                        st.code(traceback.format_exc())
                else:
                    # Show current classes
                    classes_added = st.session_state.get("classes_added", [])
                    if classes_added:
                        st.success(f"✓ Current classes in GeoJSON: {', '.join(classes_added)}")
                    else:
                        st.success("✓ Masks have been processed to GeoJSON. Results are shown below.")
            
            # Step 5: Display results in map component
            # Always check the current file (don't rely on cached path)
            final_path = None
            if ASSETS["final"].exists():
                final_path = str(ASSETS["final"])
                # Set final_geojson if file exists and we have processed GeoJSON
                if st.session_state.get("geojson_processed"):
                    st.session_state["final_geojson"] = final_path
            
            # Show map if we have processed GeoJSON in this session OR if file exists with classes
            # This ensures map shows even after rerun
            has_processed = st.session_state.get("geojson_processed", False)
            has_classes = len(st.session_state.get("classes_added", [])) > 0
            
            if final_path and Path(final_path).exists() and (has_processed or has_classes):
                # Show success message if it exists (persists after rerun)
                if "last_success_message" in st.session_state:
                    st.success(st.session_state["last_success_message"])
                    # Clear it after showing once
                    del st.session_state["last_success_message"]
                
                st.divider()
                st.subheader("🗺️ Step 4: View & Edit Results")
                
                # Show file info and reload button
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(
                        "Your segmented polygons are displayed below. You can:\n"
                        "- **Edit** polygons by clicking and dragging vertices\n"
                        "- **Add** new polygons using the drawing tools\n"
                        "- **Delete** polygons using the trash icon\n"
                        "- **Export** your edited GeoJSON using the button on the map"
                    )
                with col2:
                    file_path = Path(final_path)
                    if file_path.exists():
                        file_mtime = file_path.stat().st_mtime
                        from datetime import datetime
                        mod_time = datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d %H:%M:%S")
                        st.caption(f"File: `{file_path.name}`")
                        st.caption(f"Modified: {mod_time}")
                        st.caption(f"Path: `{file_path}`")
                        if st.button("🔄 Reload Map", key="reload_map_btn", help="Reload the GeoJSON file from disk"):
                            # Clear map editor cache to force reload
                            session_id = st.session_state.get("session_id", "default")
                            map_key = f"map_editor_{session_id}_{final_path}"
                            if map_key in st.session_state:
                                del st.session_state[map_key]
                            if f"{map_key}_mtime" in st.session_state:
                                del st.session_state[f"{map_key}_mtime"]
                            st.rerun()
                
                if MAP_COMPONENT_AVAILABLE and GeoJSONMapEditor is not None:
                    # Load available classes for the dropdown
                    try:
                        current_gdf = gpd.read_file(final_path)
                        if "class" in current_gdf.columns:
                            available_classes = sorted(current_gdf["class"].dropna().unique().tolist())
                        else:
                            available_classes = st.session_state.get("classes_added", [])
                    except Exception:
                        available_classes = st.session_state.get("classes_added", [])
                    
                    # Use session-specific key to avoid conflicts between sessions
                    session_id = st.session_state.get("session_id", "default")
                    map_key = f"map_editor_{session_id}_{final_path}"
                    file_path = Path(final_path)
                    
                    # Check if file exists and get its modification time
                    file_mtime = file_path.stat().st_mtime if file_path.exists() else 0
                    
                    # Check if we need to create or update the map editor
                    # Only recreate if it doesn't exist or if the file has changed
                    last_mtime = st.session_state.get(f"{map_key}_mtime", 0)
                    file_changed = file_mtime > last_mtime + 0.1  # Small buffer to avoid rapid reloads
                    
                    # Track if this is just a dropdown change (not a file change)
                    dropdown_only_change = False
                    if map_key in st.session_state and not file_changed:
                        # Check if only the dropdown changed by comparing with last selected class
                        last_selected_class = st.session_state.get(f"{map_key}_last_selected_class")
                        current_selected_class = st.session_state.get("map_drawing_class")
                        if last_selected_class is not None and current_selected_class != last_selected_class:
                            dropdown_only_change = True
                    
                    # Class selection - use custom HTML dropdown that doesn't trigger rerun
                    # Initialize selected_class from session state or use first available
                    if "map_drawing_class" not in st.session_state and available_classes:
                        st.session_state["map_drawing_class"] = available_classes[0]
                    
                    selected_class = st.session_state.get("map_drawing_class")
                    
                    if available_classes:
                        col1, col2 = st.columns([2, 3])
                        with col1:
                            st.write("**🏷️ Class for New Polygons**")
                            # Create custom HTML dropdown that updates via JavaScript only
                            options_html = "".join([f'<option value="{cls}" {"selected" if cls == selected_class else ""}>{cls}</option>' for cls in available_classes])
                            st.markdown(f"""
                            <select id="class_selector_{session_id}" style="width: 100%; padding: 8px; font-size: 14px; border-radius: 4px; border: 1px solid #ccc;">
                                {options_html}
                            </select>
                            <script>
                            (function() {{
                                var selector = document.getElementById('class_selector_{session_id}');
                                if (selector) {{
                                    // Function to update default class in map
                                    function updateMapDefaultClass(selectedClass) {{
                                        console.log('Attempting to update default class to:', selectedClass);
                                        var updated = false;
                                        var iframes = parent.document.getElementsByTagName('iframe');
                                        
                                        for (var i = 0; i < iframes.length; i++) {{
                                            try {{
                                                var iframe = iframes[i];
                                                var iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                                                if (iframeDoc && iframeDoc.getElementById('map')) {{
                                                    // Try direct access first
                                                    if (iframe.contentWindow && iframe.contentWindow.updateDefaultClass) {{
                                                        iframe.contentWindow.updateDefaultClass(selectedClass);
                                                        console.log('Default class updated via direct access:', selectedClass);
                                                        updated = true;
                                                    }}
                                                }}
                                            }} catch(e) {{
                                                // Cross-origin - try postMessage instead
                                                try {{
                                                    iframe.contentWindow.postMessage({{
                                                        type: 'updateDefaultClass',
                                                        class: selectedClass
                                                    }}, '*');
                                                    console.log('Default class update sent via postMessage:', selectedClass);
                                                    updated = true;
                                                }} catch(e2) {{
                                                    console.log('Could not update default class:', e2);
                                                }}
                                            }}
                                        }}
                                        
                                        if (!updated) {{
                                            console.warn('Could not find map iframe to update default class');
                                        }}
                                    }}
                                    
                                    // Update immediately on page load to set initial value
                                    setTimeout(function() {{
                                        var initialClass = selector.value;
                                        updateMapDefaultClass(initialClass);
                                    }}, 800);
                                    
                                    // Update on change
                                    selector.addEventListener('change', function() {{
                                        var selectedClass = this.value;
                                        console.log('Class selector changed to:', selectedClass);
                                        updateMapDefaultClass(selectedClass);
                                    }});
                                }}
                            }})();
                            </script>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.write("")  # Spacing
                            st.caption("💡 Select a class from the dropdown above, then draw polygons on the map. They will be assigned the selected class.")
                    
                    # Create or update map editor
                    try:
                        if map_key not in st.session_state or file_changed:
                            # Create new map editor only if it doesn't exist or file changed
                            map_editor = GeoJSONMapEditor(
                                geojson_path=file_path,
                                height=700,
                                default_class=selected_class
                            )
                            st.session_state[map_key] = map_editor
                            st.session_state[f"{map_key}_mtime"] = file_mtime
                        else:
                            # Map editor exists - just update the default class property
                            # (but don't recreate, to preserve map state)
                            map_editor = st.session_state[map_key]
                            map_editor.default_class = selected_class
                    except Exception as e:
                        st.error(f"Failed to create map: {e}")
                        st.stop()
                    
                    # Render the map with stable key - this won't change when dropdown changes
                    # because the dropdown is now custom HTML that doesn't trigger rerun
                    render_key = f"geojson_map_{session_id}_{file_path.name}"
                    st.session_state[map_key].render(key=render_key, force_reload=file_changed, auto_save=False)
                    
                    st.info("💡 Click the 'Export GeoJSON' button on the map (top-right) to download your edited polygons.")
                else:
                    st.warning("Custom map component not available. Install required dependencies.")
                
                # Class management and download section
                st.divider()
                st.subheader("📥 Download & Manage Classes")
                
                # Load current GeoJSON to get actual classes
                try:
                    current_gdf = gpd.read_file(final_path)
                    if "class" in current_gdf.columns:
                        available_classes = sorted(current_gdf["class"].dropna().unique().tolist())
                    else:
                        available_classes = st.session_state.get("classes_added", [])
                except Exception:
                    available_classes = st.session_state.get("classes_added", [])
                
                # Show classes summary
                if available_classes:
                    st.info(f"📊 **Multiclass GeoJSON**: Contains {len(available_classes)} class(es): {', '.join(available_classes)}")
                    
                    # Class deletion section
                    st.subheader("🗑️ Delete a Class")
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        class_to_delete = st.selectbox(
                            "Select class to delete:",
                            options=available_classes,
                            key="class_to_delete_select",
                            help="Select a class to remove all its polygons from the multiclass GeoJSON"
                        )
                    with col2:
                        st.write("")  # Spacing
                        st.write("")  # Spacing
                        if st.button("🗑️ Delete Class", type="secondary", key="delete_class_btn"):
                            try:
                                # Load current GeoJSON
                                gdf_to_filter = gpd.read_file(final_path)
                                
                                # Count polygons before deletion
                                before_count = len(gdf_to_filter)
                                class_count = len(gdf_to_filter[gdf_to_filter["class"] == class_to_delete])
                                
                                # Filter out the selected class
                                filtered_gdf = gdf_to_filter[gdf_to_filter["class"] != class_to_delete]
                                
                                # Save filtered GeoJSON
                                filtered_gdf.to_file(final_path, driver="GeoJSON")
                                
                                # Update classes_added list
                                if "classes_added" in st.session_state:
                                    if class_to_delete in st.session_state["classes_added"]:
                                        st.session_state["classes_added"].remove(class_to_delete)
                                
                                # Clear map cache to force reload
                                session_id = st.session_state.get("session_id", "default")
                                map_key = f"map_editor_{session_id}_{final_path}"
                                if map_key in st.session_state:
                                    del st.session_state[map_key]
                                if f"{map_key}_mtime" in st.session_state:
                                    del st.session_state[f"{map_key}_mtime"]
                                
                                st.success(f"✓ Class '{class_to_delete}' deleted! Removed {class_count} polygons. {len(filtered_gdf)} polygons remaining.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to delete class: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                else:
                    st.warning("No classes found in GeoJSON.")
                
                # Download section
                st.subheader("📥 Download Results")
                with open(final_path, "r", encoding="utf-8") as f:
                    geojson_data = f.read()
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.download_button(
                        label="📥 Download Multiclass GeoJSON",
                        data=geojson_data,
                        file_name="multiclass_segmentation.geojson",
                        mime="application/json",
                        key="download_geojson"
                    )
                with col2:
                    if st.button("➕ Add Another Class of Objects", type="primary", key="add_another_class"):
                        # Set flag to indicate we're adding another class
                        st.session_state["adding_another_class"] = True
                        # Reset processing flags but keep the workflow state
                        st.session_state["masks_generated"] = False
                        st.session_state["geojson_processed"] = False
                        st.session_state["preview_mode"] = False  # Reset preview mode
                        st.session_state["sam3_initialized"] = True  # Keep SAM3 initialized
                        st.session_state["imagery_downloaded"] = True  # Keep imagery
                        st.rerun()
                
                st.divider()
                
                # Upload edited GeoJSON section for statistics
                st.subheader("📊 View Statistics for Edited GeoJSON")
                st.write("To see statistics for your edited polygons:")
                st.write("1. Export your edited GeoJSON from the map (use the 'Export GeoJSON' button on the map)")
                st.write("2. Upload the exported file below to calculate statistics")
                
                uploaded_edited_geojson = st.file_uploader(
                    "Upload your edited GeoJSON file",
                    type=["geojson", "json"],
                    key=f"upload_edited_for_stats_{session_id}",
                    help="Upload the GeoJSON file you exported from the map widget after making your edits"
                )
                
                # Statistics section - only shown if edited GeoJSON is uploaded
                if uploaded_edited_geojson is not None:
                    # Save uploaded file temporarily
                    temp_dir = Path("temp")
                    temp_dir.mkdir(exist_ok=True)
                    temp_uploaded = temp_dir / f"uploaded_edited_{session_id}.geojson"
                    
                    with open(temp_uploaded, "wb") as f:
                        f.write(uploaded_edited_geojson.getbuffer())
                    
                    # Store path in session state
                    st.session_state[f"edited_geojson_for_stats_{session_id}"] = str(temp_uploaded)
                    
                    st.success("✓ Edited GeoJSON uploaded! Calculating statistics...")
                    
                    # Statistics section
                    st.divider()
                    st.subheader("📊 Class Statistics (Based on Edited GeoJSON)")
                    try:
                        # Read uploaded edited GeoJSON
                        final_gdf = gpd.read_file(temp_uploaded)
                        
                        # Get ROI polygon
                        roi_geojson = st.session_state.get("roi_geojson")
                        roi_crs = st.session_state.get("roi_crs", "EPSG:4326")
                        
                        if roi_geojson and "class" in final_gdf.columns:
                            # Convert ROI to GeoDataFrame
                            from shapely.geometry import shape
                            roi_shape = shape(roi_geojson)
                            roi_gdf = gpd.GeoDataFrame(geometry=[roi_shape], crs=roi_crs)
                            
                            # Always use UTM for accurate area calculation (NOT image CRS which might be Web Mercator)
                            # Convert ROI to WGS84 first to get centroid for UTM zone calculation
                            roi_wgs84 = roi_gdf.to_crs("EPSG:4326")
                            centroid = roi_wgs84.geometry.iloc[0].centroid
                            lon, lat = centroid.x, centroid.y
                            
                            # Determine appropriate UTM zone based on centroid
                            utm_zone = int((lon + 180) / 6) + 1
                            # Use UTM for accurate area calculation (Web Mercator distorts areas!)
                            if lat >= 0:
                                area_crs = f"EPSG:{32600 + utm_zone}"  # UTM North
                            else:
                                area_crs = f"EPSG:{32700 + utm_zone}"  # UTM South
                            
                            # Convert both ROI and final GeoJSON to area calculation CRS
                            roi_area_crs = roi_gdf.to_crs(area_crs)
                            final_area_crs = final_gdf.to_crs(area_crs)
                            
                            # Calculate total ROI area
                            roi_area_m2 = roi_area_crs.geometry.iloc[0].area
                            roi_area_ha = roi_area_m2 / 10000  # Convert to hectares
                            
                            # Calculate area for each class
                            class_stats = []
                            for class_name in final_gdf["class"].unique():
                                class_gdf = final_area_crs[final_area_crs["class"] == class_name]
                                class_area_m2 = class_gdf.geometry.area.sum()
                                class_area_ha = class_area_m2 / 10000
                                percentage = (class_area_m2 / roi_area_m2) * 100 if roi_area_m2 > 0 else 0
                                
                                class_stats.append({
                                    "Class": class_name,
                                    "Area (m²)": class_area_m2,
                                    "Area (ha)": class_area_ha,
                                    "Percentage (%)": percentage
                                })
                            
                            # Sort by area (descending)
                            class_stats.sort(key=lambda x: x["Area (m²)"], reverse=True)
                            
                            # Display statistics
                            st.write(f"**Total ROI Area:** {roi_area_ha:.4f} ha ({roi_area_m2:,.2f} m²)")
                            st.write("")
                            
                            # Create DataFrame for nice display
                            stats_df = pd.DataFrame(class_stats)
                            
                            # Format the DataFrame for display
                            display_df = stats_df.copy()
                            display_df["Area (m²)"] = display_df["Area (m²)"].apply(lambda x: f"{x:,.2f}")
                            display_df["Area (ha)"] = display_df["Area (ha)"].apply(lambda x: f"{x:.4f}")
                            display_df["Percentage (%)"] = display_df["Percentage (%)"].apply(lambda x: f"{x:.2f}%")
                            
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                            
                            # Summary
                            total_classified_area = sum(s["Area (m²)"] for s in class_stats)
                            unclassified_area = roi_area_m2 - total_classified_area
                            unclassified_percentage = (unclassified_area / roi_area_m2) * 100 if roi_area_m2 > 0 else 0
                            
                            if unclassified_area > 0:
                                st.info(
                                    f"**Unclassified area:** {unclassified_area/10000:.4f} ha ({unclassified_area:,.2f} m²) - "
                                    f"{unclassified_percentage:.2f}% of ROI"
                                )
                            
                            # Generate and display visualization image automatically
                            st.divider()
                            st.subheader("📊 Land Use Map Visualization")
                            
                            with st.spinner("Generating visualization..."):
                                try:
                                    # Generate the image (use original CRS versions, function will handle conversion)
                                    img_buffer = generate_land_use_map_image(
                                        final_gdf=final_gdf,  # Original CRS
                                        roi_gdf=roi_gdf,  # Original CRS
                                        class_stats=class_stats,
                                        roi_area_m2=roi_area_m2
                                    )
                                    
                                    # Display the image
                                    st.image(img_buffer, use_container_width=True, caption="Land Use Map Visualization")
                                    
                                    # Download button
                                    st.download_button(
                                        label="📥 Download Land Use Map Image",
                                        data=img_buffer.getvalue(),
                                        file_name="land_use_map.png",
                                        mime="image/png",
                                        key="download_map_image"
                                    )
                                    
                                    st.success("✓ Visualization generated successfully!")
                                except Exception as e:
                                    st.error(f"Failed to generate visualization: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())
                        else:
                            if not roi_geojson:
                                st.warning("⚠️ ROI polygon not available. Statistics require ROI information.")
                            elif "class" not in final_gdf.columns:
                                st.warning("⚠️ No 'class' column found in GeoJSON. Statistics cannot be calculated.")
                    except Exception as e:
                        st.warning(f"⚠️ Could not calculate statistics: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                else:
                    st.info("💡 **Upload your edited GeoJSON file above to see statistics and visualization.**")
                
                st.divider()
                
                # Reset button
                if st.button("🔄 Start Over (New ROI)", type="secondary", key="reset_all"):
                    reset_workflow()
                    st.rerun()
    
    elif bbox and not is_authenticated:
        st.warning("⚠️ Please log in to HuggingFace in the sidebar to proceed with segmentation.")


if __name__ == "__main__":
    main()
