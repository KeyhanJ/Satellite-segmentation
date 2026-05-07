"""
Interactive map component for editing GeoJSON polygons.

Features:
- Satellite basemap
- Import and visualize GeoJSON polygons
- Reshape polygons by dragging vertices
- Add new polygons
- Delete polygons
- Export edited GeoJSON
"""

from pathlib import Path
from typing import Optional

import folium
import geopandas as gpd
from folium.plugins import Draw
from shapely.geometry import shape


class GeoJSONMapEditor:
    """Interactive map editor for GeoJSON polygons."""
    
    def __init__(
        self,
        geojson_path: Optional[Path] = None,
        center: Optional[tuple[float, float]] = None,
        zoom_start: int = 10,
        height: int = 700,
    ):
        """
        Initialize the map editor.
        
        Args:
            geojson_path: Path to GeoJSON file to load (optional)
            center: Map center as (lat, lon). If None, will be calculated from GeoJSON bounds
            zoom_start: Initial zoom level
            height: Map height in pixels
        """
        self.geojson_path = geojson_path
        self.height = height
        self.gdf = None
        self.feature_group = None
        
        # Load GeoJSON if provided
        if geojson_path and geojson_path.exists():
            self.load_geojson(geojson_path)
        
        # Determine map center
        if center:
            self.center = center
        elif self.gdf is not None and not self.gdf.empty:
            bounds = self.gdf.total_bounds
            self.center = ((bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2)
        else:
            self.center = (0, 0)  # Default center
        
        # Create map
        self.map = folium.Map(
            location=self.center,
            zoom_start=zoom_start,
            tiles=None,
            height=height
        )
        
        # Add satellite basemap
        self._add_satellite_basemap()
        
        # Add OpenStreetMap as alternative
        folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(self.map)
        
        # Create feature group for editable polygons
        self.feature_group = folium.FeatureGroup(name="Polygons")
        # Always add feature group to map (even if empty) so Draw can reference it
        self.feature_group.add_to(self.map)
        
        # Add existing polygons if loaded
        if self.gdf is not None and not self.gdf.empty:
            self._add_polygons_to_map()
        
        # Add Draw plugin for editing (after feature group is added to map)
        self._add_draw_plugin()
        
        # Add layer control
        folium.LayerControl().add_to(self.map)
        
        # Fit bounds if we have data
        if self.gdf is not None and not self.gdf.empty:
            bounds = self.gdf.total_bounds
            self.map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    
    def _add_satellite_basemap(self):
        """Add satellite basemap tile layer."""
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(self.map)
    
    def _add_polygons_to_map(self):
        """Add polygons from GeoDataFrame to the map."""
        if self.gdf is None or self.gdf.empty:
            return
        
        # Convert to GeoJSON
        geojson_data = self.gdf.to_json()
        
        # Get non-geometry columns for tooltip
        tooltip_fields = [col for col in self.gdf.columns if col != 'geometry']
        
        # Add to feature group with styling
        folium.GeoJson(
            geojson_data,
            style_function=lambda feature: {
                'fillColor': '#3388ff',
                'color': '#3388ff',
                'weight': 2,
                'fillOpacity': 0.5,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=[f'{col}:' for col in tooltip_fields],
            ) if tooltip_fields else None,
        ).add_to(self.feature_group)
    
    def _add_draw_plugin(self):
        """Add Draw plugin for editing polygons."""
        # Create Draw plugin with edit options
        # Note: We pass the feature group, but ensure it's already added to the map
        draw = Draw(
            export=False,
            position='topleft',
            draw_options={
                'polyline': False,
                'polygon': True,
                'rectangle': True,
                'circle': False,
                'marker': False,
                'circlemarker': False,
            },
            edit_options={
                'featureGroup': self.feature_group,
                'edit': True,
                'remove': True,
            }
        )
        # Add draw plugin to map
        draw.add_to(self.map)
        
        # Also add draw plugin's drawnItems to the feature group so new drawings are editable
        # This ensures new polygons drawn with Draw are also in the editable group
        try:
            # Add a callback to add new drawings to the feature group
            # This is handled automatically by folium Draw when featureGroup is specified
            pass
        except Exception:
            # If there's an issue, continue without the callback
            pass
    
    def load_geojson(self, geojson_path: Path):
        """Load GeoJSON file."""
        try:
            self.gdf = gpd.read_file(geojson_path)
            # Ensure CRS is WGS84 (EPSG:4326) for web maps
            if self.gdf.crs is None:
                self.gdf.set_crs('EPSG:4326', inplace=True)
            elif self.gdf.crs != 'EPSG:4326':
                self.gdf = self.gdf.to_crs('EPSG:4326')
            self.geojson_path = geojson_path
        except Exception as e:
            raise ValueError(f"Failed to load GeoJSON: {e}")
    
    def get_map(self) -> folium.Map:
        """Get the folium map object."""
        return self.map
    
    def save_geojson(self, output_path: Path, map_data: Optional[dict] = None) -> bool:
        """
        Save edited GeoJSON from map state.
        
        Args:
            output_path: Path to save the GeoJSON file
            map_data: Map state dictionary from streamlit_folium (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # If no map_data provided, save original data
            if map_data is None:
                if self.gdf is not None and not self.gdf.empty:
                    self.gdf.to_file(output_path, driver="GeoJSON")
                    return True
                return False
            
            # Get all drawings from the map (includes both original and new/edited)
            all_drawings = map_data.get("all_drawings", []) if map_data else []
            
            if not all_drawings:
                # If no drawings, check if we have original data
                if self.gdf is not None and not self.gdf.empty:
                    self.gdf.to_file(output_path, driver="GeoJSON")
                    return True
                return False
            
            # Convert drawings to GeoDataFrame
            features = []
            for drawing in all_drawings:
                geom = drawing.get("geometry")
                if geom and geom.get("type") in ["Polygon", "MultiPolygon"]:
                    try:
                        # shapely's shape() expects GeoJSON format [lon, lat]
                        # streamlit_folium should return GeoJSON format already
                        features.append(shape(geom))
                    except Exception as e:
                        # Skip invalid geometries - might need coordinate swap
                        # Try swapping coordinates if first attempt failed
                        try:
                            geom_swapped = self._swap_coordinates(geom)
                            features.append(shape(geom_swapped))
                        except Exception:
                            continue
            
            if not features:
                # No valid features found, save original if available
                if self.gdf is not None and not self.gdf.empty:
                    self.gdf.to_file(output_path, driver="GeoJSON")
                    return True
                return False
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(geometry=features, crs="EPSG:4326")
            
            # Try to preserve properties from original GeoJSON
            if self.gdf is not None and not self.gdf.empty:
                # If same number of features, try to preserve properties
                if len(features) == len(self.gdf):
                    # Copy properties (this is a simple approach - may need refinement)
                    for col in self.gdf.columns:
                        if col != 'geometry':
                            gdf[col] = self.gdf[col].values
                else:
                    # Different number of features - add default properties
                    for col in self.gdf.columns:
                        if col != 'geometry':
                            gdf[col] = None
            
            # Save to file
            gdf.to_file(output_path, driver="GeoJSON")
            return True
            
        except Exception as e:
            raise ValueError(f"Failed to save GeoJSON: {e}")
    
    def _swap_coordinates(self, geom: dict) -> dict:
        """
        Swap coordinates from [lat, lon] to [lon, lat] format.
        
        Folium Draw returns coordinates as [lat, lon], but GeoJSON/shapely
        expects [lon, lat].
        """
        def swap_coords(coords):
            if isinstance(coords[0], (int, float)):
                # Single coordinate pair [lat, lon] -> [lon, lat]
                return [coords[1], coords[0]]
            else:
                # Nested coordinates - recurse
                return [swap_coords(coord) for coord in coords]
        
        geom_copy = geom.copy()
        if geom_copy.get("type") == "Polygon":
            geom_copy["coordinates"] = [swap_coords(ring) for ring in geom_copy["coordinates"]]
        elif geom_copy.get("type") == "MultiPolygon":
            geom_copy["coordinates"] = [[swap_coords(ring) for ring in poly] for poly in geom_copy["coordinates"]]
        
        return geom_copy

