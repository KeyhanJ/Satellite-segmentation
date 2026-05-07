"""
Custom Leaflet-based map component for editing GeoJSON polygons.

This component uses Leaflet.js directly to avoid folium serialization issues.
"""

import json
from pathlib import Path
from typing import Optional

import geopandas as gpd
import streamlit as st
import streamlit.components.v1 as components


class GeoJSONMapEditor:
    """Custom map editor using Leaflet.js directly."""
    
    def __init__(
        self,
        geojson_path: Optional[Path] = None,
        center: Optional[tuple[float, float]] = None,
        zoom_start: int = 10,
        height: int = 700,
        default_class: Optional[str] = None,
    ):
        """
        Initialize the map editor.
        
        Args:
            geojson_path: Path to GeoJSON file to load (optional)
            center: Map center as (lat, lon). If None, will be calculated from GeoJSON bounds
            zoom_start: Initial zoom level
            height: Map height in pixels
            default_class: Default class name to assign to newly drawn polygons
        """
        self.geojson_path = geojson_path
        self.height = height
        self.default_class = default_class
        self.gdf = None
        
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
        
        self.zoom_start = zoom_start
    
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
    
    def render(self, key: str = "geojson_map", force_reload: bool = False, auto_save: bool = False) -> dict:
        """
        Render the map component and return the edited GeoJSON data.
        
        Args:
            key: Unique key for the Streamlit component
            force_reload: If True, reload the GeoJSON file from disk before rendering
            auto_save: If True, automatically save changes back to the file
        
        Returns:
            Dictionary with 'geojson' key containing the edited GeoJSON as string
        """
        # Reload GeoJSON from file if requested or if file path exists
        if force_reload and self.geojson_path and self.geojson_path.exists():
            self.load_geojson(self.geojson_path)
        
        # Get GeoJSON data
        geojson_data = None
        if self.gdf is not None and not self.gdf.empty:
            geojson_data = self.gdf.to_json()
        
        # Calculate bounds if we have data
        bounds = None
        if self.gdf is not None and not self.gdf.empty:
            b = self.gdf.total_bounds
            bounds = [[b[1], b[0]], [b[3], b[2]]]  # [[south, west], [north, east]]
        
        # Create the HTML/JavaScript component
        html = self._create_map_html(geojson_data, bounds, key, self.default_class, auto_save)
        
        # Render the component
        # Note: components.html() doesn't accept a 'key' parameter
        result = components.html(
            html,
            height=self.height + 50,  # Extra space for controls
            width=None
        )
        
        return result
    
    def _create_map_html(self, geojson_data: Optional[str], bounds: Optional[list], key: str, default_class: Optional[str] = None, auto_save: bool = False) -> str:
        """Create the HTML/JavaScript for the Leaflet map."""
        
        # Convert GeoJSON to JavaScript format
        geojson_js = json.dumps(json.loads(geojson_data)) if geojson_data else "null"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GeoJSON Editor</title>
    
    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css" />
    
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }}
        #map {{
            width: 100%;
            height: {self.height}px;
        }}
        .info-panel {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            z-index: 1000;
            font-size: 12px;
        }}
        .export-btn {{
            background: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-top: 10px;
        }}
        .export-btn:hover {{
            background: #45a049;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
        <div class="info-panel">
        <div><strong>GeoJSON Editor</strong></div>
        <div style="margin-top: 5px; font-size: 11px;">
            Draw: Polygon/Rectangle tools<br>
            Edit: Click polygon, drag vertices<br>
            Delete: Click polygon, then trash icon
        </div>
        <button class="export-btn" onclick="exportGeoJSON()">Export GeoJSON</button>
    </div>

    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
    
    <script>
        // Initialize map
        var map = L.map('map').setView([{self.center[0]}, {self.center[1]}], {self.zoom_start});
        
        // Add satellite basemap (Esri World Imagery)
        var satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Esri',
            maxZoom: 19
        }}).addTo(map);
        
        // Add OpenStreetMap as alternative
        var osmLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: 'OpenStreetMap',
            maxZoom: 19
        }});
        
        // Layer control
        var baseMaps = {{
            "Satellite": satelliteLayer,
            "OpenStreetMap": osmLayer
        }};
        L.control.layers(baseMaps).addTo(map);
        
         // Feature group for editable layers
         var drawnItems = new L.FeatureGroup();
         map.addLayer(drawnItems);
         
         // Load initial GeoJSON if provided
         var initialGeoJSON = {geojson_js};
         if (initialGeoJSON && initialGeoJSON.features && initialGeoJSON.features.length > 0) {{
             L.geoJSON(initialGeoJSON, {{
                 style: {{
                     fillColor: '#3388ff',
                     color: '#3388ff',
                     weight: 2,
                     fillOpacity: 0.5
                 }},
                 onEachFeature: function(feature, layer) {{
                     // Add to editable group BEFORE Draw control is created
                     drawnItems.addLayer(layer);
                     
                     // Add popup with properties
                     if (feature.properties) {{
                         var props = Object.keys(feature.properties)
                             .filter(k => k !== 'geometry')
                             .map(k => k + ': ' + feature.properties[k])
                             .join('<br>');
                         if (props) {{
                             layer.bindPopup(props);
                         }}
                     }}
                 }}
             }});
         }}
         
         // Draw control - must be created AFTER layers are added to featureGroup
         var drawControl = new L.Control.Draw({{
             position: 'topleft',
             draw: {{
                 polyline: false,
                 polygon: {{
                     allowIntersection: false,
                     showArea: true
                 }},
                 rectangle: true,
                 circle: false,
                 marker: false,
                 circlemarker: false
             }},
             edit: {{
                 featureGroup: drawnItems,
                 edit: true,  // Enable editing
                 remove: true
             }}
         }});
         map.addControl(drawControl);
         
         // Enable editing on all existing layers
         drawnItems.eachLayer(function(layer) {{
             if (layer.editing) {{
                 // Layer already has editing capability
             }}
         }});
        
        // Default class for newly drawn polygons
        var defaultClass = {json.dumps(default_class)};
        
        // Make defaultClass globally accessible so it can be updated from parent
        window.defaultClass = defaultClass;
        
        // Function to update default class
        window.updateDefaultClass = function(newClass) {{
            defaultClass = newClass;
            window.defaultClass = newClass;
            console.log('Default class updated to:', newClass);
        }};
        
        // Handle drawing events
        map.on(L.Draw.Event.CREATED, function (e) {{
            var type = e.layerType;
            var layer = e.layer;
            
            // Always get the CURRENT default class value at the time of drawing
            // Don't use closure - always read from window.defaultClass or local defaultClass
            var currentDefaultClass = null;
            if (typeof window.defaultClass !== 'undefined' && window.defaultClass !== null && window.defaultClass !== '') {{
                currentDefaultClass = window.defaultClass;
            }} else if (defaultClass) {{
                currentDefaultClass = defaultClass;
            }}
            
            console.log('Polygon created - currentDefaultClass:', currentDefaultClass, 'window.defaultClass:', window.defaultClass, 'local defaultClass:', defaultClass);
            
            if (currentDefaultClass) {{
                // Get or create the GeoJSON feature
                var geoJson = layer.toGeoJSON();
                if (!geoJson.properties) {{
                    geoJson.properties = {{}};
                }}
                geoJson.properties.class = currentDefaultClass;
                
                // Store the feature back on the layer so toGeoJSON() will use it
                layer.feature = geoJson;
                
                // Also update the layer's options to ensure the class is preserved
                if (layer.options) {{
                    if (!layer.options.properties) {{
                        layer.options.properties = {{}};
                    }}
                    layer.options.properties.class = currentDefaultClass;
                }}
                
                // Set a custom property on the layer itself
                layer._assignedClass = currentDefaultClass;
                
                console.log('✓ Polygon assigned to class:', currentDefaultClass);
            }} else {{
                console.warn('⚠ No default class set, polygon will not have a class property');
            }}
            
            // Add to editable group
            drawnItems.addLayer(layer);
        }});
        
        // Fit bounds if we have data
        {f"map.fitBounds({bounds});" if bounds else ""}
        
        // Export function
        function exportGeoJSON() {{
            var features = [];
            drawnItems.eachLayer(function(layer) {{
                if (layer.toGeoJSON) {{
                    features.push(layer.toGeoJSON());
                }}
            }});
            
            var geojson = {{
                type: "FeatureCollection",
                features: features
            }};
            
            // Create download link
            var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(geojson, null, 2));
            var downloadAnchorNode = document.createElement('a');
            downloadAnchorNode.setAttribute("href", dataStr);
            downloadAnchorNode.setAttribute("download", "edited_geojson.json");
            document.body.appendChild(downloadAnchorNode);
            downloadAnchorNode.click();
            downloadAnchorNode.remove();
            
            // Also store in a global variable for Streamlit to access
            window.exportedGeoJSON = JSON.stringify(geojson);
            
            // Send to Streamlit parent
            if (window.parent) {{
                window.parent.postMessage({{
                    type: 'streamlit:setComponentValue',
                    value: JSON.stringify(geojson)
                }}, '*');
            }}
        }}
        
        // Auto-export on component value request (for Streamlit integration)
        window.streamlitSetComponentValue = function(value) {{
            exportGeoJSON();
        }};
        
        // Listen for messages from Streamlit and parent window
        window.addEventListener('message', function(event) {{
            if (event.data && event.data.type === 'streamlit:setFrameHeight') {{
                // Handle iframe height adjustment
            }} else if (event.data && event.data.type === 'updateDefaultClass') {{
                // Handle default class update from parent window via postMessage
                var newClass = event.data.class;
                console.log('Received default class update via postMessage:', newClass);
                if (window.updateDefaultClass) {{
                    window.updateDefaultClass(newClass);
                }} else {{
                    // Fallback if function doesn't exist yet
                    defaultClass = newClass;
                    window.defaultClass = newClass;
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        return html
    
    def save_geojson(self, geojson_string: str, output_path: Path) -> bool:
        """
        Save GeoJSON string to file.
        
        Args:
            geojson_string: GeoJSON as string
            output_path: Path to save the file
            
        Returns:
            True if successful
        """
        try:
            # Parse and validate GeoJSON
            geojson_data = json.loads(geojson_string)
            
            # Convert to GeoDataFrame and save
            gdf = gpd.GeoDataFrame.from_features(geojson_data['features'], crs='EPSG:4326')
            gdf.to_file(output_path, driver="GeoJSON")
            return True
        except Exception as e:
            raise ValueError(f"Failed to save GeoJSON: {e}")

