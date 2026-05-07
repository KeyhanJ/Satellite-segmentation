"""
Streamlit webapp for editing GeoJSON polygons on an interactive map.

Features:
- Import GeoJSON files
- Visualize polygons on satellite basemap
- Edit polygons (reshape, add, delete)
- Export edited GeoJSON
"""

import json
from pathlib import Path
from typing import Optional

import geopandas as gpd
import streamlit as st

from map_component import GeoJSONMapEditor

# Page configuration
st.set_page_config(
    page_title="GeoJSON Polygon Editor",
    page_icon="🗺️",
    layout="wide"
)

# Initialize session state
if "geojson_data" not in st.session_state:
    st.session_state.geojson_data = None
if "geojson_path" not in st.session_state:
    st.session_state.geojson_path = None
if "map_editor" not in st.session_state:
    st.session_state.map_editor = None


def load_geojson_file(uploaded_file) -> Optional[Path]:
    """Save uploaded GeoJSON file and return its path."""
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        temp_path = temp_dir / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return temp_path
    except Exception as e:
        st.error(f"Failed to save uploaded file: {e}")
        return None


def main():
    """Main Streamlit app."""
    st.title("🗺️ GeoJSON Polygon Editor")
    st.markdown(
        "Import a GeoJSON file, edit polygons on the map, and export your changes."
    )
    
    # Sidebar for file operations
    with st.sidebar:
        st.header("📁 File Operations")
        
        # File upload
        st.subheader("Import GeoJSON")
        uploaded_file = st.file_uploader(
            "Choose a GeoJSON file",
            type=["geojson", "json"],
            help="Upload a GeoJSON file containing polygons"
        )
        
        if uploaded_file is not None:
            # Load the file
            temp_path = load_geojson_file(uploaded_file)
            if temp_path:
                st.session_state.geojson_path = temp_path
                st.session_state.geojson_data = gpd.read_file(temp_path)
                st.success(f"✅ Loaded {len(st.session_state.geojson_data)} features")
                st.session_state.map_editor = None  # Reset map editor to reload
        
        # Load from existing files
        st.subheader("Or Load from File")
        existing_files = list(Path(".").glob("*.geojson"))
        if existing_files:
            selected_file = st.selectbox(
                "Select existing GeoJSON file",
                options=[str(f) for f in existing_files],
                key="file_selector"
            )
            if st.button("Load File", key="load_file_btn"):
                st.session_state.geojson_path = Path(selected_file)
                st.session_state.geojson_data = gpd.read_file(selected_file)
                st.success(f"✅ Loaded {len(st.session_state.geojson_data)} features")
                st.session_state.map_editor = None
                st.rerun()
        
        st.divider()
        
        # Export section
        st.subheader("💾 Export")
        if st.session_state.geojson_path:
            st.info("Use the Export button below the map to save your edits.")
        else:
            st.info("Load a GeoJSON file first to enable export.")
    
    # Main content area
    if st.session_state.geojson_path and st.session_state.geojson_path.exists():
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Features", len(st.session_state.geojson_data))
        with col2:
            st.metric("File", st.session_state.geojson_path.name)
        with col3:
            if st.button("🔄 Reload Map", key="reload_map"):
                st.session_state.map_editor = None
                st.rerun()
        
        # Create or get map editor
        if st.session_state.map_editor is None:
            with st.spinner("Loading map..."):
                try:
                    st.session_state.map_editor = GeoJSONMapEditor(
                        geojson_path=st.session_state.geojson_path,
                        height=700
                    )
                except Exception as e:
                    st.error(f"Failed to create map: {e}")
                    st.stop()
        
        # Instructions
        with st.expander("📖 How to Use", expanded=False):
            st.markdown("""
            **Editing Tools:**
            - **Add Polygon**: Click the polygon or rectangle tool in the top-left, then draw on the map
            - **Reshape Polygon**: Click on an existing polygon to select it, then drag the vertices (small squares) to reshape
            - **Delete Polygon**: Click on a polygon to select it, then click the delete button (trash icon)
            
            **Tips:**
            - Use the layer control (top-right) to switch between Satellite and OpenStreetMap
            - Zoom in for more precise editing
            - After making edits, click the Export button below to save your changes
            """)
        
        # Render map
        st.subheader("🗺️ Interactive Map")
        st.info("💡 **Tip**: Use the 'Export GeoJSON' button on the map (top-right) to download your edited polygons. The file will download automatically.")
        map_result = st.session_state.map_editor.render(key="geojson_editor_map")
        
        # Export section
        st.divider()
        st.subheader("💾 Export Edited GeoJSON")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            output_filename = st.text_input(
                "Output filename",
                value=f"edited_{st.session_state.geojson_path.stem}.geojson",
                key="output_filename"
            )
        
        with col2:
            st.write("")  # Spacing
            export_btn = st.button("📥 Export GeoJSON", type="primary", key="export_btn")
        
        st.info("""
        **Export Instructions:**
        1. Make your edits on the map (add, reshape, or delete polygons)
        2. Click the **"Export GeoJSON"** button in the top-right corner of the map
        3. The edited GeoJSON file will automatically download to your computer
        
        The exported file will contain all polygons currently visible on the map.
        """)
        
        # Display current GeoJSON info
        with st.expander("📊 Current GeoJSON Info", expanded=False):
            if st.session_state.geojson_data is not None:
                st.write(f"**Number of features:** {len(st.session_state.geojson_data)}")
                st.write(f"**CRS:** {st.session_state.geojson_data.crs}")
                st.write(f"**Columns:** {', '.join(st.session_state.geojson_data.columns)}")
                
                # Show bounds
                bounds = st.session_state.geojson_data.total_bounds
                st.write(f"**Bounds:**")
                st.write(f"- Min: ({bounds[0]:.6f}, {bounds[1]:.6f})")
                st.write(f"- Max: ({bounds[2]:.6f}, {bounds[3]:.6f})")
    
    else:
        # No file loaded - show instructions
        st.info("👈 Please upload a GeoJSON file using the sidebar to get started.")
        
        # Show example
        with st.expander("📝 Example GeoJSON Structure"):
            example = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "Example Polygon"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [-122.4, 37.8],
                                [-122.3, 37.8],
                                [-122.3, 37.9],
                                [-122.4, 37.9],
                                [-122.4, 37.8]
                            ]]
                        }
                    }
                ]
            }
            st.json(example)


if __name__ == "__main__":
    main()

