#this is some code which creates a map of the bird recording locations
#what was discovered is that the map is global
#so locational data is most likely not useful for this project

import pandas as pd
import folium
from folium.plugins import MarkerCluster
import os
import matplotlib.pyplot as plt

# Get the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read the train.csv file
train_path = os.path.join(script_dir, '../rawdata/train.csv')
print(f"Looking for file at: {train_path}")

# Read CSV and drop rows with NaN values in latitude or longitude
df = pd.read_csv(train_path)
df = df.dropna(subset=['latitude', 'longitude'])

# Print some basic statistics
print(f"Found {len(df)} valid locations")
print("\nData Distribution by Collection:")
print(df['collection'].value_counts())
print("\nGeographic Distribution:")
print(f"Latitude range: {df['latitude'].min()} to {df['latitude'].max()}")
print(f"Longitude range: {df['longitude'].min()} to {df['longitude'].max()}")

# Create a map centered at the mean of all coordinates
m = folium.Map(
    location=[df['latitude'].mean(), df['longitude'].mean()],
    zoom_start=2
)

# Create a marker cluster
marker_cluster = MarkerCluster().add_to(m)

# Define colors for different collections
collection_colors = {
    'CSA': 'red',
    'SSW': 'blue',
    'ML': 'green',
    'XC': 'purple',
    'EBIRD': 'orange',
    'OTHERS': 'gray'
}

# Add markers for each location
for idx, row in df.iterrows():
    # Get color based on collection
    color = collection_colors.get(row['collection'], 'gray')
    
    # Create popup content
    popup_content = f"""
    <div style='font-family: Arial, sans-serif;'>
        <h3 style='margin: 0 0 10px 0;'>{row['common_name']}</h3>
        <p style='margin: 5px 0;'><b>Scientific Name:</b> {row['scientific_name']}</p>
        <p style='margin: 5px 0;'><b>Collection:</b> {row['collection']}</p>
        <p style='margin: 5px 0;'><b>Location:</b> {row['latitude']:.4f}°N, {row['longitude']:.4f}°E</p>
        <p style='margin: 5px 0;'><b>Author:</b> {row['author']}</p>
        <p style='margin: 5px 0;'><b>License:</b> {row['license']}</p>
        <p style='margin: 5px 0;'><b>Date:</b> {row.get('date', 'N/A')}</p>
        <p style='margin: 5px 0;'><b>Time:</b> {row.get('time', 'N/A')}</p>
        <p style='margin: 5px 0;'><b>Rating:</b> {row.get('rating', 'N/A')}</p>
        <p style='margin: 5px 0;'><b>Type:</b> {row.get('type', 'N/A')}</p>
    </div>
    """
    
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        popup=folium.Popup(popup_content, max_width=300),
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7
    ).add_to(marker_cluster)

# Save the map to an HTML file in the same directory as this script
output_path = os.path.join(script_dir, 'bird_locations_map.html')
m.save(output_path)
print(f"Map has been created and saved as '{output_path}'")

# Create a simple plot to show the distribution
plt.figure(figsize=(10, 6))
plt.scatter(df['longitude'], df['latitude'], alpha=0.5, s=10)
plt.title('Bird Recording Locations Distribution')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.savefig(os.path.join(script_dir, 'location_distribution.png'))
print("Location distribution plot has been saved as 'location_distribution.png'")