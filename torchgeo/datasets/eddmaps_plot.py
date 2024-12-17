import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Optional

def plot(
    self,
    query: Optional[BoundingBox] = None,
    title: str = "EDDMapS Dataset",
    point_size: int = 20,
    point_color: str = 'blue',
    query_color: str = 'red',
    annotate: bool = False,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    
    """ Plot the dataset points with optional query bounding box
    Args:
        
        query: The query to look for points, in the form of a bounding box: (minx,maxx,miny,maxy,mint,maxt)
        title: Title of the plot
        point_size: The size of the points plotted
        point_color: The default color of the points, in case no such map is provided
        query_color: color for the points which fall into the query
        annotate: Controls if the points with timestamps are annotated
        figsize: Size of drawn figure in the shape: (width, height)
        
        Raises:

    ValueError: When no points could be plotted because none were valid.

    """
    
    # Filtering valid lat and long rows
    valid_data = self.data.dropna(subset = [ 'Latitude' , 'Longitude'])
    if valid_data.empty:
        raise ValueError("No valid lat/long data to plot.")

    fig, ax = plt.subplots(figsize=figsize)

    # Plot-at-all points

    ax.scatter(

    valid_data['Longitude'],

    valid_data['Latitude'],

    s = point_size,

    c = point_color,

    edgecolor = 'k',

    alpha = 0.6,

    label = 'All Observations'

    )

    #highlighting queried points (If) provided bounding box query

    if query:
        minx, maxx, miny, maxy, mint, maxt = query
        hits = self.index.intersection((minx,maxx,miny,maxy,mint, maxt))
                  
    # Get coordinates of hits to highlight
    query_points = valid_data.iloc[[list(hits)]]
    ax.scatter(
        query_points['Longitude'],
        query_points['Latitude'],
        s = point_size * 1.5,
        c = query_color,
        edgecolor = 'white',
        alpha = 0.8,
        label = 'Query Results'
              )
    
    # Draw a bounding box
    bbox_patch = patches.rectangle(
        (minx, miny), maxx - minx, maxy - miny,
        linewidth = 2, edgecolor = 'red', facecolor='none', linestyle = '--', label = "Query Bounding Box"
    )
    ax.add_patch(bbox_patch)
    
    # Optional annotations
    if annotate:
        for _, row in valid_data.iterrows():
            ax.annotate(
                row['ObsDate'], (row['Longitude'], row['Latitude']),
                fontsize=8, alpha=0.7, textcoords="offset points", xytext=(0, 5), ha='center'
            )

    # Plot styling
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    plt.show()