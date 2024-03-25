import matplotlib.pyplot as plt

def plot_deformation(deformation, coordinates, ax=None, set_title=None, point_size=5):
    """
    To scatter plot of deformation with colormap in x or y-direction using the coordinates
    """
    x_coords = coordinates[:, 0]
    y_coords = coordinates[:, 1]

    created_ax = False

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3)) 
        created_ax = True
    else:
        fig = ax.figure 

    scatter = ax.scatter(x_coords, y_coords, c=deformation, cmap='viridis', s=point_size)

    cbar = fig.colorbar(scatter, ax=ax, label='displacement')

    # Set the title
    if set_title is None:
        ax.set_title("Deformation")
    else:
        ax.set_title(set_title)

    # Set axis labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    if created_ax:
        plt.show()

   
