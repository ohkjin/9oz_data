import matplotlib as mpl
import matplotlib.pyplot as plt
import io
from PIL import Image

def show_segment(model,pred_seg):
    label_names = list(model.config.id2label)
    # Create a color map with the same number of colors as your labels
    # Use the updated method to get the colormap
    cmap = mpl.colormaps['gnuplot2'] #gnuplot2, tab20

    # Create the figure and axes for the plot and the colorbar
    fig, ax = plt.subplots()

    # Display the segmentation
    im = ax.imshow(pred_seg, cmap=cmap)

    # Create a colorbar
    cbar = fig.colorbar(im, ax=ax, ticks=range(len(label_names)))
    cbar.ax.set_yticklabels(label_names)

    # Get the number of labels
    n_labels = len(label_names)

    # Extract RGB values for each color in the colormap
    colors = cmap.colors[:n_labels]

    # Convert RGBA to RGB by omitting the Alpha value
    rgb_colors = [color[:3] for color in colors]

    # Create a dictionary mapping labels to RGB colors
    label_to_color = dict(zip(label_names, rgb_colors))

    # Display the mapping
    for label, color in label_to_color.items():
        print(f"{label}: {color}")
    
    # Export the plt image
    # plt.show()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt_img = Image.open(img_buf)
    
    return plt_img

# Labels: 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"