import open_data
import matplotlib.pyplot as plt
import tensorflow as tf

# Function to create a figure
def png_figure():
    width = 12
    height = 8
    return plt.figure(figsize=(width, height))

# Function to create a scatter plot
def scatter_col(ax, image_heights, image_widths):
    ax.set_xlabel("Height")
    ax.set_ylabel("Width")

    ax.scatter(image_heights, image_widths, s=1)
    ax.locator_params(axis='both', tight=True, nbins=6)

# Function to handle full scatter plot generation
def scatter_all(images):
    # Extract height and width of each image
    image_heights = []
    image_widths = []
    
    for image_batch in images:
        for img in image_batch:  # Each image is a tensor of shape (height, width, channels)
            image_heights.append(image_batch.shape[0])
            image_widths.append(image_batch.shape[1])

    # Create figure and plot
    fig = png_figure()
    fig.suptitle("Height and Width of the Images")
    ax = fig.add_subplot(1, 1, 1)
    
    scatter_col(ax, image_heights, image_widths)

    fig.tight_layout()
    figure_name = "Images_H_and_W.png"
    fig.savefig(figure_name)
    plt.close(fig)
    print(f"Saved as {figure_name}")

# Main function to load data and plot
def main():
    X, y = open_data.load_dataset(2)  # Load training dataset (assuming load_dataset returns images & labels)
    
    scatter_all(X)  # Pass the dataset for plotting

if __name__ == "__main__":
    main()


# my images are formated automaticly to 256*256, with 1 for the grayscale
# so they look like if you do image shape on the batch (8,256,256,1) 8 is the batch size