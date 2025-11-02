import rasterio  # Library to read/write raster files like GeoTIFF
import numpy as np  # Library for numerical operations
import matplotlib.pyplot as plt  # Library for plotting images

# Load the TIFF image
file_path = "00001.tiff"  # Path to the TIFF image file
dataset = rasterio.open(file_path)  # Open the TIFF file as a dataset

# Print basic image properties
print("Image width (columns):", dataset.width)  # Number of columns in the image
print("Image height (rows):", dataset.height)  # Number of rows in the image
print("Number of bands (channels):", dataset.count)  # Number of spectral bands
print("Data type:", dataset.dtypes[0])  # Data type of the first band
print("Coordinate Reference System (CRS):", dataset.crs)  # Spatial reference info
print("Bounding Box:", dataset.bounds)  # Geographical extent of the image

# Read the image as a NumPy array
img_array = dataset.read()  # Read all bands into a NumPy array (bands, rows, cols)
print("Numpy array shape (bands, rows, cols):", img_array.shape)  # Shape of the array

# Extract a pixel value at the center
row = dataset.height // 2  # Middle row index
col = dataset.width // 2  # Middle column index
pixel_value = img_array[:, row, col]  # Pixel values across all bands at center
print(f"Pixel value at (row={row}, col={col}):", pixel_value)  # Print pixel values

# Show image metadata
print("\nMetadata:")
print(dataset.meta)  # Print dataset metadata like driver, dtype, shape, CRS, etc.

# Visualization
if dataset.count >= 3:  # If image has 3 or more bands
    rgb = img_array[0:3, :, :]  # Take first 3 bands for RGB composite
    rgb = np.transpose(rgb, (1, 2, 0))  # Reorder to (rows, cols, bands) for plotting
    plt.imshow(rgb / np.max(rgb))  # Normalize and display RGB image
    plt.title("RGB Composite")  # Title of the plot
    plt.show()  # Show the image
else:
    plt.imshow(img_array[0, :, :], cmap="gray")  # Display single-band image in grayscale
    plt.colorbar()  # Add colorbar for reference
    plt.show()  # Show the grayscale image
