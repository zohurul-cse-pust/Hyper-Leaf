# === Corrected Python code for reading hyperspectral/GeoTIFF images ===
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# === 1. Load the TIFF image ===
file_path = "00001.tiff"   # change this to your file
dataset = rasterio.open(file_path)

# === 2. Print basic properties ===
print("Image width (columns):", dataset.width)
print("Image height (rows):", dataset.height)
print("Number of bands (channels):", dataset.count)
print("Data type:", dataset.dtypes[0])
print("Coordinate Reference System (CRS):", dataset.crs)
print("Bounding Box:", dataset.bounds)

# === 3. Read the image as a NumPy array ===
# Shape: (bands, rows, cols)
img_array = dataset.read()
print("Numpy array shape (bands, rows, cols):", img_array.shape)

# === 4. Extract a safe pixel value ===
# pick a valid row and col inside the image range
row = dataset.height // 2   # middle row (0 to height-1)
col = dataset.width // 2    # middle col (0 to width-1)

pixel_value = img_array[:, row, col]
print(f"Pixel value at (row={row}, col={col}):", pixel_value)

# === 5. Show metadata ===
print("\nMetadata:")
print(dataset.meta)

# === 6. Visualization ===
if dataset.count >= 3:
    # If the image has 3 or more bands, display RGB composite
    rgb = img_array[0:3, :, :]   # first 3 bands
    rgb = np.transpose(rgb, (1, 2, 0))  # reshape to (rows, cols, bands)
    plt.imshow(rgb / np.max(rgb))
    plt.title("RGB Composite")
    plt.show()
else:
    # If the image has only 1 band, show grayscale
    plt.imshow(img_array[0, :, :], cmap="gray")
    plt.colorbar()
    plt.show()
