import rasterio  # Library to read raster (TIFF/hyperspectral) files
import numpy as np  # Library for numerical operations
import matplotlib.pyplot as plt  # Library for plotting images
import cv2  # OpenCV library for image processing (e.g., sharpness)
from skimage import filters  # For edge detection and other image filters

# Load hyperspectral image
file_path = "00001.tiff"  # Path to the TIFF image file
dataset = rasterio.open(file_path)  # Open the TIFF file as a dataset object
num_bands = dataset.count  # Get number of bands in the hyperspectral image

print("Image width (columns):", dataset.width)  # Width (number of columns)
print("Image height (rows):", dataset.height)  # Height (number of rows)
print("Number of bands (channels):", num_bands)  # Total number of spectral bands
print("Data type:", dataset.dtypes[0])  # Data type of pixel values
print("Resolution (pixel size):", dataset.res)  # Pixel resolution (x, y)
dtype = dataset.dtypes[0]  # Store data type for bit depth calculation
bit_depth = np.iinfo(np.dtype(dtype)).bits  # Calculate bit depth of pixel values
print("Bit depth:", bit_depth)  # Print bit depth

# Wavelength info
if dataset.descriptions and any(dataset.descriptions):  # Check if band descriptions exist
    wavelengths = dataset.descriptions  # Use descriptions as wavelengths
else:
    wavelengths = np.linspace(400, 1000, num_bands)  # Approximate wavelengths (example: 400-1000 nm)

# Read all bands (hyperspectral cube)
cube = dataset.read()  # Read all bands: shape (bands, rows, cols)
cube_hwc = np.transpose(cube, (1, 2, 0))  # Reorder to (rows, cols, bands) for easier visualization

# Metrics calculation per band
print("\nBand-wise metrics:")
for i in range(num_bands):
    band_img = cube[i, :, :].astype(np.float32)  # Convert band to float32 for calculations
    band_norm = ((band_img - band_img.min()) / (band_img.max() - band_img.min()) * 255).astype(np.uint8)  # Normalize to 0-255
    
    brightness = np.mean(band_norm)  # Mean intensity as brightness
    contrast = band_norm.std()  # Standard deviation as contrast
    sharpness = cv2.Laplacian(band_norm, cv2.CV_64F).var()  # Laplacian variance for sharpness
    
    edges = filters.sobel(band_norm)  # Sobel filter to detect edges
    non_edge_pixels = band_norm[edges < np.percentile(edges, 75)]  # Pixels not on edges
    noise_est = np.std(non_edge_pixels) if non_edge_pixels.size > 0 else 1.0  # Estimate noise from non-edge pixels
    
    epsilon = 1e-8  # Small value to avoid division by zero
    snr_linear = brightness / (noise_est + epsilon)  # Linear Signal-to-Noise Ratio
    snr_db = 20 * np.log10(snr_linear)  # SNR in decibels
    
    wl = wavelengths[i] if isinstance(wavelengths[i], str) else f"{wavelengths[i]:.1f} nm"  # Wavelength info
    
    # Print all metrics for this band
    print(f"Band {i+1} | Wavelength: {wl} | Brightness: {brightness:.2f} | Contrast: {contrast:.2f} | "
          f"Sharpness: {sharpness:.2f} | Estimated Noise: {noise_est:.2f} | SNR(dB): {snr_db:.2f}")

# Optional visualization (first 3 bands as RGB)
if num_bands >= 3:  # Check if we have at least 3 bands
    rgb = np.transpose(cube[0:3, :, :], (1, 2, 0))  # Take first 3 bands as RGB
    plt.imshow(rgb / np.max(rgb))  # Normalize for display
    plt.title("RGB Composite (first 3 bands)")  # Plot title
else:
    plt.imshow(cube[0, :, :], cmap="gray")  # Single-band grayscale display
    plt.colorbar()  # Add colorbar
    plt.title("Single-band Grayscale")  # Plot title
plt.axis('off')  # Hide axis
plt.show()  # Show the image
