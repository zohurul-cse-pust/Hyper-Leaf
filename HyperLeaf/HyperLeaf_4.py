import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import filters

# === 1. Load hyperspectral image ===
file_path = "00001.tiff"
dataset = rasterio.open(file_path)
num_bands = dataset.count

print("Image width (columns):", dataset.width)
print("Image height (rows):", dataset.height)
print("Number of bands (channels):", num_bands)
print("Data type:", dataset.dtypes[0])
print("Resolution (pixel size):", dataset.res)
dtype = dataset.dtypes[0]
bit_depth = np.iinfo(np.dtype(dtype)).bits
print("Bit depth:", bit_depth)

# === 2. Wavelength info ===
if dataset.descriptions and any(dataset.descriptions):
    wavelengths = dataset.descriptions
else:
    # approximate wavelength range (example: 400-1000 nm)
    wavelengths = np.linspace(400, 1000, num_bands)

# === 3. Read all bands (hyperspectral cube) ===
cube = dataset.read()  # shape: (bands, rows, cols)
cube_hwc = np.transpose(cube, (1, 2, 0))  # (rows, cols, bands)

# === 4. Metrics calculation per band ===
print("\nBand-wise metrics:")
for i in range(num_bands):
    band_img = cube[i, :, :].astype(np.float32)
    # normalize to 0-255
    band_norm = ((band_img - band_img.min()) / (band_img.max() - band_img.min()) * 255).astype(np.uint8)
    
    brightness = np.mean(band_norm)
    contrast = band_norm.std()
    sharpness = cv2.Laplacian(band_norm, cv2.CV_64F).var()
    
    edges = filters.sobel(band_norm)
    non_edge_pixels = band_norm[edges < np.percentile(edges, 75)]
    noise_est = np.std(non_edge_pixels) if non_edge_pixels.size > 0 else 1.0
    
    epsilon = 1e-8
    snr_linear = brightness / (noise_est + epsilon)
    snr_db = 20 * np.log10(snr_linear)
    
    # wavelength
    wl = wavelengths[i] if isinstance(wavelengths[i], str) else f"{wavelengths[i]:.1f} nm"
    
    print(f"Band {i+1} | Wavelength: {wl} | Brightness: {brightness:.2f} | Contrast: {contrast:.2f} | "
          f"Sharpness: {sharpness:.2f} | Estimated Noise: {noise_est:.2f} | SNR(dB): {snr_db:.2f}")

# === 5. Optional visualization (first 3 bands as RGB) ===
if num_bands >= 3:
    rgb = np.transpose(cube[0:3, :, :], (1, 2, 0))
    plt.imshow(rgb / np.max(rgb))
    plt.title("RGB Composite (first 3 bands)")
else:
    plt.imshow(cube[0, :, :], cmap="gray")
    plt.colorbar()
    plt.title("Single-band Grayscale")
plt.axis('off')
plt.show()
