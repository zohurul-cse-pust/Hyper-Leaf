import rasterio  # Import rasterio library to read raster (TIFF) files

file_path = "00001.tiff"  # Path to the TIFF image file

dataset = rasterio.open(file_path)  # Open the TIFF file as a dataset object

print("Band count:", dataset.count)  # Print number of bands in the raster file
print("Width:", dataset.width)  # Print the width (number of columns) of the image
print("Height:", dataset.height)  # Print the height (number of rows) of the image
print("CRS:", dataset.crs)  # Print the coordinate reference system of the raster
print("Data type:", dataset.dtypes)  # Print the data type(s) of pixel values in each band
