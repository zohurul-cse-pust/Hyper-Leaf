import rasterio

file_path = "00001.tiff"
dataset = rasterio.open(file_path)

print("Band count:", dataset.count)
print("Width:", dataset.width)
print("Height:", dataset.height)
print("CRS:", dataset.crs)
print("Data type:", dataset.dtypes)
