import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#? File path
path_img = 'img/test.jpg'

#* Upload img and convert it to grayscale ('L' = Luma)
img_pil = Image.open(path_img).convert('L')

# Convert img to matriz by NumPy
# We use flat64 because we have negative values ​​when performing subtractions (derivations).
# If hold uint8 (0-255) negative nums will cause a overflow

matrix_img = np.array(img_pil, dtype=np.float64)

# Display the matrix and img dimensions to confirm

#plt.imshow(matrix_img, cmap='gray')
#plt.title('Matrix Original (Grayscale)')
#plt.show()

#TODO Calculo de Gradiente (Diferencias infinitas)

# Get high and width from matrix
high, width = matrix_img.shape

# Create empty matrix with same tam
gx = np.zeros_like(matrix_img)
gy = np.zeros_like(matrix_img)

# Diff to front in y: f(x+1, y) - f(x, y)
# Substract down row - current row
gx[:, :-1] = matrix_img[:, 1:] - matrix_img[:, :-1] #* It takes all the pixels but shifted one column to the right.

# Diff to front int y: f(x, y+1) - f(x, y)
# Substract down row - current row
gy[:-1, :] = matrix_img[1:, :] - matrix_img[:-1, :] #* It takes all pixels ignoring the last column.

# NOTE: The last colum in x and the las row in y remain at 0
# Because they do not have a "neighbor in front" to substract from

#TODO MAGNITUD DEL GRADIENTE Y VISUALIZACION

# Apply Pythagoras to mix gradients 
edge_magnitude = np.sqrt(gx**2 + gy**2)

# Normalize the resulting image so that its values are strictly between 0 and 255
edge_magnitude = (edge_magnitude / edge_magnitude.max()) * 255

# Thresholding to cut out background noise.
#edge_magnitude = np.where(edge_magnitude > 50, edge_magnitude, 0)

# Deploid and save result
# Create a row with two columns for comparation (before and after)
fig, ejes = plt.subplots(1, 2, figsize=(12, 6))

# Original img
ejes[0].imshow(matrix_img, cmap='gray')
ejes[0].set_title('Original Image (Grayscale)')
ejes[0].axis('off')

# Result img
ejes[1].imshow(edge_magnitude, cmap='gray')
ejes[1].set_title('Edges Detected (Finite Differences)')
ejes[1].axis('off')

plt.tight_layout()

# Save evidence
plt.savefig('img/resultado_bordes.png', dpi=300)
print("¡Process complete! The result was saved in the 'img' folder. ")

plt.show()