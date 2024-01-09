from utils import *
from PIL import Image

image = Image.open('Image.png')
image = image.convert("RGB")
imgdata = np.array(image)
plt.imshow(imgdata)
plt.show()

l, w, h = imgdata.shape
imgSize = imgdata.shape
imgResize = (l * w, h)
imgdata = imgdata.reshape(imgResize)

# K=20
finalD20, finalMean20, count20 = KMNS(imgdata, 20)
plotSelect(finalMean20)
Compressed20 = reconstructImg(imgSize, finalD20, finalMean20)
plt.imshow(Compressed20)
plt.show()

# K=10
finalD10, finalMean10, count10 = KMNS(imgdata, 10)
plotSelect(finalMean10)
Compressed10 = reconstructImg(imgSize, finalD10, finalMean10)
plt.imshow(Compressed10)
plt.show()

# K=5
finalD5, finalMean5, count5 = KMNS(imgdata, 5)
Compressed5 = reconstructImg(imgSize, finalD5, finalMean5)
plotSelect(finalMean5)
plt.imshow(Compressed5)
plt.show()

# K=3
finalD3, finalMean3, count3 = KMNS(imgdata, 3)
Compressed3 = reconstructImg(imgSize, finalD3, finalMean3)
plt.imshow(Compressed3)
plt.show()
