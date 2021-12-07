import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

import numpy as np

class ColorScaling:
    def preprocess(self, image, scale):
        scaledImage = image.copy()
        scaledImage = scaledImage * scale
        scaledImage = np.clip(scaledImage, 0, 1)
        return scaledImage

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage import io
   
    image_path = r"D:\Study\CaoHoc\LUANVAN\Dataset\AFLW2000_Extracted\image00008.jpg"
    mat_path = r"D:\Study\CaoHoc\LUANVAN\Dataset\AFLW2000_Extracted\image00008.mat"

    cs = ColorScaling()

    image2D = io.imread(image_path) / 255.
    [h, w, c] = image2D.shape

    scale = 1.4
    transformedImage = cs.preprocess(image2D, scale)

    plt.subplot(1,2,1)
    plt.imshow(image2D)
    plt.subplot(1,2,2)
    plt.imshow(transformedImage)
    plt.show()
