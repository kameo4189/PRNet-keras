import os, sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__parent__ = os.path.dirname(__location__)
sys.path.append(__location__)
sys.path.append(__parent__)

from preprocessing.image2D_warping import Image2DWarping
from preprocessing.image3D_transforming import Image3DTransforming
from detecting.boundingbox_detecting import BoundingboxDetecting
from configure import config

class ImageMeshWarping:

    i3t = Image3DTransforming()
    i2w = Image2DWarping()
    bd = BoundingboxDetecting()

    def preprocess(self, image, meshInfo, 
        outHeight=config.IMAGE_HEIGHT, outWidth=config.IMAGE_WIDTH, randomPertubeBB=True):

        [h, w, _] = image.shape

        kpts = meshInfo.vertices[meshInfo.kptIdxs, :]
        bbInfo = self.bd.detect(kpts, h, w, randomPertubeBB)

        (wrappedImage, tform) = self.i2w.preprocess(image, bbInfo, outHeight, outWidth)
        wrappedMesh = self.i3t.preprocessTform(meshInfo, tform, h, w, outHeight, outWidth)

        return (wrappedImage, wrappedMesh, tform)

    def preprocesses(self, images, meshInfos, 
        outHeight=config.IMAGE_HEIGHT, outWidth=config.IMAGE_WIDTH, randomPertubeBB=True):

        assert len(images) == len(meshInfos), "Size of image list isn't equal mesh list"
        warpedData = [self.preprocess(image, meshInfo, outHeight, outWidth, randomPertubeBB) 
            for image, meshInfo in list(zip(images, meshInfos))]
        (warpedImages, warpedMeshInfos, tforms) = zip(*warpedData)
        return (warpedImages, warpedMeshInfos, tforms)

if __name__ == "__main__":
    import numpy as np
    from skimage import io
    import matplotlib.pyplot as plt
    from preprocessing.image3D_extracting import Image3DExtracting
    from detecting.boundingbox_detecting import BoundingboxDetecting
    from preprocessing.image3D_to_2D import Image3DTo2D
    from preprocessing.geometric_transforming import GeometricTransforming
    from matplotlib.patches import Rectangle
    from util import file_methods

    dataPath = r"K:\Study\CaoHoc\LuanVan\Dataset\AFLW2000"
    imageWithMatPaths = file_methods.getImageWithMatList(dataPath)
    np.random.shuffle(imageWithMatPaths)
    # (imagePaths, matPaths) = zip(*imageWithMatPaths)

    for imageMat in imageWithMatPaths:
        (image_path, mat_path) = imageMat
            
        image2D = io.imread(image_path)/255.
        [h, w, c] = image2D.shape

        i3t2 = Image3DTo2D()
        i3e = Image3DExtracting()
        imw = ImageMeshWarping()
        bd = BoundingboxDetecting()
        gt = GeometricTransforming()

        meshInfo = i3e.preprocess(mat_path)

        [h, w, _] = image2D.shape

        kpts = meshInfo.vertices[meshInfo.kptIdxs, :]
        bbInfo = bd.detect(kpts, h, w, True, normalBB=True)
        (left, right, top, bottom) = bbInfo


        # # Display the image
        # plt.imshow(image2D)
        # # Create a Rectangle patch
        # plt.gca().add_patch(Rectangle((left, top),right-left,bottom-top,linewidth=1,edgecolor='r',facecolor='none'))
        # plt.show()



        (wrappedImage, wrappedMesh, tform) = imw.preprocess(image2D, meshInfo)
        [h, w, c] = wrappedImage.shape

        plt.subplot(1, 2, 1)
        plt.imshow(image2D)
        plt.gca().add_patch(Rectangle((left, top),right-left,bottom-top,linewidth=1,edgecolor='r',facecolor='none'))
        plt.subplot(1, 2, 2)
        plt.imshow(wrappedImage)
        plt.show(block=True)

        # scale = 0.8
        # rotation = 30
        # translation = (0.1 * w, 0.05 * h)
        # (transformedImage, transformedMeshInfo, tform) = gt.preprocess(wrappedImage, wrappedMesh, scale, rotation, translation, True)
        # transformedImageFace = i3t2.preprocess(transformedMeshInfo, h, w, False)
        # transformedImageFaceKpts = i3t2.drawKeypoints(transformedImageFace, transformedMeshInfo)

        # stackImage = np.concatenate((wrappedImage, transformedImage, transformedImageFaceKpts), axis=1)
        
        # plt.subplot(2, 1, 1)
        # plt.imshow(stackImage)
        # plt.subplot(2, 1, 2)
        # plt.imshow(transformedImage)
        # plt.imshow(transformedImageFaceKpts, alpha=.5)
        # plt.show(block=True)
