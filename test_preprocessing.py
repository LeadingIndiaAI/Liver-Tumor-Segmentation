import cv2
import numpy as np
import glob

class LITSPreprocessor(object):
    def __init__(self, image):
        assert len(image.shape) == 3, '==> InputError'
        self.image = image
        self.shape = image.shape
        self.depth = image.shape[-1]

    def transform_ctdata(self, windowWidth, windowCenter, normal=False):
        """
        return: trucated image according to window center and window width
        """
        minWindow = float(windowCenter) - 0.5*float(windowWidth)
        newimg = (self.image - minWindow) / float(windowWidth)
        newimg[newimg < 0] = 0
        newimg[newimg > 1] = 1
        if not normal:
            newimg = (newimg * 255).astype('uint8')
        return newimg

    def resize_3d(self, width, height):
        """
        return: resized image in shape [depth, width, height]
        """
        if not self.shape[:2] == (width, height):
            newimg = [cv2.resize(self.image[:,:,i], (height, width)) for i in range(self.depth)]
            newimg = np.array(newimg)
        else:
            newimg = self.image.transpose(2,0,1)
        return newimg

def main():
    # test
    img_path=glob.glob("C:/Users/stdtest/Documents/npy_files/*.npy")
    for i in range(len(img_path)):
        image = np.load(img_path[i])
        print(image.shape)
        lits = LITSPreprocessor(image)

    # the proper ct value for observe liver is 50~70
        image = lits.transform_ctdata(20, 60)
        image = lits.resize_3d(128, 128)
        print(image.shape)

if __name__ == '__main__':
	main()
