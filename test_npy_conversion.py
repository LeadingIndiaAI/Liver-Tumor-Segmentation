from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import cv2
import os

trainImage = "/storage/research/Intern19_v2/segmentation/TrainingData_npy/"
#trainMask = "/storage/research/Intern19_v2/segmentation/TrainingDataMasks_npy/"



def subimage_generator(image, patch_block_size, numberxy, numberz):
    """
    generate the sub images with patch_block_size
    :param image:
    :param patch_block_size:
    :param stride:
    :return:
    """
    width = np.shape(image)[1]
    height = np.shape(image)[2]
    imagez = np.shape(image)[0]
    block_width = np.array(patch_block_size)[1]
    block_height = np.array(patch_block_size)[2]
    blockz = np.array(patch_block_size)[0]

    stridewidth = (width - block_width) // (numberxy - 1)
    strideheight = (height - block_height) // (numberxy - 1)
    stridez = (imagez - blockz) // numberz


    # step 1:if image size of z is smaller than blockz,return zeros samples
    if imagez < blockz:
        nb_sub_images = numberxy * numberxy * 1
        hr_samples = np.zeros(shape=(nb_sub_images, blockz, block_width, block_height), dtype=np.float32)
        indx = 0
        for x in range(0, width - block_width + 1, stridewidth):
            for y in range(0, height - block_height + 1, strideheight):
                hr_samples[indx, 0:imagez, :, :] = image[:, x:x + block_width, y:y + block_height]
                indx += 1
        if (indx != nb_sub_images):
            print(indx)
            print(nb_sub_images)
            raise ValueError("error sub number image")
        return hr_samples

    # step 2:if stridez is bigger 1,return  numberxy * numberxy * numberz samples
    if stridez >= 1:
        nb_sub_images = numberxy * numberxy * numberz
        hr_samples = np.empty(shape=(nb_sub_images, blockz, block_width, block_height), dtype=np.float32)
        indx = 0
        for z in range(0, numberz * stridez, stridez):
            for x in range(0, width - block_width + 1, stridewidth):
                for y in range(0, height - block_height + 1, strideheight):
                    hr_samples[indx, :, :, :] = image[z:z + blockz, x:x + block_width, y:y + block_height]
                    indx += 1

        if (indx != nb_sub_images):
            print(indx)
            print(nb_sub_images)
            print("error sub number image")
        return hr_samples

    # step3: if stridez==imagez,return numberxy * numberxy * 1 samples,one is [0:blockz,:,:]
    if imagez == blockz:
        nb_sub_images = numberxy * numberxy * 1
        hr_samples = np.empty(shape=(nb_sub_images, blockz, block_width, block_height), dtype=np.float32)
        indx = 0
        for x in range(0, width - block_width + 1, stridewidth):
            for y in range(0, height - block_height + 1, strideheight):
                hr_samples[indx, :, :, :] = image[:, x:x + block_width, y:y + block_height]
                indx += 1
        if (indx != nb_sub_images):
            print("error sub number image")
            print(indx)
            print(nb_sub_images)
        return hr_samples

    # step4: if stridez==0,return numberxy * numberxy * 2 samples,one is [0:blockz,:,:],two is [-blockz-1:-1,:,:]
    if stridez == 0:
        nb_sub_images = numberxy * numberxy * 2
        hr_samples = np.empty(shape=(nb_sub_images, blockz, block_width, block_height), dtype=np.float32)
        indx = 0
        for x in range(0, width - block_width + 1, stridewidth):
            for y in range(0, height - block_height + 1, strideheight):
                hr_samples[indx, :, :, :] = image[0:blockz, x:x + block_width, y:y + block_height]
                indx += 1
                hr_samples[indx, :, :, :] = image[-blockz - 1:-1, x:x + block_width, y:y + block_height]
                indx += 1
        if (indx != nb_sub_images):
            print("error sub number image")
        return hr_samples


def make_patch(image, patch_block_size, numberxy, numberz):
    """
    make number patch
    :param image:[depth,512,512]
    :param patch_block: such as[64,128,128]
    :return:[samples,64,128,128]
    expand the dimension z range the subimage:[startpostion-blockz//2:endpostion+blockz//2,:,:]
    """

    image_subsample = subimage_generator(image=image, patch_block_size=patch_block_size, numberxy=numberxy,
                                         numberz=numberz)
    return image_subsample


def load_itk(filename):
    """
    load mhd files and normalization 0-255
    :param filename:
    :return:
    """
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    # Reads the image using SimpleITK
    itkimage = rescalFilt.Execute(sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32))
    return itkimage


def gen_image_mask(srcimg, index, shape, numberxy, numberz):

    sub_srcimages = make_patch(srcimg, patch_block_size=shape, numberxy=numberxy, numberz=numberz)

    #only save subimages (numberxy*numberxy*numberz,16, 256, 256)
    samples, imagez = np.shape(sub_srcimages)[0], np.shape(sub_srcimages)[1]
    for j in range(samples):
        sub_images = sub_srcimages.astype(np.float32)
        sub_images = np.clip(sub_images, 0, 255).astype('uint8')
            
        filepath = trainImage+"\\" + str(index) + "_" +str(j) + ".npy"
        image = sub_srcimages[j, :, :, :]
        image = image.astype(np.float32)
        image = np.clip(image, 0, 255).astype('uint8')
        np.save(filepath, image)
            #np.save(filepath2, sub_masks[j, :, :, :])


def preparetestdata():
    for i in range(0, 131, 1):
        src = load_itk("/storage/research/Intern19_v2/segmentation/Team 1/TrainingData/volume-" + str(i) + ".nii")
        srcimg = sitk.GetArrayFromImage(src)
        gen_image_mask(srcimg, i, shape=(16, 256, 256), numberxy=5, numberz=10)

preparetestdata()
