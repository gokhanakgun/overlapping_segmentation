import cv2
from scipy.signal import wiener
from scipy.ndimage import binary_fill_holes
import numpy as np
import math
from skimage.measure import label, regionprops
from skimage import morphology


def segmentNuclei(I):
    if len(I.shape) == 3:
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    minAcceptableBoundDiff = 15

    cells = {'MinSize': 110, 'MinMean': 60, 'MaxMean': 150, 'MinSolidity': 0.9}

    lowN = math.floor(cells['MinMean'] / 10) * 10
    highN = math.ceil(cells['MaxMean'] / 10) * 10

    I = wiener(I / 255, (5, 5))
    I = np.round(np.multiply(I, 255)).astype(np.uint8)

    nuclei = np.zeros(I.shape)

    allPixels = I.shape[0] * I.shape[1]

    for thresh in np.arange(lowN, highN + 1, 10):
        ret, binaryImage = cv2.threshold(I, thresh + 0.5, 1, cv2.THRESH_BINARY_INV)
        if np.sum(binaryImage) > allPixels / 5:
            break
        blobs = label(binaryImage, connectivity=2)
        regProp = regionprops(blobs, cache=False)

        addTheseRegions = np.ones(len(regProp))

        removeHighSTDTooConcaveTooSmallTooLargeBlobs = [(region.area < cells['MinSize']) or (region.solidity < cells['MinSolidity']) for region in regProp]
        addTheseRegions[removeHighSTDTooConcaveTooSmallTooLargeBlobs] = 0
        pixelsAlreadyInNuclei = (blobs != 0) & (nuclei != 0)
        blobsAlreadyInNuclei = np.unique(blobs[pixelsAlreadyInNuclei])

        nuclei = label(nuclei, connectivity=2)
        nucRegProp = regionprops(nuclei)

        if blobsAlreadyInNuclei.size != 0:
            for j in blobsAlreadyInNuclei:
                intersectWithThese = np.unique(nuclei[blobs == j])
                if regProp[j-1].solidity < max([nucRegProp[i-1].solidity for i in np.where(intersectWithThese > 0)[0]]):
                    addTheseRegions[j-1] = 0

        for i in np.where(addTheseRegions == 1)[0]:
            for px in regProp[i].coords:
                nuclei[px[0], px[1]] = 1

        nuclei = nuclei != 0


    nuclei = morphology.remove_small_objects(binary_fill_holes(nuclei), math.floor(cells['MinSize']), connectivity=2)
    dilatedSeg = morphology.dilation(nuclei, morphology.disk(1))
    regionsLabel = label(nuclei, connectivity=2)
    dilatedRegionsLabel, numOfDilatedRegions = label(dilatedSeg, connectivity=2, return_num=True)
    for i in range(1, numOfDilatedRegions + 1):
        if(len(np.unique(regionsLabel[dilatedRegionsLabel == i])) >= 3):
            dilatedSeg[dilatedRegionsLabel == i] = nuclei[dilatedRegionsLabel == i]

    nuclei = dilatedSeg
    labels, totalLabels = label(nuclei, connectivity=2, return_num=True)
    meanDiff = np.zeros(totalLabels)

    for i in range(1, totalLabels + 1):
        dilatedNucleus = morphology.dilation(labels == i, morphology.disk(3))
        meanDiff[i - 1] = np.mean(I[dilatedNucleus & (labels == 0)]) - np.mean(I[labels == i])

    regProp = regionprops(labels)
    removeNotHighMeanDiff = meanDiff < minAcceptableBoundDiff

    for i in np.where(removeNotHighMeanDiff == 1)[0]:
        for px in regProp[i].coords:
            nuclei[px[0], px[1]] = 0

    L, num = label(nuclei, connectivity=2, return_num=True)

    contourArea = [0] * num
    contourSize = [0] * num

    for i in range(num):
        contourArea[i] = (L == (i + 1))
        contourSize[i] = np.count_nonzero(contourArea[i])

    return nuclei, contourArea, contourSize





