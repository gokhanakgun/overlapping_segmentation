from scipy.signal import wiener
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import cv2
from skimage import morphology
from skimage.measure import label, regionprops
import numpy as np
import mahotas


def segmentClumps(I):
    if len(I.shape) == 3:
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    q = 0.06
    W = np.rint(wiener(I / 255, (5, 5)) * 255)
    gm = GaussianMixture(n_components=2).fit(W.flatten('F').reshape(-1, 1))
    i1 = norm.ppf(q, loc=gm.means_[0][0], scale=gm.covariances_[0][0][0]**0.5)
    i2 = norm.ppf(q, loc=gm.means_[1][0], scale=gm.covariances_[1][0][0]**0.5)
    while (max(i1, i2) < 150) or (gm.covariances_[0][0][0] == gm.covariances_[1][0][0]):
        gm = GaussianMixture(n_components=2).fit(W.flatten('F').reshape(-1, 1))
        i1 = norm.ppf(q, loc=gm.means_[0][0], scale=gm.covariances_[0][0][0] ** 0.5)
        i2 = norm.ppf(q, loc=gm.means_[1][0], scale=gm.covariances_[1][0][0] ** 0.5)
    minMeanIntensity = max(norm.ppf(0.0001, loc=gm.means_[0], scale=gm.covariances_[0] ** 0.5), norm.ppf(0.0001, loc=gm.means_[1], scale=gm.covariances_[1] ** 0.5))
    ret, clump = cv2.threshold(I, max(i1, i2), 1, cv2.THRESH_BINARY_INV)
    clump = morphology.binary_opening(clump, mahotas.disk(5))
    clump = morphology.remove_small_objects(clump, 2000, connectivity=2)
    lclump = label(clump, connectivity=2)
    clumpProps = regionprops(lclump, W)
    for region in clumpProps:
        if region.mean_intensity > minMeanIntensity:
            for px in region.coords:
                clump[px[0], px[1]] = 0
    L, num = label(clump, connectivity=2, return_num=True)
    allClumps = [None] * num
    for i in range(num):
        allClumps[i] = np.logical_not(morphology.remove_small_objects(np.logical_not(morphology.binary_dilation(L == (i+1), morphology.disk(1))), 20, connectivity=2))
    return allClumps






