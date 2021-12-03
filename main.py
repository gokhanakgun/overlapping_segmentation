
from segmentNuclei import segmentNuclei
from segmentClumps import segmentClumps
from segmentCytoplasms import segmentCytoplasms
import cv2


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    I = cv2.imread('frame022.png')
    nuclei, area, size = segmentNuclei(I)
    allClumps = segmentClumps(I)
    cytoplasms = segmentCytoplasms(I, size, area, allClumps)



