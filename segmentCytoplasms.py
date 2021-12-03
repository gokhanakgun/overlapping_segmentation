import numpy as np
from skimage.measure import label, regionprops, regionprops_table
from skimage import morphology
from skimage.draw import polygon2mask
from skimage.morphology import reconstruction
from scipy.ndimage import binary_fill_holes
import math
import cv2
from bresenham import bresenham
from itertools import compress
from scipy.signal import wiener, savgol_filter
from skfuzzy.membership import sigmf


def segmentCytoplasms(I, contourSize, contourArea, allClumps, alpha=1.75, beta=20, loadedVolumeImages=None):
    if len(I.shape) == 3:
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    gridWidth = 8
    includeDetectedNucleus = np.zeros(len(contourArea)).astype(int)
    for i in range(includeDetectedNucleus.shape[0]):
        for j in range(len(allClumps)):
            if np.count_nonzero(np.logical_and(contourArea[i], allClumps[j])) >= 0.7 * contourSize[i]:
                includeDetectedNucleus[i] = j + 1
                break

    nucleusGridSquareRatio = 0.1
    squareArea = gridWidth * gridWidth

    cytoplasms = [None] * len(contourSize)
    canvas = np.full(allClumps[0].shape, False)
    for c in range(len(allClumps)):
        nucleiInClump = np.where(includeDetectedNucleus == c + 1)
        numOfNuclei = len(nucleiInClump[0])
        if len(nucleiInClump[0]) == 0:
            continue
        sClump = regionprops(allClumps[c].astype(int))
        W1 = max(math.floor(sClump[0].bbox[0] - gridWidth), 0)
        W2 = min(math.ceil(sClump[0].bbox[2] + gridWidth), canvas.shape[0] - 1)
        H1 = max(math.floor(sClump[0].bbox[1] - gridWidth), 0)
        H2 = min(math.ceil(sClump[0].bbox[3] + gridWidth), canvas.shape[1] - 1)

        if W1 == 0:
            W2 = W2 - ((W2 - W1 + 1) % gridWidth)
        else:
            W1 = W1 + ((W2 - W1 + 1) % gridWidth)
        if H1 == 0:
            H2 = H2 - ((H2 - H1 + 1) % gridWidth)
        else:
            H1 = H1 + ((H2 - H1 + 1) % gridWidth)


        gridSquareNucleiEffect = np.zeros((math.floor((W2 - W1 + 1) / gridWidth), math.floor((H2 - H1 + 1) / gridWidth), numOfNuclei))

        # Loaded volume images here if exists

        nucleusGridSquares = [None] * numOfNuclei
        nucleusCenterCoord = [None] * numOfNuclei
        nucleusCenterCoordOrig = [None] * numOfNuclei



        for n in range(numOfNuclei):
            nucleusGridSquares[n] = []
            nucleusInClumpArea = contourArea[nucleiInClump[0][n]][W1:W2+1, H1:H2+1]
            nucleusProp = regionprops_table(nucleusInClumpArea.astype(int), properties=['centroid'])
            nucleusCenterCoordOrig[n] = (int(np.round(nucleusProp['centroid-0'][0])), int(np.round(nucleusProp['centroid-1'][0])))
            nucleusCenterCoord[n] = (int(np.floor(nucleusProp['centroid-0'][0] / gridWidth)), int(np.floor(nucleusProp['centroid-1'][0] / gridWidth)))
            for i in range((W2 - W1 + 1) // gridWidth):
                for j in range((H2 - H1 + 1) // gridWidth):
                    if np.count_nonzero(nucleusInClumpArea[i * gridWidth:(i + 1) * gridWidth, j * gridWidth:(j + 1) * gridWidth]) > nucleusGridSquareRatio * squareArea:
                        nucleusGridSquares[n].append([i, j])
            if len(nucleusGridSquares[n]) == 0:
                return
            for i in range((W2 - W1 + 1) // gridWidth):
                for j in range((H2 - H1 + 1) // gridWidth):
                    for k in range(len(nucleusGridSquares[n])):
                        gridSquareNucleiEffect[i, j, n] += math.exp(-1 * (np.linalg.norm([i - nucleusGridSquares[n][k][0], j - nucleusGridSquares[n][k][1]]) ** 2 / (2 * alpha ** 2)))
                    gridSquareNucleiEffect[i, j, n] /= len(nucleusGridSquares[n])

        sumOfNucleiEffect = np.sum(gridSquareNucleiEffect, axis=2)
        shrinkedClump = cv2.resize(allClumps[c][W1:W2+1, H1:H2+1].astype('float32'), dsize=((H2 - H1 + 1) // gridWidth, (W2 - W1 + 1) // gridWidth), interpolation=cv2.INTER_LINEAR)
        shrinkedClump = shrinkedClump > 0.5
        for n in range(numOfNuclei):
            thisNucleusEffect = beta * gridSquareNucleiEffect[:, :, n] - sumOfNucleiEffect
            cellBlobs = binary_fill_holes(thisNucleusEffect > 0.0)
            cellBlobs = morphology.binary_opening(np.logical_and(cellBlobs, shrinkedClump), morphology.diamond(1))

            needToContinue = True
            while needToContinue:
                needToContinue = False
                pixX, pixY = np.nonzero(cellBlobs)
                for i in range(len(pixX)):
                    inBetweenIndexesList = list(bresenham(pixX[i], pixY[i], nucleusCenterCoord[n][0], nucleusCenterCoord[n][1]))
                    inBetweenXs = [c[0] for c in inBetweenIndexesList]
                    inBetweenYs = [c[1] for c in inBetweenIndexesList]
                    if not np.all(cellBlobs[inBetweenXs, inBetweenYs]):
                        cellBlobs[pixX[i], pixY[i]] = False
                        needToContinue = True

            cytoplasms[nucleiInClump[0][n]] = np.copy(canvas)
            cytoplasms[nucleiInClump[0][n]][W1:W2+1, H1:H2+1] = np.copy(np.logical_and(allClumps[c][W1:W2+1, H1:H2+1], cv2.resize(cellBlobs.astype('float32'), dsize=(cellBlobs.shape[1] * gridWidth, cellBlobs.shape[0] * gridWidth), interpolation=cv2.INTER_LINEAR) > 0.5)) #(cellBlobs.shape[0] * gridWidth, cellBlobs.shape[1] * gridWidth))))

            labeledCellBlobs = label(cytoplasms[nucleiInClump[0][n]][W1:W2+1, H1:H2+1], connectivity=1)
            cytoplasms[nucleiInClump[0][n]][W1:W2+1, H1:H2+1] = labeledCellBlobs == labeledCellBlobs[nucleusCenterCoordOrig[n][0], nucleusCenterCoordOrig[n][1]]

            if not labeledCellBlobs[nucleusCenterCoordOrig[n][0], nucleusCenterCoordOrig[n][1]]:
                print("ERROR")

    cytoplasms = list(compress(cytoplasms, includeDetectedNucleus != 0))

    discUnits, angleSin = np.meshgrid(np.arange(0, 301, 1), np.sin(np.arange(0, 2 * np.pi - 0.00001, 2 * np.pi / 360)))
    xInc = np.round(discUnits * angleSin)
    _, angleCos = np.meshgrid(np.arange(0, 301, 1), np.cos(np.arange(0, 2 * np.pi, 2 * np.pi / 360)))
    yInc = np.round(discUnits * angleCos)

    contourArea = list(compress(contourArea, includeDetectedNucleus != 0))
    includeDetectedNucleus = list(compress(includeDetectedNucleus, includeDetectedNucleus != 0))

    cytoIsChanging = np.full(len(cytoplasms), True)


    for X in range(20):
        if not np.any(cytoIsChanging):
            break

        if X == 0:
            for i in range(len(contourArea)):

                expanded = morphology.dilation(contourArea[i], morphology.disk(7))

                expanded[contourArea[i]] = False

                I[morphology.dilation(contourArea[i], morphology.disk(1))] = np.round(np.mean(I[expanded]))

            meanNucleiPix = regionprops_table(label(np.array(sum(contourArea), dtype=bool)), I, properties=['mean_intensity'])
            I = np.maximum(I, np.min(np.array([meanNucleiPix['mean_intensity']], dtype=int)))

            I = wiener(I / 255, (5, 5))
            I = np.round(np.multiply(I, 255)).astype(np.uint8) / 255

            intensityImage = np.copy(I)
            foreground = np.any(np.dstack(allClumps), axis=2)
            intensityImage[np.logical_not(foreground)] = 1


        a = 10
        c = .5
        rlm = np.zeros((len(cytoplasms), 360))
        poses = np.zeros((len(cytoplasms), 360))

        for i in range(len(cytoplasms)):
            if not cytoIsChanging[i]:
                continue

            cytoplasmProp = regionprops(cytoplasms[i].astype(int))
            nucleusCent = regionprops(contourArea[i].astype(int))
            nucY = round(nucleusCent[0].centroid[1])
            nucX = round(nucleusCent[0].centroid[0])

            y1 = cytoplasmProp[0].bbox[1]
            y2 = cytoplasmProp[0].bbox[3] - 1
            x1 = cytoplasmProp[0].bbox[0]
            x2 = cytoplasmProp[0].bbox[2] - 1
            allX = (nucX - xInc).astype(int)
            allY = (nucY + yInc).astype(int)

            inImage = np.logical_and(np.logical_and((allX < intensityImage.shape[0]), (allX >= 0)), np.logical_and((allY < intensityImage.shape[1]), (allY >= 0)))
            inCytoplasm = np.logical_and(np.logical_and((allX <= x2), (allX >= x1)), np.logical_and((allY <= y2), (allY >= y1)))

            inRangeInd = np.column_stack((allX[inCytoplasm], allY[inCytoplasm]))
            inCytoplasm[inCytoplasm] = (inRangeInd[:, None] == cytoplasmProp[0].coords).all(-1).any(-1)

            allBoundaryPoints = np.zeros((360, 2))
            previousBoundaryPoints = np.zeros((360, 2))
            includeBoundaryPoints = np.full(360, True)
            boundaryPointChanged = np.full(360, True)

            for r in range(360):
                if X == 0:
                    firstPixOutCyto = np.where(inCytoplasm[r, :] == False)[0][0]
                    lastPixInCyto = firstPixOutCyto - 1
                    rayLength = min(2 * (lastPixInCyto + 1), allX.shape[1])
                    outImg = np.where(inImage[r, :] == False)
                    if outImg[0].size != 0:
                        rayLength = min(rayLength, outImg[0][0]) # To be fixed
                    rayWeightVector = 1 - abs((np.linspace(1, rayLength, rayLength)) - (lastPixInCyto + 1)) / (lastPixInCyto + 1)
                    rayWeightVector = sigmf(rayWeightVector, c, a)
                else:
                    firstPixOutCyto = np.where(inCytoplasm[r, :] == False)[0][0]
                    lastPixInCyto = firstPixOutCyto - 1
                    rayLength = lastPixInCyto + 1
                    outImg = np.where(inImage[r, :] == False)
                    if outImg[0].size != 0:
                        rayLength = min(rayLength, outImg[0][0])
                    rayWeightVector = np.linspace(1, rayLength + 1, rayLength) / rayLength
                    rayWeightVector = sigmf(rayWeightVector, c, a)

                rayX = np.copy(allX[r, :rayLength])
                rayY = np.copy(allY[r, :rayLength])
                rlm[i, r] = rayLength
                previousBoundaryPoints[r, 0] = np.copy(allX[r, lastPixInCyto])
                previousBoundaryPoints[r, 1] = np.copy(allY[r, lastPixInCyto])
                rayPixels = np.copy(intensityImage[rayX, rayY])
                tmp = np.pad(rayPixels, (2, 2), 'constant', constant_values=(0, 1))

                diff = np.zeros(rayPixels.shape[0])

                for xR in range(2, tmp.shape[0] - 2):
                    diff[xR - 2] = tmp[xR - 1] - tmp[xR + 1]

                pos = np.argmin(rayWeightVector * diff)
                poses[i, r] = pos

                if rayLength - pos <= 5:
                    boundaryPointChanged[r] = False
                allBoundaryPoints[r, 0] = rayX[pos]
                allBoundaryPoints[r, 1] = rayY[pos]

            allBoundaryPoints = allBoundaryPoints[includeBoundaryPoints, :]


            fittingPolynomialOrder = 3
            fittingWindowWidth = 2 * allBoundaryPoints.shape[0] // 8 + 1

            fittingMargin = fittingWindowWidth // 2
            smoothX = savgol_filter(np.concatenate((allBoundaryPoints[allBoundaryPoints.shape[0]-fittingMargin:, 1],
                                                   allBoundaryPoints[:, 1], allBoundaryPoints[:fittingMargin, 1])) + 1,
                                    fittingWindowWidth, fittingPolynomialOrder)
            smoothY = savgol_filter(np.concatenate((allBoundaryPoints[allBoundaryPoints.shape[0] - fittingMargin:, 0],
                                                   allBoundaryPoints[:, 0], allBoundaryPoints[:fittingMargin, 0])) + 1,
                                    fittingWindowWidth, fittingPolynomialOrder)
            smoothX = smoothX[fittingMargin:(-1 * fittingMargin)]
            smoothY = smoothY[fittingMargin:(-1 * fittingMargin)]

            distSmoothBoundary = np.stack((smoothY, smoothX), axis=1) - allBoundaryPoints - 1
            distValue = np.sqrt(np.sum(np.square(distSmoothBoundary), axis=1))

            newBoundaryPoints = allBoundaryPoints[distValue <= 10, :]
            fittingWindowWidth = 2 * (newBoundaryPoints.shape[0] // 16) + 1
            fittingMargin = fittingWindowWidth // 2

            smoothX = savgol_filter(np.concatenate((newBoundaryPoints[newBoundaryPoints.shape[0] - fittingMargin:, 1],
                                                   newBoundaryPoints[:, 1], newBoundaryPoints[:fittingMargin, 1])),
                                    fittingWindowWidth, fittingPolynomialOrder)
            smoothY = savgol_filter(np.concatenate((newBoundaryPoints[newBoundaryPoints.shape[0] - fittingMargin:, 0],
                                                   newBoundaryPoints[:, 0], newBoundaryPoints[:fittingMargin, 0])),
                                    fittingWindowWidth, fittingPolynomialOrder)
            smoothX = smoothX[fittingMargin:(-1 * fittingMargin)]
            smoothY = smoothY[fittingMargin:(-1 * fittingMargin)]

            newCyto = polygon2mask((intensityImage.shape[0], intensityImage.shape[1]), np.stack((smoothY, smoothX), axis=-1))

            addX = np.round(smoothX).astype(int)
            addY = np.round(smoothY).astype(int)
            addX = np.minimum(addX, newCyto.shape[1] - 1)
            addY = np.minimum(addY, newCyto.shape[0] - 1)
            addX = np.maximum(addX, 0)
            addY = np.maximum(addY, 0)
            new_array = [tuple(row) for row in np.stack((addY, addX), axis=-1)]
            uniques = np.unique(new_array)

            newCyto[addY, addX] = True
            newCyto = np.logical_and(newCyto, allClumps[includeDetectedNucleus[i]-1])

            newCyto = np.logical_or(reconstruction(newCyto, contourArea[i]), contourArea[i])
            newAndPreDiff = np.logical_or(np.logical_and(newCyto, np.logical_not(cytoplasms[i])), np.logical_and(np.logical_not(newCyto), cytoplasms[i]))

            if (np.count_nonzero(newAndPreDiff) / np.count_nonzero(cytoplasms[i])) < 0.01:
                cytoIsChanging[i] = False

            cytoplasms[i] = newCyto

    return cytoplasms


















