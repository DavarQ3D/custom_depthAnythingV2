import os
import sys
from custom_utils import *
from enum import Enum

if sys.platform == 'darwin':
    import coremltools as ct
else:
    ct = None

class Dataset(Enum):
    IPHONE = 1
    NYU2 = 2

#=============================================================================================================

if __name__ == '__main__':

    outdir = "./data/outputs"
    os.makedirs(outdir, exist_ok=True)

    #--------------------- settings
    dtSet = Dataset.NYU2
    inputPath = f"./data/iphone/" if dtSet == Dataset.IPHONE else f"./data/nyu2_test/"
    checkIfSynced = False and dtSet == Dataset.IPHONE
    sampleToTest = 6

    encoder = "vits"
    useCoreML = False and ct is not None
    
    weightedLsq = True
    fitOnDepth = False
    k_hi = 2.5 if dtSet == Dataset.IPHONE else 3.0

    makeSquareInput = True
    borderType = cv2.BORDER_CONSTANT
    
    normalizeVisualError = False
    showVisuals = True

    #--------------------- load models
    torch_model = loadTorchModel(f'checkpoints/depth_anything_v2_{encoder}.pth', encoder)              # torch

    if makeSquareInput:
        rows = 518
        cols = 518
    else:
        if dtSet == Dataset.IPHONE:
            rows = 686
            cols = 518
        else:
            rows = 518
            cols = 686

    mlProgram = ct.models.CompiledMLModel(f"./checkpoints/custom_vits_F16_{rows}_{cols}.mlmodelc") if useCoreML else None  # coreml

    #------------------ inference loop
    #------------------------------------------------------------------
    numFiles = len(os.listdir(inputPath)) // 2
    totalError = 0.0
    meanErr = 0.0
    sampleWithLowestError = 0
    samplewithHighestError = 0
    minRMSE = float('inf')
    maxRMSE = float('-inf')
    sampleCounter = 0
    start = max(0, sampleToTest - 3) if checkIfSynced else 0

    for idx in range(start, numFiles):

        if checkIfSynced:
            print("sample to test:", sampleToTest)
            
        print('\n'"========================================")
        print(f'============= sample --> {idx} =============')
        print("========================================", '\n')

        if dtSet == Dataset.IPHONE:

            rgbFileName = f"RGB_{idx:04d}.png"
            rgbPath = inputPath + rgbFileName 
            raw_image = cv2.imread(rgbPath)
            raw_image = cv2.rotate(raw_image, cv2.ROTATE_90_CLOCKWISE)

            index = sampleToTest if checkIfSynced else idx
            gtPath = inputPath + f"ARKit_DepthValues_{index:04d}.txt" 
            gt = loadMatrixFromFile(gtPath)
            gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)

            if checkIfSynced:
                overlayInputs(raw_image, gt)
                continue

        elif dtSet == Dataset.NYU2:

            rgbFileName = f"{idx:05d}_colors.png"
            rgbPath = inputPath + rgbFileName 
            raw_image = cv2.imread(rgbPath)

            gtPath = inputPath + f"{idx:05d}_depth.png"
            gt = cv2.imread(gtPath, cv2.IMREAD_UNCHANGED)
            gt = gt.astype(np.float64) / 1000.0           # scale to meters

            margin = 8   # remove white margin
            raw_image = raw_image[margin:-margin, margin:-margin, :]
            gt = gt[margin:-margin, margin:-margin]

        else:
            raise ValueError("Unsupported dataset")

        pred_disparity, gt, cropped = handlePredictionSteps(raw_image, gt, makeSquareInput, borderType, useCoreML, mlProgram, torch_model)

        #--------------------- fit in disparity

        gt_disparity = 1 / (gt + 1e-8)               # convert depth to disparity (inverse depth)

        if weightedLsq: 
            scale, shift, mask = weightedLeastSquared(pred_disparity, gt_disparity, guessInitPrms=True, k_lo=0.2, k_hi=k_hi, num_iters=10, fit_shift=True, verbose=False)
        else: 
            scale, shift, mask = estimateParametersRANSAC(pred_disparity, gt_disparity) 

        print("Scale:", fp(scale), ", Shift:", fp(shift), '\n')

        pred_disparity = scale * pred_disparity + shift

        pred = 1 / (pred_disparity + 1e-8)           # convert back to depth

        #--------------------- fit in depth

        if fitOnDepth:
           
            if weightedLsq: 
                scale, shift, mask = weightedLeastSquared(pred, gt, guessInitPrms=True, k_lo=0.2, k_hi=k_hi, num_iters=10, fit_shift=True, verbose=False)
            else: 
                scale, shift, mask = estimateParametersRANSAC(pred, gt) 

            pred = scale * pred + shift            

        #-----------------------------------------------------------------------
        #-----------------------------------------------------------------------

        visualRes, rmse = analyzeAndPrepVis(cropped, mask, gt, pred, mode="color", normalizeError=normalizeVisualError)

        if rmse < minRMSE:
            minRMSE = rmse
            sampleWithLowestError = idx
        if rmse > maxRMSE:
            maxRMSE = rmse
            samplewithHighestError = idx

        totalError += rmse
        meanErr = totalError / (sampleCounter + 1)
        sampleCounter += 1
        print("\nmean across all images so far --> RMSE =", fp(meanErr, 6))
        print("\nimage with lowest error:", sampleWithLowestError, "--> RMSE =", fp(minRMSE, 6))
        print("image with highest error:", samplewithHighestError, "--> RMSE =", fp(maxRMSE, 6))

        if showVisuals:
            if dtSet == Dataset.IPHONE:
                ssc = 2.5 if ct else 2
            elif dtSet == Dataset.NYU2:
                ssc = 0.6
            else:
                raise ValueError("Unsupported dataset")
            visualRes = cv2.resize(visualRes, None, fx=ssc, fy=ssc, interpolation=cv2.INTER_CUBIC)
            displayImage("visualRes", visualRes)


        





        





