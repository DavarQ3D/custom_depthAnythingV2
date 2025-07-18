import os
import sys
from custom_utils import *

if sys.platform == 'darwin':
    import coremltools as ct
else:
    ct = None

#=============================================================================================================

if __name__ == '__main__':

    outdir = "./data/outputs"
    os.makedirs(outdir, exist_ok=True)

    #--------------------- settings
    inputPath = f"./data/iphone/"
    checkIfSynced = False
    sampleToTest = 6

    encoder = "vits"
    useCoreML = False and ct is not None
    
    weightedLsq = True
    inlier_bottom = 0.02 
    outlier_cap = 0.1 
    num_iters = 10 
    fit_shift = True 
    verbose = False
    guessInitPrms = True
    
    makeSquareInput = True
    borderType = cv2.BORDER_CONSTANT
    
    normalizeVisualError = False
    showVisuals = True

    #--------------------- load models
    torch_model = loadTorchModel(f'checkpoints/depth_anything_v2_{encoder}.pth', encoder)              # torch
    rows = 518 if makeSquareInput else 686
    mlProgram = ct.models.CompiledMLModel(f"./checkpoints/custom_vits_F16_{rows}_{518}.mlmodelc") if useCoreML else None  # coreml

    #------------------ inference loop
    #------------------------------------------------------------------
    numFiles = len(os.listdir(inputPath)) // 2
    totalError = 0.0
    meanErr = 0.0
    sampleWithLowestError = 0
    samplewithHighestError = 0
    minRMSE = float('inf')
    maxRMSE = float('-inf')
    start = max(0, sampleToTest - 3) if checkIfSynced else 0

    for idx in range(start, numFiles):

        if checkIfSynced:
            print("sample to test:", sampleToTest)
            
        print('\n'"========================================")
        print(f'============= sample --> {idx} =============')
        print("========================================", '\n')

        rgbPath = inputPath + f"RGB_{idx:04d}.png"
        raw_image = cv2.imread(rgbPath)
        raw_image = cv2.rotate(raw_image, cv2.ROTATE_90_CLOCKWISE)

        index = sampleToTest if checkIfSynced else idx
        gtPath = inputPath + f"ARKit_DepthValues_{index:04d}.txt" 
        gt = loadMatrixFromFile(gtPath)
        gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)

        if checkIfSynced:
            overlayInputs(raw_image, gt)
            continue

        pred, cropped = handlePredictionSteps(raw_image, gt, makeSquareInput, borderType, useCoreML, mlProgram, torch_model)

        #--------------------- fitting process
        #-----------------------------------------------------------------------

        gt = 1 / gt + 1e-8      # convert depth to disparity (inverse depth)

        if weightedLsq: 
            scale, shift, mask = weightedLeastSquared(pred, gt, guessInitPrms, inlier_bottom, outlier_cap, num_iters, fit_shift, verbose)
        else: 
            scale, shift, mask = estimateParametersRANSAC(pred, gt) 

        pred = scale * pred + shift

        print("Scale:", fp(scale), ", Shift:", fp(shift), '\n')

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
        meanErr = totalError / (idx + 1)
        print("\nmean across all images so far --> RMSE =", fp(meanErr, 6))
        print("\nimage with lowest error:", sampleWithLowestError, "--> RMSE =", fp(minRMSE, 6))
        print("image with highest error:", samplewithHighestError, "--> RMSE =", fp(maxRMSE, 6))

        if showVisuals:
            ssc = 2.5 if ct else 2
            visualRes = cv2.resize(visualRes, None, fx=ssc, fy=ssc, interpolation=cv2.INTER_CUBIC)
            displayImage("visualRes", visualRes)


        





        





