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

    encoder = "vits"
    useCoreML = False and ct is not None
    
    weightedLsq = True
    inlier_bottom = 0.02 
    outlier_cap = 0.1 
    num_iters = 10 
    fit_shift = True 
    verbose = False
    
    makeSquareInput = True
    borderType = cv2.BORDER_CONSTANT
    
    normalizeVisualError = False
    showVisuals = False

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
    
    for idx in range(numFiles):

        print('\n'"========================================")
        print(f'============= sample --> {idx} =============')
        print("========================================", '\n')

        rgbPath = inputPath + f"RGB_{idx:04d}.png"
        raw_image = cv2.imread(rgbPath)
        raw_image = cv2.rotate(raw_image, cv2.ROTATE_90_CLOCKWISE)

        index = 4 if checkIfSynced else idx
        gtPath = inputPath + f"ARKit_DepthValues_{index:04d}.txt" 
        gt = loadMatrixFromFile(gtPath)
        gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)

        if checkIfSynced:
            overlayInputs(raw_image, gt)
            continue

        gt = 1 / gt + 1e-8                                           # convert depth to disparity (inverse depth)
        gt = normalize(gt)

        if makeSquareInput:
            sc = 518 / max(raw_image.shape[:2])
            resized = cv2.resize(raw_image, (int(raw_image.shape[1] * sc), int(raw_image.shape[0] * sc)), interpolation=cv2.INTER_CUBIC)
            r = 518
            c = 518
            cropped, _, left = center_crop_or_pad(resized, r, c, borderType)
            pred = inferFromCoreml(mlProgram, cropped) if useCoreML else inferFromTorch(torch_model, cropped, min(r, c))
            pred = pred[:, left: left + resized.shape[1]]
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
            cropped = cropped[:, left: left + resized.shape[1], :]
            cropped = cv2.resize(cropped, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        else:    
            sc = 518 / min(raw_image.shape[:2])
            resized = cv2.resize(raw_image, (int(raw_image.shape[1] * sc), int(raw_image.shape[0] * sc)), interpolation=cv2.INTER_CUBIC)
            r = ensure_multiple_of(resized.shape[0], multiple_of=14)
            c = ensure_multiple_of(resized.shape[1], multiple_of=14)
            cropped, top, _ = center_crop_or_pad(resized, r, c)
            pred = inferFromCoreml(mlProgram, cropped) if useCoreML else inferFromTorch(torch_model, cropped, min(r, c))
            gtMarg = (top * 2) / (resized.shape[0] / gt.shape[0])
            gtMarg = round(gtMarg / 2)                               # round to the nearest even number
            gt = gt[gtMarg : -gtMarg, :]
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
            cropped = cv2.resize(cropped, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)

        pred = normalize(pred)

        if weightedLsq: 
            scale, shift, mask = weightedLeastSquared(pred, gt, inlier_bottom, outlier_cap, num_iters, fit_shift, verbose)
        else: 
            scale, shift, mask = estimateParametersRANSAC(pred, gt) 

        pred = scale * pred + shift

        print("Scale:", fp(scale), ", Shift:", fp(shift), '\n')

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


        





        





