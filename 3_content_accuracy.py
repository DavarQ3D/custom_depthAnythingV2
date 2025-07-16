import os
import coremltools as ct
from custom_utils import *

#=============================================================================================================

if __name__ == '__main__':

    outdir = "./data/outputs"
    os.makedirs(outdir, exist_ok=True)

    #--------------------- settings
    batch = 1
    checkIfSynced = False
    encoder = "vits"
    useCoreML = False
    weightedLsq = True
    fitShift = True
    makeSquareInput = True
    borderType = cv2.BORDER_CONSTANT
    normalizeVisualError = False

    #--------------------- load the torch model
    torch_model = loadTorchModel(f'checkpoints/depth_anything_v2_{encoder}.pth', encoder)

    #--------------------- load coreml model
    mlProgram = ct.models.CompiledMLModel(f"./checkpoints/custom_vits_F16_{686}_{518}.mlmodelc")

    #------------------ configs
    inputPath = f"./data/batch_{batch}/"
    numFiles = len(os.listdir(inputPath)) // 2

    #------------------ inference loop
    #------------------------------------------------------------------
    
    for idx in range(numFiles):

        print('\n'"=========================================================")
        print(f'========= sample --> {idx} =========')
        print("=========================================================", '\n')

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
        scale, shift, mask = weightedLeastSquared(pred, gt, inlier_bottom=0.02, outlier_cap=0.1, fit_shift=fitShift) if weightedLsq else estimateParametersRANSAC(pred, gt) 
        pred = scale * pred + shift

        print("\nScale:", fp(scale), ", Shift:", fp(shift), '\n')

        visualRes = analyzeAndPrepVis(cropped, mask, gt, pred, mode="color", normalizeError=normalizeVisualError)
        visualRes = cv2.resize(visualRes, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        displayImage("visualRes", visualRes)


        





        





