import os
import coremltools as ct
from custom_utils import *

#=============================================================================================================

def center_crop_or_pad(img: np.ndarray, desiredRow: int, desiredCol: int, borderType = cv2.BORDER_CONSTANT) -> np.ndarray:

    h, w = img.shape[:2]

    # centre crop if the dimension is too large 
    if h > desiredRow:
        top = (h - desiredRow) // 2
        img = img[top : top + desiredRow, :, :]
        h = desiredRow
    if w > desiredCol:
        left = (w - desiredCol) // 2
        img = img[:, left : left + desiredCol, :]
        w = desiredCol

    # symmetric padding if the dimension is too small 
    pad_top    = (desiredRow - h) // 2
    pad_bottom = desiredRow - h - pad_top
    pad_left   = (desiredCol - w) // 2
    pad_right  = desiredCol - w - pad_left

    if any(p > 0 for p in (pad_top, pad_bottom, pad_left, pad_right)):

        if borderType == cv2.BORDER_CONSTANT:
            img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=0)
        elif borderType == cv2.BORDER_REFLECT_101:
            img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_REFLECT_101)
        else:
            raise ValueError(f"Unsupported border type: {borderType}")

    return img, top

#=============================================================================================================

if __name__ == '__main__':

    #--------------------- settings
    batch = 1
    checkIfSynced = False
    encoder = "vits"
    smallInference = False
    useCoreML = False
    weightedLsq = True
    seed = 3
    normalizeVisualError = False
    fitShift = True

    #--------------------- load the torch model
    torch_model = loadTorchModel(f'checkpoints/depth_anything_v2_{encoder}.pth', encoder)

    #--------------------- load coreml model
    mlProgram = ct.models.CompiledMLModel(f"./checkpoints/custom_vits_F16_{686}_{518}.mlmodelc")

    #------------------ configs
    inputPath = f"./data/batch_{batch}/"
    outdir = "./data/outputs"
    os.makedirs(outdir, exist_ok=True)
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

        index = 9 if checkIfSynced else idx
        gtPath = inputPath + f"ARKit_DepthValues_{index:04d}.txt" 
        gt = loadMatrixFromFile(gtPath)
        gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)

        if checkIfSynced:
            overlayInputs(raw_image, gt)
            continue

        gt = 1 / gt + 1e-8                                           # convert depth to disparity (inverse depth)
        # gt = normalize(gt)

        if smallInference:
            resized = cv2.resize(raw_image, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
            r = ensure_multiple_of(resized.shape[0], multiple_of=14)
            c = ensure_multiple_of(resized.shape[1], multiple_of=14)
            cropped = resized[0 : r, 0 : c, :]
            gt = gt[0 : r, 0 : c]
            pred = inferFromTorch(torch_model, cropped, min(r, c))   # inferred disparity

        else:
            sc = 518 / min(raw_image.shape[:2])
            resized = cv2.resize(raw_image, (int(raw_image.shape[1] * sc), int(raw_image.shape[0] * sc)), interpolation=cv2.INTER_CUBIC)
            r = ensure_multiple_of(resized.shape[0], multiple_of=14)
            c = ensure_multiple_of(resized.shape[1], multiple_of=14)
            cropped, top = center_crop_or_pad(resized, r, c)
            pred = inferFromCoreml(mlProgram, cropped) if useCoreML else inferFromTorch(torch_model, cropped, min(r, c))
            gtMarg = (top * 2) / (resized.shape[0] / gt.shape[0])
            gtMarg = round(gtMarg / 2)                               # round to the nearest even number
            gt = gt[gtMarg : -gtMarg, :]
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
            cropped = cv2.resize(cropped, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)

        # pred = normalize(pred)
        scale, shift, mask = weightedLeastSquared(pred, gt, inlier_bottom=0.02, outlier_cap=0.1, fit_shift=fitShift) if weightedLsq else estimateParametersRANSAC(pred, gt, seed) 
        pred = scale * pred + shift

        print("\nScale:", fp(scale), ", Shift:", fp(shift), '\n')

        visualRes = analyzeAndPrepVis(cropped, mask, gt, pred, mode="color", normalizeError=normalizeVisualError)
        visualRes = cv2.resize(visualRes, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        displayImage("visualRes", visualRes)


        





        





