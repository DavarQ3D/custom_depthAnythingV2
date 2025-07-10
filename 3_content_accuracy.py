import cv2
import numpy as np
import torch
from PIL import Image
from depth_anything_v2.dpt import DepthAnythingV2
import os
from depth_anything_v2.util import transform
from enum import Enum
from pathlib import Path
from sklearn.linear_model import RANSACRegressor
import coremltools as ct

#=============================================================================================================

def loadTorchModel(modelPath, encoder):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }    
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(modelPath, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    return depth_anything    

#=============================================================================================================

def center_crop_or_pad(img: np.ndarray, desiredRow: int, desiredCol: int) -> np.ndarray:

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
        img = cv2.copyMakeBorder(
            img,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT_101,
            # borderType=cv2.BORDER_CONSTANT,
            # value = 0
        )

    return img, top

#=============================================================================================================

def inferFromTorch(model, image, input_size):
    return model.infer_image(image, input_size, doResize=False)

#=============================================================================================================

def inferFromCoreml(mlProg, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_input = Image.fromarray(rgb)
    pred = mlProg.predict({"image": pil_input})
    depth = np.array(pred["depth"], dtype=np.float32)
    return depth

#=============================================================================================================

def fp(value, precision=4):
    return f"{value:.{precision}f}" 

def normalize(image):
    return (image - image.min()) / (image.max() - image.min() + 1e-8)

def denormalize(image):
    return (image * 255).astype(np.uint8)

def analyzeAndPrepVis(rgb, mask, ref, pred, mode = "color", normalizeError=True):

    assert mode in ("color", "grayscale")
    
    err = np.abs(ref - pred)
    valid = err[mask]
    print("err ---> min:", fp(valid.min()), ", max:", fp(valid.max()), "--> RMSE:", fp(np.sqrt((valid**2).mean()), 6))

    ref = normalize(ref)
    pred = normalize(pred)

    if normalizeError:
        err = normalize(err)

    ref = denormalize(ref)
    pred = denormalize(pred)    
    err = denormalize(err)

    if mode == "grayscale":
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        return cv2.hconcat([gray, ref, pred, err])

    mask = mask.astype(np.uint8) * 255
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
    pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    err = cv2.cvtColor(err, cv2.COLOR_GRAY2BGR)
    err = cv2.applyColorMap(err, cv2.COLORMAP_JET)

    return cv2.hconcat([rgb, mask, ref, pred, err])

#=============================================================================================================

def displayImage(title, image):
    cv2.imshow(title, image)
    key = cv2.waitKey(0)
    if key == 27:  
        cv2.destroyAllWindows()
        exit()

#=============================================================================================================

def customResize(image, lower_dim, resizeMode = "lower_bound"):

    assert resizeMode in ("lower_bound", "upper_bound")

    resizer = transform.Resize(
        width=lower_dim,                      
        height=lower_dim,                     
        resize_target=False,                  
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method=resizeMode,      
        image_interpolation_method=cv2.INTER_CUBIC,
    )

    sample = {"image": image}
    sample = resizer(sample)               
    return sample["image"]   

#=============================================================================================================

def loadMatrixFromFile(path):
    path = Path(path)
    matrix = np.loadtxt(path, delimiter=',', dtype=np.float64)
    return matrix

#=============================================================================================================

def ensure_multiple_of(x, multiple_of=14):
    return (np.floor(x / multiple_of) * multiple_of).astype(int)

#=============================================================================================================

def checkIfSynced(rgb, depth):
    rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gray = normalize(gray)
    depth = normalize(depth)
    diff = np.abs(gray - depth)
    gray = cv2.resize(gray, (gray.shape[1] * 4, gray.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
    depth = cv2.resize(depth, (depth.shape[1] * 4, depth.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
    diff = cv2.resize(diff, (diff.shape[1] * 4, diff.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("gray", gray)
    # cv2.imshow("depth", depth)
    cv2.imshow("diff", diff)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
        exit()

#=============================================================================================================

class FittingMode(Enum):
    SolveForScaleOnly = 0
    SolveForScaleAndShift = 1
    MedianScaleOnly = 2

def estimateParameters(pred, gt, mode, mask=None):

    if mask is None:
        mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred)

    if not mask.any():
        raise ValueError("No valid pixels in mask for scale/shift fitting")

    if mode == FittingMode.SolveForScaleOnly:                                      # Least-squares: [pred] ⋅ [s] ≈ gt
        A = pred[mask].ravel()[:, np.newaxis]
        scale = np.linalg.lstsq(A, gt[mask].ravel(), rcond=None)[0][0]
        shift = 0.0

    elif mode == FittingMode.SolveForScaleAndShift:                                # Least-squares: [pred, 1] ⋅ [s, t]ᵀ ≈ gt
        A = np.vstack([pred[mask].ravel(), np.ones(mask.sum())]).T
        scale, shift = np.linalg.lstsq(A, gt[mask].ravel(), rcond=None)[0]

    elif mode == FittingMode.MedianScaleOnly:                                      # Median-based scale only (Monodepth-style)                                                    # Median-based scale only (Monodepth-style)
        eps = 1e-8  
        scale = (np.median(gt[mask]) / (np.median(pred[mask]) + eps))
        shift = 0.0

    else:
        raise ValueError(f"Unknown fitting mode: {mode}")

    return scale, shift, mask

#=============================================================================================================

def estimateParametersRANSAC(pred, gt, mask=None):

    if mask is None:
        mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred)

    if not mask.any():
        raise ValueError("No valid pixels in mask for scale/shift fitting")

    x = pred[mask].ravel().reshape(-1, 1)
    y = gt[mask].ravel()

    ransac = RANSACRegressor()
    ransac.fit(x, y)

    scale = float(ransac.estimator_.coef_[0])
    shift = float(ransac.estimator_.intercept_)

    inliers = ransac.inlier_mask_
    m = np.zeros_like(mask)
    m[mask] = inliers

    return scale, shift, m

#=============================================================================================================

if __name__ == '__main__':

    #--------------------- load the torch model
    encoder = "vits"
    torch_model = loadTorchModel(f'checkpoints/depth_anything_v2_{encoder}.pth', encoder)

    #--------------------- load coreml model
    mlProgram = ct.models.CompiledMLModel(f"./checkpoints/custom_vits_F16_{686}_{518}.mlmodelc")

    #------------------ configs
    img_path = "./data/iphone_images/"
    lidar_path = "./data/iphone_pro_lidar/"
    outdir   = "./data/outputs"
    os.makedirs(outdir, exist_ok=True)
    numFiles = len(os.listdir(img_path)) 

    #------------------ inference loop
    #------------------------------------------------------------------

    smallInference = False
    useCoreML = False
    
    for idx in range(numFiles):

        print('\n'"=========================================================")
        print(f'========= sample --> {idx} =========')
        print("=========================================================", '\n')

        rgbPath = img_path + f"RGB_{idx+2:04d}.JPG"
        raw_image = cv2.imread(rgbPath)
        raw_image = cv2.rotate(raw_image, cv2.ROTATE_90_CLOCKWISE)

        gtPath = lidar_path + f"DepthValues_{idx+1:04d}.txt"
        gt = loadMatrixFromFile(gtPath)
        gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)

        # checkIfSynced(raw_image, gt)

        gt = 1 / gt + 1e-8                                           # convert depth to disparity (inverse depth)
        gt = normalize(gt)

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

        pred = normalize(pred)
        scale, shift, mask = estimateParametersRANSAC(pred, gt) 
        # scale, shift, mask = estimateParameters(pred, gt, mode=FittingMode.SolveForScaleAndShift)    
        pred = scale * pred + shift

        print("Scale:", fp(scale), ", Shift:", fp(shift), '\n')

        visualRes = analyzeAndPrepVis(cropped, mask, gt, pred, mode="color", normalizeError=False)
        visualRes = cv2.resize(visualRes, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        displayImage("visualRes", visualRes)


        





        





