import cv2
import numpy as np
import torch
from PIL import Image
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util import transform
from pathlib import Path
from sklearn.linear_model import RANSACRegressor

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

def analyzeAndPrepVis(rgb, mask, ref, pred, mode = "color", normalizeError=True, vertConcat=False):

    assert mode in ("color", "grayscale")
    
    err = np.abs(ref - pred)
    valid = err[mask]
    rmse = np.sqrt((valid**2).mean())
    print("valid pixels RMSE =", fp(rmse, 6))

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

    visualRes = cv2.vconcat([rgb, mask, ref, pred, err]) if vertConcat else cv2.hconcat([rgb, mask, ref, pred, err])

    return visualRes, rmse

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

def overlayInputs(rgb, depth):
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

def estimateParametersRANSAC(pred, gt, seed = 3, mask=None):

    if mask is None:
        mask = (gt > 0) & (pred > 0) & np.isfinite(gt) & np.isfinite(pred)

    if not mask.any():
        raise ValueError("No valid pixels in mask for scale/shift fitting")

    x = pred[mask].ravel().reshape(-1, 1)
    y = gt[mask].ravel()

    ransac = RANSACRegressor(random_state=seed)
    ransac.fit(x, y)

    scale = float(ransac.estimator_.coef_[0])
    shift = float(ransac.estimator_.intercept_)

    inliers = ransac.inlier_mask_
    m = np.zeros_like(mask)
    m[mask] = inliers

    return scale, shift, m

#=============================================================================================================

def center_crop_or_pad(img: np.ndarray, desiredRow: int, desiredCol: int, borderType = cv2.BORDER_CONSTANT) -> np.ndarray:

    h, w = img.shape[:2]

    # centre crop if the dimension is too large 
    top = 0
    left = 0
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

    return img, pad_top, pad_left, top, left

#=============================================================================================================

def robustNormalize(x):
    t = np.median(x)
    s = np.mean(np.abs(x - t))
    if s == 0:
        raise ValueError("Input array has zero variance; cannot normalize.")
    normalized = (x - t) / s
    return normalized

#=============================================================================================================

def estimateInitialParams(x, y, fitShift):

    t_pred = np.median(x)
    s_pred = np.mean(np.abs(x - t_pred)) if len(x) > 0 else 1.0
    
    t_gt = np.median(y)     
    s_gt = np.mean(np.abs(y - t_gt)) if len(y) > 0 else 1.0  

    scale = s_gt / s_pred 
    shift = t_gt - scale * t_pred if fitShift else 0.0
    
    return scale, shift

#=============================================================================================================

def weightedLeastSquared(pred, gt, guessInitPrms, k_lo=0.5, k_hi=3, num_iters=5, fit_shift=True, verbose=True, mask=None):

    if mask is None:
        mask = (gt > 0) & (pred > 0) & np.isfinite(gt) & np.isfinite(pred)

    if not mask.any():
        raise ValueError("No valid pixels")

    x = pred[mask].ravel()  
    y = gt[mask].ravel()    

    scale, shift = estimateInitialParams(x, y, fit_shift) if guessInitPrms else (1.0, 0.0)

    for iter in range(num_iters):

        fit = scale * x + shift
        residuals = y - fit
        abs_res = np.abs(residuals)

        sigma = 1.4826 * np.median(abs_res) + 1e-12
        outlier_cap = k_hi * sigma
        inlier_bottom = k_lo * sigma

        if verbose:
            print("inlier_bottom =", fp(inlier_bottom, 2), ", outlier_cap =", fp(outlier_cap, 2))
            print("iter =", iter, "-> scale =", fp(scale), ", shift =", fp(shift), ", all pixels err -> max:", fp(abs_res.max()), ", mean:", fp(abs_res.mean()), '\n')

        valid = abs_res <= outlier_cap    # hard skip outliers mask
        if not valid.any():
            break  

        abs_res = abs_res[valid]
        x_valid = x[valid]
        residuals = residuals[valid]

        weights = np.ones_like(abs_res)
        moderate = (abs_res > inlier_bottom) & (abs_res <= outlier_cap)        # moderate residuals mask in which weights aren't 1
        weights[moderate] = 1 - (abs_res[moderate] - inlier_bottom) / (outlier_cap - inlier_bottom)

        wx = weights * x_valid
        wr = weights * residuals

        if fit_shift:
            jtj = np.zeros((2, 2))
            jty = np.zeros(2)
            jtj[0, 0] = np.dot(wx, x_valid)
            jtj[0, 1] = jtj[1, 0] = np.sum(wx)
            jtj[1, 1] = np.sum(weights)
            jty[0] = np.dot(wr, x_valid)
            jty[1] = np.sum(wr)
            update = np.linalg.solve(jtj, jty)
            scale += update[0]
            shift += update[1]
        else:
            jtj_scalar = np.dot(wx, x_valid)
            jty_scalar = np.dot(wr, x_valid)
            update = 0.0 if jtj_scalar == 0 else jty_scalar / jtj_scalar
            scale += update

    final_fit = scale * pred + shift
    final_res = np.abs(gt - final_fit)
    inlier_mask = (final_res <= outlier_cap) & mask

    if verbose:
        print()

    return scale, shift, inlier_mask

#=============================================================================================================

def getValidDataMask(image):
    return (image > 0) & np.isfinite(image)

#=============================================================================================================

def handlePredictionSteps(raw_image, gt, makeSquareInput, borderType, useCoreML, mlProgram, torch_model):

    if makeSquareInput:
        
        sc = 518 / max(raw_image.shape[:2])
        resized = cv2.resize(raw_image, (int(raw_image.shape[1] * sc), int(raw_image.shape[0] * sc)), interpolation=cv2.INTER_CUBIC)

        r = 518
        c = 518
        cropped, pad_top, pad_left, _, _ = center_crop_or_pad(resized, r, c, borderType)

        pred = inferFromCoreml(mlProgram, cropped) if useCoreML else inferFromTorch(torch_model, cropped, min(r, c))

        pred    = pred   [pad_top : pad_top + resized.shape[0], pad_left: pad_left + resized.shape[1]]
        cropped = cropped[pad_top : pad_top + resized.shape[0], pad_left: pad_left + resized.shape[1], :]
        
        pred    = cv2.resize(pred,    (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        cropped = cv2.resize(cropped, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)

    else:    
        sc = 518 / min(raw_image.shape[:2])
        resized = cv2.resize(raw_image, (int(raw_image.shape[1] * sc), int(raw_image.shape[0] * sc)), interpolation=cv2.INTER_CUBIC)

        r = ensure_multiple_of(resized.shape[0], multiple_of=14)
        c = ensure_multiple_of(resized.shape[1], multiple_of=14)
        cropped, _, _, top, left = center_crop_or_pad(resized, r, c)
        
        pred = inferFromCoreml(mlProgram, cropped) if useCoreML else inferFromTorch(torch_model, cropped, min(r, c))
        
        vertMarg  = (top * 2) / (resized.shape[0] / gt.shape[0])
        horizMarg = (left * 2) / (resized.shape[1] / gt.shape[1])
        vertMarg = round(vertMarg / 2)                                # round to the nearest even number
        horizMarg = round(horizMarg / 2)
        gt = gt[vertMarg or None : (-vertMarg) or None, horizMarg or None : (-horizMarg) or None]
        
        pred    = cv2.resize(pred,    (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        cropped = cv2.resize(cropped, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)

    pred = np.maximum(pred, 1e-2)     # ensure no negative values

    return pred, gt, cropped
