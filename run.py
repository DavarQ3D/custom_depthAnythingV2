import cv2
import glob
import numpy as np
import os
import torch
from depth_anything_v2.dpt import DepthAnythingV2

if __name__ == '__main__':

    img_path    = "./data/camera"
    outdir      = "data/outputs"
    input_size  = 518
    encoder     = "vits"

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    filenames = glob.glob(os.path.join(img_path, '**/*'), recursive=True)
    os.makedirs(outdir, exist_ok=True)
    
    for k, filename in enumerate(filenames):

        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        depth = depth_anything.infer_image(raw_image, input_size)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)

        split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
        combined_result = cv2.hconcat([raw_image, split_region, depth])
        cv2.imwrite(os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)