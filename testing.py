import os
import shutil
import re

def extractIndex(path):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    m = re.search(r'_(\d+)$', name)
    if not m:
        raise ValueError(f"Cannot extract index from filename: {path!r}")
    return int(m.group(1))


numFiles = 8
srcPath = "/Users/3dsensing/Desktop/Archive 5/"
dstPath = "/Users/3dsensing/Desktop/projects/custom_depthAnythingV2/data/batch_6/"

for i in range(numFiles):

    depthSrc = srcPath + f"ARKit_DepthValues_{i+1:04d}.txt"
    depthDst = dstPath + f"ARKit_DepthValues_{i:04d}.txt"

    rgbSrc = srcPath + f"RGB_{i+1:04d}.png"
    rgbDst = dstPath + f"RGB_{i:04d}.png"

    if not os.path.exists(depthSrc): 
        raise FileNotFoundError(f"Depth source path not found: {depthSrc}")
    if not os.path.exists(rgbSrc):
        raise FileNotFoundError(f"RGB source path not found: {rgbSrc}")
    if not os.path.exists(dstPath):
        os.makedirs(dstPath)

    shutil.copy(depthSrc, depthDst)
    shutil.copy(rgbSrc, rgbDst)
    



