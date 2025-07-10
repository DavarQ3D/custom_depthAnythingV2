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


srcPath = "/Users/3dsensing/Desktop/projects/custom_depthAnythingV2/data/iphone_pro_lidar_1"
dstPath = "/Users/3dsensing/Desktop/projects/custom_depthAnythingV2/data/batch_1"

files = os.listdir(srcPath)
ordered_files = sorted(files, key=extractIndex)

for i, filename in enumerate(ordered_files):
    print("filename:", filename)
    src_file = os.path.join(srcPath, filename)
    dst_file = os.path.join(dstPath, f"DepthValues_{i:04d}.txt")

    # print("src_file:", src_file)
    # print("dst_file:", dst_file)
    # print()

    shutil.copy(src_file, dst_file)

    
    

    


