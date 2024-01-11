import os
import open3d as o3d
import glob
import numpy as np
from utils import read_pfm


#https://github.com/isl-org/MiDaS/issues/4

intrinsic = o3d.io.read_pinhole_camera_intrinsic("intrinsic.json")

c_imgs = glob.glob("./input/*.png")
c_imgs.sort()
d_imgs = glob.glob("./output/*.pfm")
d_imgs.sort()

for idx in range(len(c_imgs)):
    color = o3d.io.read_image(c_imgs[idx])
    idepth = read_pfm(d_imgs[idx])[0]

    idepth = idepth - np.amin(idepth)
    idepth /= np.amax(idepth)
    focal = intrinsic.intrinsic_matrix[0, 0]
    depth = focal / (idepth)
    depth = o3d.geometry.Image(depth)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    # Extract the name of the depth image file
    depth_filename = d_imgs[idx].split("/")[-1]
    # Remove the file extension
    depth_filename = depth_filename.split(".")[0]

    # Create the "poincloud" directory if it doesn't exist
    output_dir = "pointcloud"
    os.makedirs(output_dir, exist_ok=True)

    pcd_dir = os.path.join(output_dir, depth_filename + ".ply")
    o3d.io.write_point_cloud(pcd_dir, pcd)

#---------------------------------------------------------------------------

# intrinsic = o3d.io.read_pinhole_camera_intrinsic("intrinsic.json")

# c_imgs = glob.glob("./input/*.png")
# c_imgs.sort()
# d_imgs = glob.glob("./output/*.png")
# d_imgs.sort()
# for idx in range(len(c_imgs)):
#     color = o3d.io.read_image(c_imgs[idx])
#     depth = o3d.io.read_image(d_imgs[idx])
#     rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)
#     pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
#     pcd_dir = "pc" + str(idx) + ".ply"
#     o3d.io.write_point_cloud(pcd_dir, pcd)

