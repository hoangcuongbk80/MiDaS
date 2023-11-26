import os
import open3d as o3d
import cv2
import glob
import numpy as np
from utils import read_pfm

# https://github.com/isl-org/MiDA

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
    depth_np = focal / idepth  # Convert to NumPy array for modification
    depth = o3d.geometry.Image(depth_np)

    # Extract the name of the depth image file
    depth_filename = d_imgs[idx].split("/")[-1]
    # Remove the file extension
    depth_filename = depth_filename.split(".")[0]

    # Extract the name of the color image file
    color_filename = c_imgs[idx].split("/")[-1]
    label_dir = "./seg/" + color_filename
    label_image = cv2.imread(label_dir, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((5,5), np.uint8)  # Adjust the kernel size as needed
    label_image = cv2.dilate(label_image, kernel, iterations=3)
    label_image = cv2.erode(label_image, kernel, iterations=3)
    _, label_binary = cv2.threshold(label_image, 1, 255, cv2.THRESH_BINARY)
    depth_np[label_binary == 0] = 0 # Objects
    #depth_np[label_binary != 0] = 0 # Background

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, o3d.geometry.Image(depth_np))
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    # Create the "pointcloud_objects" directory if it doesn't exist
    output_dir = "pointcloud_objects"
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

