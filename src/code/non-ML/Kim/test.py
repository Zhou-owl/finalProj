import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
# f = h5py.File('/home/sc/Desktop/yongqi/depth/Depth-Estimation-Light-Field/DisneyDispPy-master/results_full/disparity_map.hdf5','r')
# print(len(f['disparity_map']))
# for idx,i in enumerate(f['disparity_map']):
#     if idx == 50:
#         img = np.array(i)
#         plt.imshow(img, cmap='gray')
#         plt.show()


files = os.listdir("/home/sc/Desktop/yongqi/depth/Depth-Estimation-Light-Field/DisneyDispPy-master/results_all/Plots/Disparity/512x512")

# Filter only image files
image_files = [file for file in files if file.lower().endswith('.png')]

# Read each image
images = []
for file in image_files:
    image_path = os.path.join("/home/sc/Desktop/yongqi/depth/Depth-Estimation-Light-Field/DisneyDispPy-master/results_all/Plots/Disparity/512x512", file)
    image = Image.open(image_path)
    image = np.array(image)
    #print(np.mean(image))
    images.append(image)
res = np.zeros(shape=images[0].shape)
for i in images:
    res+=i/9
print(np.max(res))
print(np.min(res))
res=np.clip(res,0,255)
res/=255
res = 1-res
res=np.clip(res,0,1)
plt.imshow(res[:,:,:-1])
plt.show()

# import os
# import cv2

# # Function to read images from a folder, clip them, and save to another folder
# def clip_images(input_folder, output_folder, clip_size=(100, 100)):
#     # Create the output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Loop through each file in the input folder
#     for filename in os.listdir(input_folder):
#         # Check if the file is an image (you can add more image extensions if needed)
#         if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
#             # Read the image
#             img_path = os.path.join(input_folder, filename)
#             img = cv2.imread(img_path)

#             # Check if the image is not None (i.e., if it was successfully read)
#             if img is not None:
#                 # Clip the image (for example, resize it to the specified clip_size)
#                 clipped_img = img[400:464,400:464]  # You can use other methods to clip images

#                 # Save the clipped image to the output folder
#                 output_path = os.path.join(output_folder, filename)
#                 cv2.imwrite(output_path, clipped_img)

#                 print(f"Processed: {filename}")

# # Example usage:
# input_folder = "./dataset/box"
# output_folder = "./dataset/clipped"
# clip_images(input_folder, output_folder)