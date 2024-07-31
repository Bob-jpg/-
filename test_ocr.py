# # #########################保存单个图片
# # import cv2
# # import time
# # from onnxocr.onnx_paddleocr import ONNXPaddleOcr,sav2Img
# # import sys
# # import  numpy as np
# # #固定到onnx路径·
# # # sys.path.append('./paddle_to_onnx/onnx')
# # import  os
# #
# #
# #
# #
# # def gaussian_blur(image, ksize=3):
# #     return cv2.GaussianBlur(image, (ksize, ksize), 0)
# #
# # def sharpen(image):
# #     kernel = np.array([[0, -1, 0],
# #                        [-1, 5, -1],
# #                        [0, -1, 0]])
# #     return cv2.filter2D(image, -1, kernel)
# #
# #
# #
# #
# #
# #
# # model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False)
# #
# #
# # img_path=r"D:\fujingbo\11111\OnnxOCR-main\onnxocr\test_images\9.jpg"
# #
# #
# #
# #
# # # 示例
# # img = cv2.imread(img_path)
# #
# # img = gaussian_blur(img)
# # img = sharpen(img )
# #
# # s = time.time()
# # result = model.ocr(img)
# # e = time.time()
# # print("total time: {:.3f}".format(e - s))
# #
# # print("result:", result)
# #
# # for box in result[0]:
# #     print(box)
# #
# # sav2Img(img, result)
#
#
#
# ###################################
# import cv2
# import time
# from onnxocr.onnx_paddleocr import ONNXPaddleOcr,sav2Imglist
# import sys
# import  numpy as np
# #固定到onnx路径·
# # sys.path.append('./paddle_to_onnx/onnx')
# import  os
#
#
#
#
# def gaussian_blur(image, ksize=3):
#     return cv2.GaussianBlur(image, (ksize, ksize), 0)
#
# def sharpen(image):
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5, -1],
#                        [0, -1, 0]])
#     return cv2.filter2D(image, -1, kernel)
#
#
#
#
#
#
# model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=True)
#
#
# img_path=r"D:\fujingbo\11111\yolov8-guanjiandianshujuzhizuo\ceshi"
#
# output_dir=r"D:\fujingbo\11111\OnnxOCR-main\result1"
#
# # 结果列表
# image_results = []
#
# imagename=[]
#
# for image_name in os.listdir(img_path):
#     image_path = os.path.join(img_path, image_name)
#     print("Processing:", image_path)
#
#     img = cv2.imread(image_path)
#
#     img = gaussian_blur(img)
#     img = sharpen(img)
#
#     s = time.time()
#     result = model.ocr(img)
#     e = time.time()
#     print("Total time: {:.3f}".format(e - s))
#
#     print("Result:", result)
#
#
# ######修改
#     for box in result[0]:
#         print(box[-1])
#
#
#
#     # 添加到结果列表
#     image_results.append((img, result))
#     imagename.append(image_name)
# # 保存所有处理后的图像
# sav2Imglist(image_results,imagename, output_dir)
#


# #########################保存单个图片
# import cv2
# import time
# from onnxocr.onnx_paddleocr import ONNXPaddleOcr,sav2Img
# import sys
# import  numpy as np
# #固定到onnx路径·
# # sys.path.append('./paddle_to_onnx/onnx')
# import  os
#
#
#
#
# def gaussian_blur(image, ksize=3):
#     return cv2.GaussianBlur(image, (ksize, ksize), 0)
#
# def sharpen(image):
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5, -1],
#                        [0, -1, 0]])
#     return cv2.filter2D(image, -1, kernel)
#
#
#
#
#
#
# model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=False)
#
#
# img_path=r"D:\fujingbo\11111\OnnxOCR-main\onnxocr\test_images\9.jpg"
#
#
#
#
# # 示例
# img = cv2.imread(img_path)
#
# img = gaussian_blur(img)
# img = sharpen(img )
#
# s = time.time()
# result = model.ocr(img)
# e = time.time()
# print("total time: {:.3f}".format(e - s))
#
# print("result:", result)
#
# for box in result[0]:
#     print(box)
#
# sav2Img(img, result)


###################################
import cv2
import time
from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2Imglist
import sys
import numpy as np
# 固定到onnx路径·
# sys.path.append('./paddle_to_onnx/onnx')
import os

def get_pose(result):
    # 筛选忽略小数点后长度不超过3的数字
    filtered_result = []
    for item in result:
        for sub_item in item:
            coordinates, (text, confidence) = sub_item
            # 忽略小数点后的字符长度检查
            if text.replace('.', '').isdigit() and len(text.replace('.', '')) < 3:
                filtered_result.append(sub_item)

    print("筛选结果:", filtered_result)

    # 计算中心点坐标
    center_points = []

    for item in filtered_result:
        coordinates, (text, confidence) = item
        # 左上角和右下角坐标
        top_left = coordinates[0]
        bottom_right = coordinates[2]
        # 计算中心点
        center_x = (top_left[0] + bottom_right[0]) / 2
        center_y = (top_left[1] + bottom_right[1]) / 2
        center_points.append(((center_x, center_y), text))

    #print("中心点坐标及对应数字:", center_points)

    return center_points

def find_nearest_point(center_points, target_point):
    nearest_point = None
    min_distance = float('inf')
    for point, text in center_points:
        distance = math.sqrt((point[0] - target_point[0])**2 + (point[1] - target_point[1])**2)
        if distance < min_distance:
            min_distance = distance
            nearest_point = (point, text)
    return nearest_point


def gaussian_blur(image, ksize=3):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def sharpen(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=True)

img_path = r"D:\fujingbo\11111\yolov8-guanjiandianshujuzhizuo\ceshi"

output_dir = r"D:\fujingbo\11111\OnnxOCR-main\result1"

# 结果列表
image_results = []

imagename = []

for image_name in os.listdir(img_path):
    image_path = os.path.join(img_path, image_name)
    print("Processing:", image_path)

    img = cv2.imread(image_path)

    img = gaussian_blur(img)
    img = sharpen(img)

    s = time.time()
    result = model.ocr(img)
    e = time.time()
    print("Total time: {:.3f}".format(e - s))

    print("Result:", result)
    # ######修改
    pose = get_pose(result)
    print(pose)
    # result1=[]
    # for box in result[0]:
    #     #print(box[-1])
    #     result1.append(box[-1][0])
    # print(result1)
    # 添加到结果列表
    image_results.append((img, result))
    imagename.append(image_name)
# 保存所有处理后的图像
sav2Imglist(image_results, imagename, output_dir)

