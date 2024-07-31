from ultralytics import YOLO
import cv2
import numpy as np
import os
import pandas as pd
import math
import time
#########
import cv2
import time
from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2Imglist
import sys
import numpy as np



##############ocr

######################假如表盘的刻度有0.或者0开头的数字，则最大刻度中间插入.
def get_pose(result):
    filtered_result = []
    decimal_required = False

    for item in result:
        for sub_item in item:
            coordinates, (text, confidence) = sub_item
            # 筛选忽略小数点后长度不超过3的数字
            if text.replace('.', '').isdigit() and len(text.replace('.', '')) <= 3:
                filtered_result.append(sub_item)
                # 检查文本是否符合"0+数字"或"0.+数字"的模式
                if text.startswith('0') and text!="0":
                    decimal_required = True

    print("筛选结果:", filtered_result)

    center_points = []
    for item in filtered_result:
        coordinates, (text, confidence) = item
        top_left = coordinates[0]
        bottom_right = coordinates[2]
        center_x = (top_left[0] + bottom_right[0]) / 2
        center_y = (top_left[1] + bottom_right[1]) / 2
        center_points.append(((center_x, center_y), text))

    print("中心点坐标及对应数字:", center_points)

    return center_points, decimal_required



####################该函数找出最大刻度点最近的OCR识别的刻度值
def find_nearest_point(center_points, target_point, decimal_required):
    nearest_point = None
    min_distance = float('inf')
    for point, text in center_points:
        distance = math.sqrt((point[0] - target_point[-1][0]) ** 2 + (point[1] - target_point[-1][1]) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_point = (point, text)

    # 如果需要小数点且文本没有小数点，则在中间位置添加小数点
    if nearest_point is not None:
        point, text = nearest_point
        if decimal_required and '.' not in text:
            # 计算中间位置并插入小数点
            insert_pos = len(text) // 2
            text = text[:insert_pos] + '.' + text[insert_pos:]
        nearest_point = (point, text)

    return nearest_point


#############高斯模糊
def gaussian_blur(image, ksize=3):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)



###########锐化
def sharpen(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)



########################计算角度
def calculate_counterclockwise_angle(base, start, target):
    # 计算基准向量和目标向量（在图像坐标系下，Y轴从上到下增大）
    base_start = (start[0] - base[0], start[1] - base[1])
    base_target = (target[0] - base[0], target[1] - base[1])

    # 计算基准向量和目标向量的模（长度）
    base_start_magnitude = math.sqrt(base_start[0] ** 2 + base_start[1] ** 2)
    base_target_magnitude = math.sqrt(base_target[0] ** 2 + base_target[1] ** 2)

    # 计算点积
    dot_product = base_start[0] * base_target[0] + base_start[1] * base_target[1]

    # 计算叉积的 z 分量
    cross_product = base_start[0] * base_target[1] - base_start[1] * base_target[0]

    # 计算夹角的余弦值
    cos_theta = dot_product / (base_start_magnitude * base_target_magnitude)

    # 限制 cos_theta 在 -1 到 1 之间，以防止由于浮点数误差导致的错误
    cos_theta = max(-1, min(1, cos_theta))

    # 使用反余弦函数计算角度 θ（弧度）
    theta = math.acos(cos_theta)

    # 将弧度转换为角度
    theta_degrees = math.degrees(theta)

    # 判断角度的方向（在图像坐标系下，Y轴从上到下增大）
    if cross_product < 0:
        theta_degrees = -theta_degrees

    # 确保角度在0到360度之间
    if theta_degrees < 0:
        theta_degrees += 360

    return theta, theta_degrees




if __name__ == '__main__':
    model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=True)
    # 模型加载
    model1 = YOLO(r"D:\fujingbo\11111\OnnxOCR-main/best.pt")  # 模型1的路径
    folder_path = r"D:\fujingbo\11111\OnnxOCR-main\ceshi"
    output_folder = r"D:\fujingbo\11111\OnnxOCR-main\result/"

    #cropped_folder=r"D:\fujingbo\11111\yolov8-guanjiandianshujuzhizuo\cropped_folder"


    # 数据列表
    data = []
    threshold = 0.2  # 设置置信度阈值

    # 关键点类别标签（假设每个关键点有一个对应的类别标签）
    keypoint_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 确保文件是图片格式的
            # 构造图片的完整路径
            image_path = os.path.join(folder_path, filename)
            # 加载图片
            image = cv2.imread(image_path)
            print(filename)
            #################################
            image= gaussian_blur(image)
            image = sharpen(image)
            ocrresult = model.ocr(image)
            print("ocrResult:", ocrresult)

            pose , decimal_required= get_pose(ocrresult)
            print(pose)

            # 进行模型预测和标注
            results1 = model1.predict(source=image)
            if len(results1[0].boxes.xyxy) == 0:  # 如果没有检测到结果，跳过当前循环
                print(f"未在 {filename} 中检测到任何对象，跳过该图像。")
                continue

            # 初始化图像数据字典
            image_data = {
                '文件名': filename,
                '检测结果': []
            }

            # 获取检测框和置信度
            boxes = results1[0].boxes.xyxy
            scores = results1[0].boxes.conf

            # 转换为NumPy数组
            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()

            # 执行非极大值抑制
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=threshold, nms_threshold=0.5)

            # 如果没有剩余的框，跳过当前图像
            if len(indices) == 0:
                print(f"未在 {filename} 中检测到任何对象，跳过该图像。")
                continue

            # 处理剩余的框
            for i in indices.flatten():
                # 选择颜色（这里选择红色）
                color = (0, 0, 255)

                # 获取框的坐标
                box = boxes[i]
                x1, y1, x2, y2 = map(int, box)

                # # 在图像上绘制矩形框
                # cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                # # 裁剪矩形框
                # cropped_image = image[y1:y2, x1:x2]
                # cropped_path = os.path.join(cropped_folder, f"cropped_{filename}_{i}.png")
                # cv2.imwrite(cropped_path, cropped_image)

                # 关键点的显示
                point1 = results1[0].keypoints.xy[i]
                point1_scores = results1[0].keypoints.conf[i]  # 获取关键点的置信度

                # 提取数值
                point1_numpy = point1.cpu().numpy()

                # 存储每个框的检测结果
                detection_info = {
                    '关键点': []
                }

                for j in range(len(point1_numpy)):
                    if point1_scores[j] > threshold:
                        point_j = point1_numpy[j].astype(int)

                        # # 画关键点
                        # cv2.circle(image, tuple(point_j), 5, color, -1)
                        # # 标注关键点类别
                        # if j < len(keypoint_labels):
                        #     keypoint_label = keypoint_labels[j]
                        #     text_size = cv2.getTextSize(keypoint_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        #     text_x = point_j[0] - text_size[0] // 2
                        #     text_y = point_j[1] - 25
                        #     cv2.putText(image, keypoint_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                        coord_text = f"({point_j[0]}, {point_j[1]})"
                        coord_size = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        coord_x = point_j[0] - coord_size[0] // 2
                        coord_y = point_j[1] + 15 + coord_size[1]  # 在关键点下方显示坐标
                        cv2.putText(image, coord_text, (coord_x, coord_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                        # 将关键点和类别信息添加到检测结果中
                        detection_info['关键点'].append({
                            '坐标': (point_j[0], point_j[1]),
                            '类别': keypoint_labels[j]
                        })

                # 将检测结果添加到图像数据中
                image_data['检测结果'].append(detection_info)

            # 提取并排序关键点信息
            all_keypoints = [kp for detection in image_data['检测结果'] for kp in detection['关键点']]
            sorted_keypoints = sorted(all_keypoints, key=lambda x: int(x['类别']))

            # 只保留坐标并保存为数组
            coordinates_array = [kp['坐标'] for kp in sorted_keypoints]
            print(coordinates_array)
            nearest_point=find_nearest_point(pose,coordinates_array, decimal_required)
            print("最近的中心点坐标及对应数字:", nearest_point)
            pose_number=float(nearest_point[-1])
            print("------",pose_number)



            all_angel=[]

            # 确保至少有三个关键点
            if len(coordinates_array) >= 3:
                base_point = coordinates_array[0]
                start_point = coordinates_array[2]

                for i in range(1, len(coordinates_array)):
                    target_point = coordinates_array[i]

                    # 计算夹角
                    _, angle_degrees = calculate_counterclockwise_angle(base_point, start_point, target_point)
                    angle_text = f"{angle_degrees:.2f}"
                    print("angle",angle_text)
                    all_angel.append(float(angle_text))
                    # 在图像上绘制夹角
                    text_x = (target_point[0] + target_point[0]) // 2
                    text_y = (target_point[1] + target_point[1]) // 2
                    cv2.putText(image, angle_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    # 画线条
                    cv2.line(image, base_point, start_point, (0, 255, 0), 1)
                    cv2.line(image, base_point, target_point, (0, 255, 0), 1)
            print("----------111",len(coordinates_array))
            # 保存带有标记的图片
            if all_angel[0] > all_angel[2]:
                kedu_result = (pose_number / all_angel[-1]) * (all_angel[0] - all_angel[2]) + pose_number / (
                            len(coordinates_array) -3)
            else:
                kedu_result = all_angel[0] * ((pose_number / (len(coordinates_array) - 3)) / all_angel[2])

            # 将kedu_result画在target_point[1]的下面
            result_text = f"kedu_result: {kedu_result:.2f}"
            text_x = coordinates_array[0][0]  # 使用最后一个target_point的x坐标
            text_y = coordinates_array[0][1]  # 在最后一个target_point的y坐标下方30像素
            cv2.putText(image, result_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            print("1111111", kedu_result)


            print("1111111",kedu_result)
            # cv2.imshow("result",image)
            # cv2.waitKey()
            output_path = os.path.join(output_folder, f"output_{filename}")
            cv2.imwrite(output_path, image)

    print(f"结果已保存到 {output_folder}")
