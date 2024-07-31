import math


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


def find_nearest_point(center_points, target_point, decimal_required):
    nearest_point = None
    min_distance = float('inf')
    for point, text in center_points:
        distance = math.sqrt((point[0] - target_point[0]) ** 2 + (point[1] - target_point[1]) ** 2)
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

if __name__ == '__main__':
    result1 = [
        [[[[160.0, 67.0], [193.0, 67.0], [193.0, 94.0], [160.0, 94.0]], ('8', 0.9995718002319336)],
         [[[130.0, 105.0], [222.0, 109.0], [222.0, 134.0], [129.0, 131.0]], ('膜盒压力表', 0.9980553388595581)],
         [[[74.0, 120.0], [112.0, 123.0], [110.0, 153.0], [72.0, 150.0]], ('4', 0.9990125298500061)],
         [[[157.0, 134.0], [193.0, 134.0], [193.0, 154.0], [157.0, 154.0]], ('kPa', 0.8554465174674988)],
         [[[243.0, 128.0], [273.0, 128.0], [273.0, 155.0], [243.0, 155.0]], ('12', 0.9998997449874878)],
         [[[90.0, 209.0], [104.0, 209.0], [104.0, 233.0], [90.0, 233.0]], ('0', 0.9680286645889282)],
         [[[166.0, 201.0], [181.0, 201.0], [181.0, 211.0], [166.0, 211.0]], ('25', 0.8062974214553833)],
         [[[160.0, 223.0], [184.0, 223.0], [184.0, 239.0], [160.0, 239.0]], ('国', 0.9020943641662598)],
         [[[228.0, 219.0], [256.0, 219.0], [256.0, 246.0], [228.0, 246.0]], ('16', 0.9984807968139648)],
         [[[143.0, 234.0], [201.0, 237.0], [200.0, 255.0], [143.0, 252.0]], ('12810126', 0.9981772899627686)],
         [[[103.0, 257.0], [242.0, 262.0], [241.0, 280.0], [102.0, 275.0]],
          ('沪春自动化仪表（上海）有限公司', 0.9146907329559326)],
         [[[123.0, 284.0], [142.0, 284.0], [142.0, 296.0], [123.0, 296.0]], ('NO', 0.7576274871826172)],
         [[[171.0, 278.0], [214.0, 278.0], [214.0, 290.0], [171.0, 290.0]], ('308', 0.5266245603561401)]]
    ]
    center_points, decimal_required = get_pose(result1)
    target_point = (249, 250)
    nearest_point = find_nearest_point(center_points, target_point, decimal_required)
    print("最近的中心点坐标及对应数字:", nearest_point)
