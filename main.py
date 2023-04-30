import cv2
import numpy as np
import time
import os



# OpenCV
OpenCV_path = "./OpenCV"
a_path = "image_a.bmp"
b_paths = ["image_b1.bmp", "image_b2.bmp", "image_b3.bmp"]
def euclidean_distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
def is_far_enough(candidate, best_match_locs, min_distance):
    for loc in best_match_locs:
        if euclidean_distance(candidate, loc) < min_distance:
            return False
    return True


def search_images( b_paths ,a_path="now",print=False, pyramid=False, srt=0.85, max_matches_per_image=5, min_distance=10):
    """
    :param a_path: 原大圖 "image_a.bmp"
    :param b_paths: 要搜尋的目標"image_b1.bmp", "image_b2.bmp", "image_b3.bmp"
    :param print: 是否顯示搜索結果圖
    :param pyramid: 金字塔搜索 參數微調在scales = np.arange(0.75, 1.5, 0.25)
    :param srt: 相識度閾值
    :param max_matches_per_image: 最大搜尋數量
    :param min_distance: 判定兩圖片的最小間距(避免同一圖片被重複辨識)
    :return:
    """
    if  a_path =="now":
        a_path = "./OpenCV\image_a.bmp"
    elif a_path =="test":
        a_path = "./OpenCV\image_a.bmp"
    if  b_paths =="ALL":
        image_folder = "./OpenCV"
        image_extensions = [".bmp", ".jpg", ".jpeg", ".png", ".gif"]
        b_paths = []
        for file in os.listdir(image_folder):
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in image_extensions:
                file_path = os.path.join(image_folder, file)
                b_paths.append(file_path)
    else:
        b_paths = [os.path.join(OpenCV_path, b_path) for b_path in b_paths]

    img_a = cv2.imread(a_path)
    best_match_locs_list, img_bs = [], []
    start_time = time.time()
    for b_path in b_paths:
        img_b = cv2.imread(b_path)
        img_bs.append(img_b)

        if pyramid:
            pyramid_onoff = "True"
            scales = np.arange(0.75, 1.5, 0.25)
        else:
            pyramid_onoff = "False"
            scales = np.array([1.0])

        best_match_locs = []
        for scale in scales:
            resized_a = cv2.resize(img_a, (int(img_a.shape[1] * scale), int(img_a.shape[0] * scale)))
            match_result = cv2.matchTemplate(resized_a, img_b, cv2.TM_CCOEFF_NORMED)
            for _ in range(max_matches_per_image):
                _, max_val, _, max_loc = cv2.minMaxLoc(match_result)

                candidate_loc = (int(max_loc[0] / scale), int(max_loc[1] / scale))
                if max_val >= srt and is_far_enough(candidate_loc, best_match_locs, min_distance):
                    best_match_locs.append(candidate_loc)
                    match_result[max_loc[1], max_loc[0]] = -1
                else:
                    break
        best_match_locs_list.append(best_match_locs)
    end_time = time.time()
    time_elapsed = end_time - start_time
    ## 繪圖
    if print:
        marked_img_a = img_a.copy()
        for idx, img_b in enumerate(img_bs):
            marked_img_a = draw_rectangle_on_matches(marked_img_a, img_b, best_match_locs_list[idx], b_paths[idx])
        cv2.imshow(f"OpenCV _ LP={pyramid_onoff} _ SRT={srt} _ Time={time_elapsed:.2f}s", marked_img_a)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return best_match_locs_list, img_a, img_bs

def draw_rectangle_on_matches(img_a, img_b, best_match_locs, b_path):
    if not best_match_locs:
        #  print(f"Image B ({b_path}) not found in image A with the given similarity threshold.")
        return img_a

    for best_match_loc in best_match_locs:
        width, height = img_b.shape[1], img_b.shape[0]  #
        top_left, bottom_right = best_match_loc, (best_match_loc[0] + width, best_match_loc[1] + height)
        cv2.rectangle(img_a, top_left, bottom_right, (0, 235, 0), 2)
    for best_match_loc in best_match_locs:
        base_filename = os.path.splitext(os.path.basename(b_path))[0]  # 去除文件擴展名，只保留基本文件名
        text = f"{base_filename} {best_match_loc}"
        font, font_scale, font_thickness, font_color = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, (0, 0, 0)
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x, text_y = best_match_loc[0]-1, best_match_loc[1] +7
        cv2.rectangle(img_a, (text_x, text_y - 25), (text_x + text_size[0]-2, text_y - 8), (0, 235, 0), -1)
        cv2.putText(img_a, text, (text_x, text_y - text_size[1]), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    return img_a

def cv_showpos(target):
    b_paths = [target]
    a_path
    best_match_locs_list, img_a, img_bs = search_images(b_paths, print=False, pyramid=False, srt=0.8)
    if target not in b_paths:
        print("Image path not found in b_paths.")
        return
    idx = b_paths.index(target)
    coordinates_str = "|".join([f"{loc[0]}, {loc[1]}" for loc in best_match_locs_list[idx]])
    return coordinates_str


# 大漠棒定







search_images(b_paths, a_path="test", print=True, pyramid=True, srt=0.8)
print("Searching images2")