import numpy as np
import cv2
import os

import torch
import torchvision.transforms as T
from PIL import Image
 
def adjust_brightness(image, value):
    """이미지 밝기 조정"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v + value, 0, 255).astype(np.uint8)
    adjusted_hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)
 
def adjust_gamma(image, gamma=1.0):
    """감마 조정"""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)
 
def add_padding(image, padding, color=(0, 0, 0)):
    """이미지 패딩 추가"""
    return cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=color)
 
def rotate_image(image, angle):
    """이미지 회전"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))
 
def translate_image(image, x, y):
    """이미지 수직, 수평 이동"""
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
 
def crop_image(image, x, y, width, height):
    """이미지 크롭 (부분 잘라내기)"""
    return image[y:y+height, x:x+width]
 
def flip_image(image, mode):
    """이미지 좌우/상하 반전 (mode: 1=좌우, 0=상하, -1=상하좌우)"""
    return cv2.flip(image, mode)

def remove_background(image):
    """DeepLabV3를 이용해 신체 부위 분할 후 배경 제거 및 크롭 (투명 배경 적용)"""
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()

    image_pil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환
    image_pil = Image.fromarray(image_pil)

    preprocess = T.Compose([
        T.Resize((520, 520)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image_pil).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)["out"][0]

    output_predictions = output.argmax(0).byte().cpu().numpy()
    mask = (output_predictions == 15).astype(np.uint8) * 255  # 신체 부분을 255로 설정

    # **마스크 크기 조정 (원본 이미지 크기에 맞춤)**
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # **바운딩 박스로 신체 영역 찾기**
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None  # 신체 감지 안되면 None 반환

    x, y, w, h = cv2.boundingRect(coords)  # 바운딩 박스

    # 🟢 **RGBA 변환**
    b, g, r = cv2.split(image)
    alpha = mask  # alpha 채널로 사용할 마스크

    # **여기서 크기 오류 방지**
    if alpha.shape != b.shape:
        alpha = cv2.resize(alpha, (b.shape[1], b.shape[0]))

    rgba = cv2.merge([b, g, r, alpha])  # 크기 맞춘 후 merge

    # **신체 부분만 크롭**
    cropped = rgba[y:y+h, x:x+w]

    return cropped


# 사용 예시
# image = cv2.imread("example.jpg")
# bright_image = adjust_brightness(image, 50)
# gamma_image = adjust_gamma(image, 1.5)
# padded_image = add_padding(image, 20, (255, 255, 255))
# rotated_image = rotate_image(image, 45)
# translated_image = translate_image(image, 30, -20)
# cropped_image = crop_image(image, 50, 50, 100, 100)
# flipped_image = flip_image(image, 1)
 
# 실제 사용
if __name__ == "__main__":
    input_dir = "C:/Users/EL0018/Downloads/test" # 사진 원본 폴더
    output_dir = "C:/Users/EL0018/Downloads/output" # 증강한 사진(변환한 사진) 저장할 폴더
    images = os.listdir(input_dir)
    print(f"원본 사진 개수 : {len(images)}")
    i = 0
    for image_name in images:
        img = cv2.imread(os.path.join(input_dir, image_name))
        ###################### 사용할 함수 입력 ##############################
        # img = rotate_image(img, 90)
        res_img = remove_background(img)
        #####################################################################
        output_file_name = os.path.join(output_dir, f'{image_name}_{str(i)}.png')
        if res_img is not None:
            cv2.imwrite(output_file_name, res_img)
        else:
            cv2.imwrite(output_file_name, img)
        i = i + 1
        # if i > 10000:
        #     break