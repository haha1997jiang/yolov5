import ctypes
import math
import os

import cv2
import csv
import time
import numpy as np
import pyautogui
import dxcam
import torch
from pynput.mouse import Listener
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from utils.augmentations import (
    Albumentations,
    augment_hsv,
    classify_albumentations,
    classify_transforms,
    copy_paste,
    letterbox,
    mixup,
    random_perspective,
)
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)


def video_record():
    target_fps = 60
    camera = dxcam.create(output_idx=0, output_color="BGR")
    camera.start(target_fps=target_fps, video_mode=True)
    writer = cv2.VideoWriter(
        "video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (2560, 1440)
    )
    time.sleep(3)
    print(f"录屏开始：{time.time()}")
    # 创建CSV文件并写入表头
    with open('mouse_positions.csv', mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Frame', 'Timestamp', 'X', 'Y'])
        for i in range(600):
            writer.write(camera.get_latest_frame())
            mouse_x, mouse_y = pyautogui.position()
            # 记录时间戳
            timestamp = time.time()

            # 将鼠标位置写入CSV文件
            csv_writer.writerow([i + 1, timestamp, mouse_x, mouse_y])

    print(f"录屏结束：{time.time()}")

    camera.stop()
    writer.release()


def on_click(x, y, button, pressed):
    global mouse_down
    if button.name == 'left':
        if pressed:
            mouse_down = True
        else:
            mouse_down = False


def get_key_lock_state(key: int):
    r"""获取指定按键锁定状态

    :param key: 按键码
    :return:
    """

    key_lock_state = ctypes.windll.user32.GetKeyState(key)
    return key_lock_state & 0x0001 != 0

def get_center(xyxy):
    x1, y1, x2, y2 = xyxy
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)


def calculate_relative_position_int(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    relative_x = int(x2 - x1)
    relative_y = int(y2 - y1)
    return (relative_x, relative_y)


def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def screen_record(weights="yolov5s.pt",  # model path or triton URL
                  source="data/images",  # file/dir/URL/glob/screen/0(webcam)
                  data="data/coco128.yaml",  # dataset.yaml path
                  imgsz=(640, 640),  # inference size (height, width)
                  conf_thres=0.25,  # confidence threshold
                  iou_thres=0.45,  # NMS IOU threshold
                  max_det=1000,  # maximum detections per image
                  device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                  view_img=False,  # show results
                  save_txt=False,  # save results to *.txt
                  save_format=0,
                  # save boxes coordinates in YOLO format or Pascal-VOC format (0 for YOLO and 1 for Pascal-VOC)
                  save_csv=False,  # save results in CSV format
                  save_conf=False,  # save confidences in --save-txt labels
                  save_crop=False,  # save cropped prediction boxes
                  nosave=False,  # do not save images/videos
                  classes=None,  # filter by class: --class 0, or --class 0 2 3
                  agnostic_nms=False,  # class-agnostic NMS
                  augment=False,  # augmented inference
                  visualize=False,  # visualize features
                  update=False,  # update all models
                  project="runs/detect",  # save results to project/name
                  name="exp",  # save results to project/name
                  exist_ok=False,  # existing project/name ok, do not increment
                  line_thickness=3,  # bounding box thickness (pixels)
                  hide_labels=False,  # hide labels
                  hide_conf=False,  # hide confidences
                  half=False,  # use FP16 half-precision inference
                  dnn=False,  # use OpenCV DNN for ONNX inference
                  vid_stride=1,  # video frame-rate stride
                  ):
    # 创建摄像头对象
    camera = dxcam.create(output_idx=0, output_color="BGR")
    # 获取屏幕分辨率
    left, top = (2560 - 320) // 2, (1440 - 320) // 2
    right, bottom = left + 320, top + 320
    # 设置捕获区域
    region = (left, top, right, bottom)
    target_fps = 10
    # 启动捕获
    camera.start(region=region, target_fps=target_fps)
    # 创建窗口并设置为可调整大小
    cv2.namedWindow('Screen Recording', cv2.WINDOW_NORMAL)

    # 将窗口设置为置顶模式
    cv2.setWindowProperty('Screen Recording', cv2.WND_PROP_TOPMOST, 1)

    # 取消置顶模式
    # cv2.setWindowProperty('Screen Recording', cv2.WND_PROP_TOPMOST, 0)
    # 创建一个目录用于保存图片
    dataset_path = "D:\Temp\datasets"
    # 获取dataset_path下文件夹数量
    num_folders = len([name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))])
    dataset_path = os.path.join(dataset_path, f"{num_folders + 1}")
    os.mkdir(dataset_path)
    # 获取dataset路径下文件数量
    start_num = 0

    # 开始监听鼠标点击
    listener = Listener(on_click=on_click)
    listener.start()
    # Load model
    device = select_device("")
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride) \
 \
        # Run inference
    bs = 1
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    img_size = 640
    stride = 32
    auto = True
    # 加载飞易来DLL
    dll_path = os.path.join(os.getcwd(), 'msdk_x64.dll')
    dll = ctypes.WinDLL(dll_path)
    # 打开USB输入硬件
    handle = dll.M_Open(1)
    # 设置当前分辨率
    dll.M_ResolutionUsed(handle, 2560, 1440)
    global mouse_down

    try:
        while camera.is_capturing:
            # 获取最新一帧
            im0s = camera.get_latest_frame()
            if not get_key_lock_state(20):
                continue
            im = letterbox(im0s, img_size, stride=stride, auto=auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)

            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                if model.xml and im.shape[0] > 1:
                    ims = torch.chunk(im, im.shape[0], 0)

            # Inference
            with dt[1]:
                pred = model(im, augment=False, visualize=False)
            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            xOffset, yOffset = 160, 160
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                im0, frame = im0s.copy(), 0

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = names[c] if hide_conf else f"{names[c]}"
                        if label == "person":
                            # confidence = float(conf)
                            # confidence_str = f"{confidence:.2f}"
                            x, y = calculate_relative_position_int((160, 160), get_center(xyxy))
                            distance = calculate_distance((0, 0), (x, y))
                            if calculate_distance((0, 0), (xOffset, yOffset)) > distance:
                                xOffset, yOffset = x, y
                                if mouse_down:
                                    start_num += 1
                                    # if xOffset != 160 and yOffset != 160:
                                    start_time = time.time()
                                    dll.M_MoveR2(handle, int(xOffset), int(yOffset))
                                    end_time = time.time()
                                    print(f"{(end_time - start_time) * 1000} ms")
                                    print(f"{int(xOffset)} {int(yOffset)}")

                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                            annotator.box_label(xyxy, label, color=colors(c, True))

                # Stream results
                im0 = annotator.result()

            # # 显示结果帧
            # cv2.imshow('Screen Recording', im0)
            # # 检查是否按下了 'q' 键来退出
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     # camera.stop()
            #     break
    finally:
        # 确保释放资源
        dll.M_Close(handle)
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    time.sleep(3)
    mouse_down = False
    screen_record()
