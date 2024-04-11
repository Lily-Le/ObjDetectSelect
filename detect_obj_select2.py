# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5m_Objects365.pt --source 0 /home/cll/work/code/project/video/batc_lab1.mp4                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import threading

import torch
import yaml
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders_rtps import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

import cv2
import tkinter as tk
import queue
# pipeline = "rtspsrc location=\"rtsp://login:password@host:port/\" ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, format=(string)BGRx! videoconvert ! appsink"
# capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
pipline = "rtsp://10.181.184.53:8554/cam"
# while capture.isOpened():
#     res, frame = capture.read()
#     cv2.imshow("Video", frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break
# capture.release()
# cv2.destroyAllWindows()

@smart_inference_mode()
def run(
        # weights=ROOT / 'yolov5m_Objects365.pt',  # model path or triton URL
        model_cfgs,
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        # obj_to_dectect = 'All'
        
):
    global obj_to_dectect
    global obj_queue
    global last_model
    global weight_queue
    source = str(source) #pipline #str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    current_model = weight_queue.queue[-1]
    current_weights = model_cfgs[current_model]['weights']
    model = DetectMultiBackend(current_weights, device=device, dnn=dnn, data=data, fp16=half)
    last_model = current_model
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
        print(f'bs{bs}')
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    for path, im, im0s, vid_cap, s in dataset:
        current_model = weight_queue.queue[-1]
        current_weights = model_cfgs[current_model]['weights']
        if current_model != last_model:
            model = DetectMultiBackend(current_weights, device=device, dnn=dnn, data=data, fp16=half)
            last_model = current_model
            stride, names, pt = model.stride, model.names, model.pt
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        k = cv2.waitKey(1)
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        obj_to_dectect = obj_queue.queue[-1]
                        if (obj_to_dectect in label) or (obj_to_dectect == 'All'):
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if k == 1:
                            mouseX, mouseY = cv2.getWindowImageRect('Object Detection')  # 获取窗口坐标
                            if xyxy[0] < mouseX < xyxy[2]  and xyxy[1] < mouseY < xyxy[3]:
                                print(f"鼠标点击了物体：{label}")
                            # annotator.box_label(xyxy, "中文", color=colors(c, True))
                        # annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")


    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(current_weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5m_Objects365.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='/home/cll/work/code/project/fire-smoke-detect-yolov4/yolov5/dataset/ForestFire1.avi', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--model-cfgs', type=str, default={'objects365':{'weights':'yolov5m_Objects365.pt','labels':'Objects365.yaml'},'coco':{'weights':'yolov5s.pt','labels':'data/coco.yaml'},'fire':{'weights':'best.pt'}}, help='path to label cfg file')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

# def input_listener():
#     global obj_to_dectect
#     while True:
#         obj_to_dectect = input("请输入您想要识别的物品名：")
#         print(f'您想要识别的物品名是：{obj_to_dectect}')

def input_listener(label_list):
    global obj_to_detect
    global obj_queue
    def update_obj_to_detect():
        selected_item = listbox.curselection()
        if selected_item:
            obj_to_detect = listbox.get(selected_item[0])
            obj_queue.put(obj_to_detect)
            # success_label.config(text=f"成功更新！ 目前检测的物品是 {obj_to_detect}")
            success_label.config(text=f"Update Succeeded! The object to detect is now: {obj_to_detect}")
    
    def search_items():
        search_text = search_entry.get()
        filtered_items = [item for item in items if search_text.lower() in item.lower()]
        listbox.delete(0, tk.END)
        for item in filtered_items:
            listbox.insert(tk.END, item)

    # 创建新窗口
    window = tk.Tk()
# 创建搜索框
    search_entry = tk.Entry(window)
    search_entry.pack()
    # 创建输入框和按钮
    # input_entry = tk.Entry(window)
    # input_entry.pack()
    
    listbox = tk.Listbox(window)
    listbox.pack()
    
    items = label_list # 替换为实际的物品名称列表
    for item in items:
        listbox.insert(tk.END, item)
    # update_button = tk.Button(window, text="更新物品", command=update_obj_to_detect)
    update_button = tk.Button(window, text="Update object: ", command=update_obj_to_detect)

    update_button.pack()

    success_label = tk.Label(window, text="")
    success_label.pack()
    # 进入窗口的事件循环
    window.mainloop()

def input_listener2(label_dict,last_model):
    global obj_to_detect
    global obj_queue
    global weight_queue
    
    model_names = label_dict.keys()
    # last_model = init_model
    def update_obj_to_detect():
        selected_item = label_list.curselection()
        if selected_item:
            obj_to_detect = label_list.get(selected_item[0])
            obj_queue.put(obj_to_detect)
            # success_label.config(text=f"成功更新！ 目前检测的物品是 {obj_to_detect}")
            success_label.config(text=f"Update Succeeded! The object to detect is now: {obj_to_detect}")
    
    def search_items():
        search_text = search_entry.get()
        filtered_items = [item for item in items if search_text.lower() in item.lower()]
        label_list.delete(0, tk.END)
        for item in filtered_items:
            label_list.insert(tk.END, item)

    def update_model():
        global last_model
        selected_model = model_list.get(model_list.curselection())
        weight_queue.put(selected_model)
        # if last_model is None:
        #     last_model = init_model
        # if last_model != selected_model:
        # if selected_model != 'fire':
        for n in model_names:
                    if selected_model == n:
                        label_list.delete(0, tk.END)
                        for item in label_dict[n]:
                            label_list.insert(tk.END, item)
            # last_model = selected_model
        # return last_model
        
    # 创建新窗口
    window = tk.Tk()
# 创建搜索框
    search_entry = tk.Entry(window)
    search_entry.pack()
    # 创建输入框和按钮
    # input_entry = tk.Entry(window)
    # input_entry.pack()
    model_list = tk.Listbox(window)
    for n in model_names:
        model_list.insert(tk.END, n)
    model_list.bind("<<ListboxSelect>>", update_model)
    model_list.pack()
    
    label_list = tk.Listbox(window)
    label_list.pack()
    
    items = label_dict[last_model] # 替换为实际的物品名称列表
    for item in items:
        label_list.insert(tk.END, item)
    
    # last_model = init_model
    # update_button = tk.Button(window, text="更新物品", command=update_obj_to_detect)
    update_button_obj = tk.Button(window, text="Update object: ", command=update_obj_to_detect)

    update_button_obj.pack()

    update_button_model = tk.Button(window, text="Update model: ", command=update_model)

    update_button_model.pack()

    success_label = tk.Label(window, text="")
    success_label.pack()
    # 进入窗口的事件循环
    window.mainloop()

def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))



if __name__ == '__main__':
    global obj_to_dectect
    # obj_to_dectect = input("请输入您想要识别的物品名：")
    obj_to_dectect = input("Input the object to detect: ")
    print(f'The object to detect is: {obj_to_dectect}')
    init_model= input("Input the initial model name: ")
    print(f'The initial model is: {init_model}')
        
    global obj_queue
    global weight_queue
    obj_queue = queue.Queue()
    weight_queue = queue.Queue()
    obj_queue.put(obj_to_dectect) #,block=False)
    weight_queue.put(init_model)
    opt = parse_opt()    
    
    
    
    label_cfgs={}
    weight_paths={}
    for n in opt.model_cfgs.keys():
        weight_paths[n]=opt.model_cfgs[n]['weights']
        if ('labels' in opt.model_cfgs[n].keys()) and opt.model_cfgs[n]['labels'] is not None:
            with open(opt.model_cfgs[n]['labels'],'r') as f:
                label_cfgs[n] = yaml.load(f,Loader=yaml.FullLoader)
        else:
            label_cfgs[n] = None
            print(f'No label configs for model {n}')
    
    label_lists={}
    
    for model_nm in label_cfgs.keys():
        label_lists[model_nm]=[]
        if label_cfgs[model_nm] is not None:
            for n in label_cfgs[model_nm]['names']:
                label_lists[model_nm].append(label_cfgs[model_nm]['names'][n])
            label_lists[model_nm].sort()
    # del opt.model_cfgs #用完就扔
    # global obj_to_dectect
    # obj_to_dectect = input("请输入您想要识别的物品名：")
    # print(label_list)
    # init_model = 'objects365'
    thread1 = threading.Thread(target=main, args=(opt,))
    thread2 = threading.Thread(target=input_listener2,args = (label_lists,init_model,))
    
    thread1.daemon = True
    
    thread1.start()
    thread2.start()
    
    thread1.join()

    
