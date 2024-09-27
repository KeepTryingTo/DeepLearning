
import os
import cv2
import time
import cvzone
import numpy as np
import onnxruntime
from PIL import Image

import torch
from torch import nn
from torchvision import transforms

import torch
from torchvision.ops import nms
from werkzeug.utils import secure_filename
from flask import Flask,render_template,request,redirect,url_for,abort,jsonify,Response

app=Flask(__name__)

conf_threshold = 0.20
iou_threshold = 0.05
img_size = 640
device = "cuda" if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([
    transforms.ToTensor()
])

className = []
with open(file=r'D:\conda3\Transfer_Learning\B Stand\day18\main\Flask_deploy\Flask\yolov5_classes.txt',
          mode='r', encoding='utf-8') as fp:
    lines = fp.readlines()
for line in lines:
    className.append(line.strip('\n'))
#加载模型
onnxYolov5 = onnxruntime.InferenceSession(r"D:\conda3\Transfer_Learning\B Stand\day18\main\Flask_deploy\weights\yolov5s.onnx")

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def onnxDetectImage(threshold = 0.45,iou_threshold = 0.05,
        img_path = r'D:\conda3\Transfer_Learning\B站\day18\main\Flask_deploy\images\6.png'):

    start = time.time()
    im0 = Image.open(img_path)
    width, height = im0.size

    im = im0.resize(size=(img_size, img_size))
    im = transform(im).unsqueeze(dim=0).to('cpu')
    im = im.numpy()

    predictions = onnxYolov5.run(output_names=['output0'], input_feed={'images': im})
    # [1,25200,85] = [1,25200,xywh + conf + 80]
    print('predictions.shape: {}'.format(predictions[0].shape))
    pred = np.squeeze(predictions)
    boxes = xywh2xyxy(pred[..., :4])
    confidences = pred[..., 4]
    cls_prob = pred[..., 5:85]
    labels = np.argmax(cls_prob, axis=-1)
    confidences = confidences * np.max(cls_prob, axis=-1)
    boxes, confidences, labels = (torch.tensor(boxes, dtype=torch.float32),
                                  torch.tensor(confidences,dtype=torch.float32), torch.tensor(labels))

    indexs = nms(boxes=boxes, scores=confidences, iou_threshold=iou_threshold)
    boxes = boxes[indexs]
    confidences = confidences[indexs]
    labels = labels[indexs]

    # TODO PIL convert Opencv
    frame = np.array(im0)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    for k in range(boxes.size()[0]):
        # 左上角坐标(xleft,yleft)和右下角坐标(xright,yright)
        xleft = int(boxes[k][0] / img_size * width)
        yleft = int(boxes[k][1] / img_size * height)
        xright = int(boxes[k][2] / img_size * width)
        yright = int(boxes[k][3] / img_size * height)

        confidence = confidences[k].item()
        class_id = labels[k].item()

        # 这里只输出检测是人并且概率值最大的
        if confidence > conf_threshold:
            text = className[class_id] + ': ' + str('{:.2f}%'.format(confidence * 100))
            cv2.rectangle(frame, (xleft, yleft), (xright, yright), (255, 0, 255), 2)
            cvzone.putTextRect(img=frame, text=text, pos=(xleft + 9, yleft - 12),
                               scale=1, thickness=1, colorR=(0, 255, 0))
    # cv2.imwrite(os.path.join('outputs', imgName), frame)
    # cv2.imshow('img', frame)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    end = time.time()

    return frame, (end-start)


@app.route(rule='/submit_conf',methods=['POST','GET'])
def submit_conf():
    global conf_threshold
    conf_threshold = request.form.get('slider_value', type = float)
    print("conf_threshold: ",conf_threshold)
    return jsonify(value = conf_threshold)

@app.route(rule='/submit_iou',methods=['POST','GET'])
def submit_iou():
    global iou_threshold
    iou_threshold = request.form.get('slider_iou_value', type = float)
    print("iou_threshold: ",iou_threshold)
    return jsonify(value = iou_threshold)

@app.route(rule='/image',methods=['POST','GET'])
def appDetectImage():
    global conf_threshold
    global iou_threshold
    if request.method=='POST':
        file = request.files.get('filename')
        if file:
            #对文件进行完全检测
            filename=secure_filename(file.filename)
            print('filename: {}'.format(file.filename))
            # 获得当前的图片
            save_imgPath=os.getcwd()+'\\static\\images\\'+filename
            print('save image path: {}'.format(save_imgPath))
            file.save(save_imgPath)

            cv_img,runTime = onnxDetectImage(threshold=conf_threshold,iou_threshold=iou_threshold,img_path=save_imgPath)
            #保存当前的图片到static文件夹中
            if os.path.isfile(save_imgPath):
                os.remove(save_imgPath)
                cv2.imwrite(os.path.join(os.getcwd(),'static','images',filename),cv_img)
            # cv2.imshow('img',cv_img)
            # cv2.waitKey(0)

            response={
                "detect_time": round(runTime * 1000,2),
                "image_path":"./static/images/"+filename
            }
            #注意这里返回到index中的图片路径问题
            return render_template(template_name_or_list='detectImage.html',response=response)
    response={
            "detect_time": "",
            'image_path':""
        }
    return render_template(template_name_or_list='detectImage.html',response=response)

def timeDetect():
    # 计算开始时间
    start_time = time.time()
    # 计算帧率
    countFPS = 0
    # 开启摄像头
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(src=frame, dsize=(img_size, img_size))
        frame = cv2.flip(src=frame, flipCode=2)
        # 将其opencv读取的图像格式转换为PIL读取的类型格式
        frame_PIL = Image.fromarray(frame)
        width, height = frame_PIL.size
        img_transform = transform(frame_PIL).unsqueeze(dim=0).to('cpu')
        im = img_transform.numpy()

        predictions = onnxYolov5.run(output_names=['output0'], input_feed={'images': im})
        # [1,25200,85] = [1,25200,xywh + conf + 80]
        # print('predictions.shape: {}'.format(predictions[0].shape))
        pred = np.squeeze(predictions)
        boxes = xywh2xyxy(pred[..., :4])
        confidences = pred[..., 4]
        cls_prob = pred[..., 5:85]
        labels = np.argmax(cls_prob, axis=-1)
        confidences = confidences * np.max(cls_prob, axis=-1)
        boxes, confidences, labels = (torch.tensor(boxes, dtype=torch.float32),
                                      torch.tensor(confidences, dtype=torch.float32), torch.tensor(labels))

        indexs = nms(boxes=boxes, scores=confidences, iou_threshold=iou_threshold)
        boxes = boxes[indexs]
        confidences = confidences[indexs]
        labels = labels[indexs]

        for k in range(boxes.size()[0]):
            # 左上角坐标(xleft,yleft)和右下角坐标(xright,yright)
            xleft = int(boxes[k][0] / img_size * width)
            yleft = int(boxes[k][1] / img_size * height)
            xright = int(boxes[k][2] / img_size * width)
            yright = int(boxes[k][3] / img_size * height)

            confidence = confidences[k].item()
            class_id = labels[k].item()

            # 这里只输出检测是人并且概率值最大的
            if confidence > conf_threshold:
                text = className[class_id] + ': ' + str('{:.2f}%'.format(confidence * 100))
                cv2.rectangle(frame, (xleft, yleft), (xright, yright), (255, 0, 255), 2)
                cvzone.putTextRect(img=frame, text=text, pos=(xleft + 9, yleft - 12),
                                   scale=1, thickness=1, colorR=(0, 255, 0))

        image = cv2.imencode('.jpg', frame)[1].tobytes()
        ## 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route(rule='/video', methods=['POST', 'GET'])
def appVideo():
    ## 这个地址返回视频流响应
    return Response(timeDetect(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route(rule='/',methods=['POST','GET'])
def index():
    if request.method == "POST":
        btn_type = request.form.get('submit')
        if btn_type == "detectImage":
            return redirect(url_for("image"))
        elif btn_type == "detectTime":
            return redirect(url_for("video"))
    return render_template('index.html')


if __name__ == '__main__':
    print('Pycharm')
    app.run(debug=True)