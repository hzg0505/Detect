import torch
import cv2
import matplotlib.pyplot as plt

#@save
def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

#@hzg
def MouseOnImg(img, winName='img'):
    '''
    mouse callback
    CV_EVENT_MOUSEMOVE  0   滑动   EVENT_LBUTTONDOWN   1   左键点击  EVENT_RBUTTONDOWN   2   右键点击
    EVENT_MBUTTONDOWN   3   中间点击  EVENT_LBUTTONUP 4   左键释放  EVENT_RBUTTONUP 5   右键释放
    EVENT_MBUTTONUP 6   中间释放  EVENT_LBUTTONDBLCLK 7   左键双击  EVENT_RBUTTONDBLCLK 8   右键双击
    EVENT_MBUTTONDBLCLK 9   中间双击
    :param img:
    :return:
    '''
    bbox = []
    def callback(event, x, y, flags, param):
        if event == 1:
            print(x, y)
            bbox.append((x,y))
    img_copy = img.copy()

    cv2.namedWindow(winName)
    cv2.setMouseCallback(winName, callback)
    # 设置回调函数窗口 回调函数
    while(True):
        cv2.imshow(winName, img_copy)
        if(len(bbox)==2):
            cv2.rectangle(img_copy, bbox[0], bbox[1], (0, 255, 0), thickness=1)
            bbox = []
        if cv2.waitKey(20) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
# MouseOnImg(img)

#@save
def bbox_to_rect(bbox, color):
    # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
    # ((左上x,左上y),宽,高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

# PATH and get the data of image.
# imgPath = "F:\code\pytorch\Detect\datasets\data\img.png"
# img = cv2.imread(imgPath)
# the bounding box of the cat
# catbbox = [253, 118, 425, 310]
# img_copy = img.copy()
# cv2.rectangle(img_copy, (catbbox[0], catbbox[1]), (catbbox[2], catbbox[3]), (0, 255, 0), thickness=1)
# cv2.imshow("img", img_copy)
# cv2.waitKey(0)

# fig = plt.imshow(img)
# fig.axes.add_patch(bbox_to_rect(catbbox, 'blue'))
# plt.show()
# fig.axes.add_patch(bbox_to_rect(catbbox, 'red'))