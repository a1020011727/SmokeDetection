import numpy as np
from PIL import Image
from skimage.measure import label
from scipy.signal import correlate2d
from skimage import io
from skimage.color import rgb2gray
import cv2
import time
from skimage import measure, morphology
from matplotlib import pyplot as plt
from skimage.measure import regionprops
import matplotlib.patches as mpatches
import torch

def example2d(test_1,test_2,p):
    # t1 = Image.open('test_data/2.png')
    # t1 = t1.resize((1280,720),Image.NONE)
    # t2 = Image.open('test_data/3.png')
    # t2 = t2.resize((1280, 720), Image.NONE)
    # test_image1 = t1.convert('L')
    # test_image2 = t2.convert('L')
    start1 = time.time()
    h = 800
    w = 600
    # test_image1 = cv2.imread(filename1)
    # test_image2 = cv2.imread(filename2)
    # test_image1 = cv2.resize(test_1, (h, w), interpolation=cv2.INTER_AREA)
    # test_image2 = cv2.resize(test_2, (h, w), interpolation=cv2.INTER_AREA)

    test_image1 = rgb2gray(test_1)
    test_image2 = rgb2gray(test_2)
    prewitt = np.asarray([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    Laplace = np.asarray([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    isotropic_sobel_operator = np.asarray([[-1, -(2)**0.5, -1], [0, 0, 0], [1, 2**0.5, 1]])
    sobel_operator = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # lapulas算子，一阶微分梯度算子，检测图像边缘 。。。索贝尔算子
    gray_image2d1 = test_image1  # 转换类型
    gray_image2d2 = test_image2

    # grad_x1 = correlate2d(gray_image2d1, Laplace, mode='same')  # 计算梯度
    grad_y1 = correlate2d(gray_image2d1, np.transpose(Laplace), mode='same')

    # grad_x2 = correlate2d(gray_image2d2, Laplace, mode='same')
    grad_y2 = correlate2d(gray_image2d2, np.transpose(Laplace), mode='same')

    # print(time.time() - start1)

    test_image2 = test_2
    # test_image2 = Image.open('test_data/3.png')
    # test_image2 = test_image2.resize((1280, 720), Image.NONE)
    # test_image2 = cv2.resize(test_image2, (h, w), interpolation=cv2.INTER_AREA)

    start = time.time()
    grad = np.asarray((grad_y1-grad_y2))

    grad = np.where((abs(grad)> p), 1, 0).astype('uint8')
    grad = cv2.medianBlur(grad, 3)
    grad = cv2.medianBlur(grad, 3)
    grad = cv2.medianBlur(grad, 3)
    grad = cv2.medianBlur(grad, 3)
    grad = cv2.medianBlur(grad, 3)

    # _, binary = cv2.threshold(grad, 0.1, 1, cv2.THRESH_BINARY)
    # threshold = h / 50 * w / 50
    # contours, hierarch = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # for i in range(len(contours)):
    #     area = cv2.contourArea(contours[i])  # 计算轮廓所占面积
    #     if area < threshold:  # 将area小于阈值区域填充背景色，由于OpenCV读出的是BGR值
    #         cv2.drawContours(grad, [contours[i]], 0, 0, thickness=-1)  # 原始图片背景BGR值(84,1,68)
    #         continue
    # grad = measure.label(grad, background=0, connectivity=1)
    # grad = morphology.remove_small_objects(grad,min_size=256,connectivity=1)
    grad = largestConnectComponent(grad)
    grad = np.asarray(grad).astype('uint8')
    grad = np.expand_dims(grad, axis=2)

    _, binary = cv2.threshold(grad, 0.1, 1, cv2.THRESH_BINARY)
    contours, hierarch = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grad = grad * test_image2
    count = 0
    for c in contours:
        # 获取矩形框边界坐标
        x, y, w, h = cv2.boundingRect(c)
        # 计算矩形框的面积
        area = cv2.contourArea(c)
        count += 1
        if 20 < area < 10000000:
            cv2.rectangle(grad, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # print(count)
    # print(time.time() - start)
    bbox = [x,y,x+w,y+h]
    return bbox
    # cv2.imshow("img", grad)
    #cv2.imwrite('./img.png',img=grad)
    # cv2.waitKey(0)


# def cau(grad_x1, grad_x2, grad_y1, grad_y2, test_image2, w, h):
#     for y in range(0, w):
#         for x in range(0, h):
#             if (grad_x1[x, y] == grad_x2[x, y]) & (grad_y1[x, y] == grad_y2[x, y]):
#                 test_image2[x][y] = 0
#             elif (abs(grad_x2[x, y] - grad_x1[x, y]) <= 0.05) | (abs(grad_y2[x, y] - grad_y1[x, y]) <= 0.05):
#                 test_image2[x][y] = 0
#
#     return test_image2

# im = gray_image2d2 - gray_image2d1
def largestConnectComponent(bw_img, ):
    '''
    compute largest Connect component of a binary image

    Parameters:
    ---

    bw_img: ndarray
        binary image

	Returns:
	---

	lcc: ndarray
		largest connect component.

    Example:
    ---
        >>> lcc = largestConnectComponent(bw_img)

    '''

    labeled_img, num = label(bw_img, connectivity=1, background=0, return_num=True)
    # plt.figure(), plt.imshow(labeled_img, 'gray')

    max_label = 0
    max_num = 0
    for i in range(1, num + 1):  # 这里从1开始，防止将背景设置为最大连通域
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)

    return lcc


if __name__ == '__main__':
    '''
    视频
    # cap = cv2.VideoCapture('test_data/7.mp4')
    # lastframe = None
    # while (1):
    # 
    #    (ret,frame) = cap.read()
    #     start = time.time()
    #     if lastframe is None:
    #         lastframe = frame
    #         continue
    #     res = example2d(lastframe,frame)
    #     print('帧率：'+str(1/(time.time() - start)))
    #     cv2.imshow("img",res)
    #     if cv2.waitKey(200) & 0xFF == ord('q'):
    #         break
'''
    lastframe = cv2.imread('../test_data/2.png')
    frame = cv2.imread('../test_data/3.png')
    import draw_anchor
    import bbox_iou
    n = draw_anchor()
    n = torch.Tensor([n]).cuda(0)
    first_iou = None
    i =0
    p=0.1
    b = torch.Tensor([example2d(lastframe, frame, 0.15)]).cuda(0)
    first_iou = bbox_iou(b, n)
    while True:
        b = torch.Tensor([example2d(lastframe, frame, p)]).cuda(0)
        iou = bbox_iou(b, n)
        if iou >= 0.75:
            break
        p = p - 0.001

        print(iou)
        print(p)



