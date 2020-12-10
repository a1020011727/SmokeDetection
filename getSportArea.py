from scipy.signal import correlate2d
from skimage.color import rgb2gray
import numpy as np
import cv2
from skimage.measure import label
import time
def imgGradent(test_1,test_2,p):
    test_image1 = rgb2gray(test_1)
    test_image2 = rgb2gray(test_2)
    prewitt = np.asarray([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    Laplace = np.asarray([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    isotropic_sobel_operator = np.asarray([[-1, -(2) ** 0.5, -1], [0, 0, 0], [1, 2 ** 0.5, 1]])
    sobel_operator = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # lapulas算子，一阶微分梯度算子，检测图像边缘 。。。索贝尔算子
    gray_image2d1 = test_image1  # 转换类型
    gray_image2d2 = test_image2
    start = time.time()
    # grad_x1 = correlate2d(gray_image2d1, Laplace, mode='same')  # 计算梯度
    grad_y1 =  cv2.Sobel(test_image1, cv2.CV_64F,0,1,ksize=5)

    # grad_x2 = correlate2d(gray_image2d2, Laplace, mode='same')
    grad_y2 =  cv2.Sobel(test_image2, cv2.CV_64F,0,1,ksize=5)
    test_image2 = test_2

    # test_image2 = Image.open('test_data/3.png')
    # test_image2 = test_image2.resize((1280, 720), Image.NONE)
    # test_image2 = cv2.resize(test_image2, (h, w), interpolation=cv2.INTER_AREA)


    grad = np.asarray((grad_y1 - grad_y2))
    print(time.time() - start)
    grad = np.where((abs(grad) > p), 1, 0).astype('uint8')
    print(time.time() - start)
    grad = cv2.medianBlur(grad, 3)
    print(time.time() - start)
    # _, binary = cv2.threshold(grad, 0, 1, cv2.THRESH_BINARY)
    # threshold = 10*10
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
        if 2000 < area < 1000000:
            cv2.rectangle(grad, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print(time.time() - start)
    cv2.imshow("img", grad)
    cv2.imwrite('directsub.jpg', grad)
    print(time.time() - start)
    # cv2.imwrite('./img.png',img=grad)
    cv2.waitKey(0)

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
    start = time.time()
    lastframe = cv2.imread('test_data/135.jpg')
    lastframe = cv2.GaussianBlur(lastframe,(3,3),0)
    lastframe = cv2.medianBlur(lastframe,3)
    frame = cv2.imread('test_data/136.jpg')
    frame = cv2.GaussianBlur(frame,(3,3),0)
    frame = cv2.medianBlur(frame,3)
    imgGradent(lastframe, frame, 0.4130000000000003)
    print(time.time()-start)