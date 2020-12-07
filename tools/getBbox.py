import os
import xml.dom.minidom
import cv2 as cv

def draw_anchor():
        imgfile = '../test_data/3.png'
        xmlfile = '../test_data/3.xml'
        # print(image)
        # 打开xml文档
        DOMTree = xml.dom.minidom.parse(xmlfile)
        # 得到文档元素对象
        collection = DOMTree.documentElement
        # 读取图片
        img = cv.imread(imgfile)
        # 得到标签名为object的信息
        objectlist = collection.getElementsByTagName("object")
        for objects in objectlist:
        # 每个object中得到子标签名为name的信息
            namelist = objects.getElementsByTagName('name')
            # 通过此语句得到具体的某个name的值
            objectname = namelist[0].childNodes[0].data
            bndbox = objects.getElementsByTagName('bndbox')
            # print(bndbox)
        for box in bndbox:
            x1_list = box.getElementsByTagName('xmin')
            x1 = int(x1_list[0].childNodes[0].data)
            y1_list = box.getElementsByTagName('ymin')
            y1 = int(y1_list[0].childNodes[0].data)
            x2_list = box.getElementsByTagName('xmax') #注意坐标，看是否需要转换
            x2 = int(x2_list[0].childNodes[0].data)
            y2_list = box.getElementsByTagName('ymax')
            y2 = int(y2_list[0].childNodes[0].data)
            bbox = [x1,y1,x2,y2]
            return bbox
            # cv.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)
            # cv.putText(img, objectname, (x1, y1), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0),
            # thickness=2)
            # cv.imshow('head', img)
            # cv.imwrite(save_path+'/'+filename, img
