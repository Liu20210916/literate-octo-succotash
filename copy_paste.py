import cv2
import numpy as np



def copy_paste(sour_path,dest_path):
    # 首先读取源image和目标image
    src=cv2.imread(sour_path+'.jpg')
    dst=cv2.imread(dest_path+'.jpg')
    sourlabel_path = sour_path.replace('images','labels')

    with open(sourlabel_path+'.txt') as f:
        # 读取源image中的所有label，并转换成相应list，此时是xywh格式
        lb=[x.split() for x in f.read().strip().splitlines() if len(x)]
        lb=np.array(lb, dtype=np.float32)
        f.close()

    # 转换成xyxy格式
    y = xywhn2xyxy(lb[:, 1:], w=1280, h=720)
    # 求出矩阵的四个角
    y=xyxyxyxy(y)

    poly = np.array(y, np.int32)
    for nums,i in enumerate(poly):
        src_mask = np.zeros(src.shape, src.dtype)
        cv2.fillPoly(src_mask, [i], (255, 255, 255))
        # 得到矩阵的中心
        minx,miny,maxx,maxy=np.min(i[:,0]),np.min(i[:,1]),np.max(i[:,0]),np.max(i[:,1])
        center_x,center_y = (minx+maxx)/2 ,(miny+maxy)/2

        # 得到相应矩阵中xy的相对位置
        lb[nums,1] = lb[nums,1] - center_x / src.shape[1]
        lb[nums, 2] = lb[nums, 2] - center_y / src.shape[0]

        center = (np.random.randint(1,src.shape[1]-200)+100,np.random.randint(1,src.shape[0]-200)+100)
        dst = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)
        lb[nums, 1] = lb[nums, 1] + center[0] / src.shape[1]
        lb[nums, 2] = lb[nums, 2] + center[1] / src.shape[0]

    cv2.imwrite(dest_path + '.jpg', dst)
    new_path = dest_path.replace('images', 'labels')
    with open(new_path+'.txt','a') as f:
        for i in lb:
            f.write('1 {:.6} {:.6} {:.6} {:.6}'.format(i[1], i[2], i[3], i[4]) + '\n')
        f.close()


# 该函数实现，将xyxy输出正（长）方形边界框的四个点
def xyxyxyxy(x):
    y = np.zeros((x.shape[0],8), x.dtype)
    # 左上
    y[:,:2]=x[:,:2]
    # 右上
    y[:,2:4]=x[:,[2,1]]
    # 右下
    y[:,4:6]=x[:,2:]
    # 左下
    y[:,6:]=x[:,[0,3]]
    return y.reshape(x.shape[0],4,2)


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


if __name__ == '__main__':
    sour_path='D:/myproject/demo/mydata/images/train/56'
    dest_path='D:/myproject/demo/mydata/images/train/0'
    copy_paste(sour_path,dest_path)
