import numpy as np
import torch
import cv2

# 之前的策略是：训练一个二分模型，然后将二分模型误判为真样本的假样本纳入数据集（即修改其标签）
# 现在的策略：不用二分模型，生成样本后，直接用baseline预测，然后将置信度高的假样本调整为真样本；
def star_aug(predict,label,thresh):

    # 读取预测txt文件，和label.txt文件
    with open(predict,'r') as f:
        lb_p=[x.split() for x in f.read().strip().splitlines() if len(x)]
        lb_p = np.array(lb_p, dtype=np.float32)
        lb_p = lb_p[lb_p[:, 5] > thresh, :]
        f.close()
    if lb_p.size>0:
        with open(label,'r') as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            lb = np.array(lb, dtype=np.float32)
            f.close()

        # 将两个都转化成tensor
        lb_p=torch.tensor(lb_p)
        lb=torch.tensor(lb)
        lo = xywhn2xyxy(lb[:,1:],w = 1280,h = 720).int()

        # 从lb里面挨个取数，并与预测label里的每个框计算IOU，如果某个样本IOU超过阈值，且预测为0，而真实类别为1，那么将1修改为0;
        for nums,i in enumerate(lb):
            # 如果本身就是正样本，就跳过，不需要修改
            if i[0]!=1:
                continue
            # 计算当前边界框与所有预测边界框的IOU；
            _ = bbox_iou(i[1:], lb_p[:, 1:5], x1y1x2y2=False)
            if torch.max(_) > 0.6:
                # 走到这一步，首先保留下来的pre_lb要求置信度都大于设定的阈值；
                # 然后还要和相应数据集中label==1的样本高度重合（IOU>0.6）
                # 满足了上述条件，就将其label修改为0；
                lb[nums,0] = 0
            else:
                img = cv2.imread(label.replace('labels','images').split('.')[0]+'.jpg')
                img[ lo[nums, 1]:lo[nums, 3],lo[nums, 0]:lo[nums, 2], :] = (114, 114, 114)
                cv2.imwrite(label.replace('labels','images').split('.')[0]+'.jpg',img)

        # 用w，即覆盖原label
        with open(label,'w') as f:
            for i in lb:
                f.write('{:.0f} {:.6} {:.6} {:.6} {:.6}'.format(i[0],i[1], i[2], i[3], i[4]) + '\n')
            f.close()



def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU





if __name__ == '__main__':
    # source是我们预测txt路径；target是我们数据集txt路径
    label='D:/myproject/demo/mydata/labels/train/2-8869.txt'
    predict='D:/myproject/demo/mydata/labels/2-8869.txt'
    star_aug(predict,label,0.01)