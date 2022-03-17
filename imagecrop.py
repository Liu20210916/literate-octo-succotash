import cv2
import numpy as np
import albumentations as A
import os

def imagecrop(image_dir,new_path,include=[]):
    image_list = os.listdir(image_dir)
    image_list = [ i for i in image_list if i[0]!='.']
    for image in image_list:
        if image.split('.')[1] =='jpg':
            name = image.split('.')[0]
            img = cv2.imread(os.path.join(image_dir,image))
            w ,h = img.shape[1] ,img.shape[0]
            _ =image_dir.replace('images','labels')
            with open( os.path.join(_,name)+'.txt' ,'r') as f:
                norm_lb=[x.split() for x in f.read().strip().splitlines() if len(x)]
                norm_lb=np.array(norm_lb, dtype=np.float32)
                f.close()

            if include:
                for i in include:
                    norm_lb = norm_lb[norm_lb[:,0]==i,:]

            sl_w ,sl_h = w//2 ,h//2

            # for i in range(k+1):
            xyxy = np.zeros_like(norm_lb[:,1:],dtype=np.float32)
            # xmin
            xyxy[:,0] = norm_lb[:,1] - norm_lb[:,3]/2
            # xmax
            xyxy[:,1] = norm_lb[:,1] + norm_lb[:,3]/2
            # ymin
            xyxy[:,2] = norm_lb[:,2] - norm_lb[:,4]/2
            # ymax
            xyxy[:,3] = norm_lb[:,2] + norm_lb[:,4]/2

            norm_lb = norm_lb[np.array(np.array(xyxy[:,0]<0.5,dtype=np.int0)*np.array(xyxy[:,1]>0.5,dtype=np.int0)+
            np.array(xyxy[:,2]<0.5,dtype=np.int0)*np.array(xyxy[:,3]>0.5,dtype=np.int0),dtype=np.bool8)==False,:]

            for i in range(2):
                for j in range(2):
                    transform = A.Compose(
                        [A.Crop(x_min = j*sl_w, x_max = (j+1)*sl_w,
                                y_min = i*sl_h , y_max = (i+1)*sl_h,
                                p=1)],
                        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
                    new = transform(image=img, bboxes=norm_lb[:, 1:], class_labels=norm_lb[:, 0])  # transformed
                    im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
                    if labels.size > 0:
                        new_image_path = os.path.join(new_path,name+'-{}{}'.format(i,j))
                        cv2.imwrite(os.path.join(new_image_path+'.jpg') ,im)
                        labels_path = new_image_path.replace('images','labels')
                        with open(labels_path+'.txt','w+') as f:
                            for n in labels:
                                f.write('{:.0f} {:.6} {:.6} {:.6} {:.6}'.format(n[0],n[1], n[2], n[3], n[4]) + '\n')
                            f.close()



if __name__ == '__main__':
    root='D:/myproject/demo/mydata/images/train'
    new_path='D:/myproject/demo/mydata/images/train'
    imagecrop(root ,new_path ,[0])
