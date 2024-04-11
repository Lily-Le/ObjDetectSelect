#%%
import rosbag
from cv_bridge import CvBridge
import cv2
import os
from sensor_msgs.msg import Image
import torch
import matplotlib.pyplot as plt
CONFIDENCE_THRESHOLD = 0.5
bridge = CvBridge()
fp = '/home/cll/work/data/project/robot/2023-10-09-16-36-26.bag'

weights = 'yolov5m_Objects365.pt'
model = torch.hub.load('.', 'custom', path=weights, source='local',device=0)

with rosbag.Bag(fp, 'r') as bag:
    base_dir = '/sensors/stereo_cam'
    depth_fp = '/depth/depth_registered'
    rgb_fp = '/rgb/image_rect_color/compressed'
        #topic是/center_camera/image_color/compressed/
    i = 0
    print(bag.get_type_and_topic_info())
#%%
    depth_imgs = []
    rgb_imgs = []
        
    for topic, msg, t in bag.read_messages(base_dir+rgb_fp):
        rgb_imgs.append( bridge.compressed_imgmsg_to_cv2(msg, "bgr8"))
    for topic, msg, t in bag.read_messages(base_dir+depth_fp):
            # 由于topic是被压缩的，所以采用compressed_imgmsg_to_cv2读取
        depth_imgs.append(bridge.imgmsg_to_cv2(msg, "passthrough") ) # Convert the message to an image
           
    #%%
for frame,dep in zip(rgb_imgs,depth_imgs):
    results = model(frame)
    for i in range(len(results.pred[0])):
        bbox = results.xyxy[0][i]
        label = results.names[bbox[-1].item()]
        confidence = bbox[-2].item() # 调节查全率与查准率
        # if label == 'Trash bin Can' and confidence > CONFIDENCE_THRESHOLD:
        if  confidence > CONFIDENCE_THRESHOLD:
            # 左上角： (int(bbox[0].item()), int(bbox[1].item()))
            # 右下角： (int(bbox[2].item()), int(bbox[3].item()))
            cv2.rectangle(frame, (int(bbox[0].item()), int(bbox[1].item())), (int(bbox[2].item()), int(bbox[3].item())), (0, 0, 255), 2)
            cv2.putText(frame, label, (int(bbox[0].item()), int(bbox[1].item())-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            center_point = (int((bbox[0].item()+bbox[2].item)/2),int((bbox[1].item()+bbox[3].item)/2))
            # 在center_point处画点
            cv2.circle()
            
            
            
    cv2.imshow('', frame)
    if cv2.waitKey(10) & 0XFF == ord('q'):
        break

    
        i +=1
        # 如果topic是无压缩的，可以采用bridge.imgmsg_to_cv2(msg,"bgr8")
        print(i)
        timestr = "%.6f" % msg.header.stamp.to_sec()
        # %.6f表示小数点后带有6位，可根据精确度需要修改；
        image_name = timestr + ".png"  # 图像命名：时间戳.png
        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path)
        cv2.imwrite(os.path.join(rgb_path,image_name), cv_image)  # 保存

    #%% 
class RosDataset():
    def __init__(self, bag, depth_dir, rgb_dir):
        self.bag = bag
        self.depth_dir = depth_dir
        self.rgb_dir = rgb_dir
        with rosbag.Bag(self.bag, 'r') as bag:
            for topic, msg, t in bag.read_messages(self.depth_dir ):
                self.depth_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            for topic, msg, t in bag.read_messages(self.rgb_dir):
                self.rgb_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        assert len(self.rgb_image) == len(self.depth_image), "The size of rgb message should be the same as the size of depth message!"
    
    def __len__(self):
        return len(self.rgb_image)
    
    def __getitem__(self,index):
        depth_im = self.depth_image[index]
        rgb_im = self.rgb_image[index]
        return rgb_im,depth_im
    

        
    
# %%

