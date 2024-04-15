# ObjDetectSelect
基于TK实现，检测时实时自定义检测模型与检测类别。`detect_obj_select2.py`  
可供选择的模型： ` --model-cfgs，  default={'objects365':{'weights':'yolov5m_Objects365.pt','labels':'Objects365.yaml'},'coco':{'weights':'yolov5s.pt','labels':'data/coco.yaml'},'fire':{'weights':'best.pt'}}, help='path to label cfg file')`，字典形式，不限多少模型，weights: 模型weight路径，labels: 模型数据的config文件。

基于tk实现控制界面。一个线程单独监听点击事件，另一个线程执行检测程序。初始时输入初始模型与初始检测的物品类别，程序运行后可依据实时视频中显示内容在界面中自行选择切换模型以及检测类别。从而实现多模式的切换。fire: 火灾检测模型。

![image](https://github.com/Lily-Le/ObjDetectSelect/assets/21274476/ae455253-75c3-4109-a60a-615fb5d37895)



