### 文件说明

```bash
hist-cell
├─assets                 # README.md 相关图片
├─dataset                # 数据集
│  └─hist-cell		
│      ├─annotations     # coco格式的标注文件
│      ├─gt				# 原图的可视化（分部位）
│      ├─pred			# 不同模型预测结果的可视化（分部位）
│      │  ├─mask-rcnn	 
│      │  ├─rtmdet		
│      │  ├─solov2		
│      │  └─sparseinst	
│      ├─test			# 测试集（分部位）
│      ├─train			# 训练集
│      └─val			# 验证集
├─models			    # 模型相关 
│   ├─configs            # 配置文件
│   ├─loss			    # loss图
│   ├─map			    # map图
│   ├─matrix             # 混淆矩阵
│   └─results            # 测试结果
│       ├─mask-rcnn
│       ├─rtmdet
│       ├─solov2
│       └─sparseinst
├─eval                  # test集合的评估标准的扩充，fpr暂时有点问题，因为里面的tn（预测为负例但实际为正例的样本数）太少，导致fp/(fp+tn)永远趋于1
│                       # rtmdet的准确率等有点问题，有空我再看看
├─data_preprocess.py    # 数据预处理文件
├─labelme2coco.py       # 将原始的labelme格式数据集转为coco格式
├─README.md             # 项目说明
├─tiqu.py               # 提取文件的tool代码
├─tree.txt              # mmdetection平台的结构树
└─update_json.py        # 部位的coco标注文件的缺少类别数，补全类别代码
```



### 查看数据

#### 原始数据

<img src="assets/image-20230702104223345.png" alt="image-20230702104223345" style="zoom:70%;" /> 

经过对原始数据的观察，可以看到数据集中是包含各个部位的png图片以及打上标签后的json文件

其中包含十九个部位，分别对应

| Liver | Esophagus | Uterus | Skin | Testis | Bile-duct | Breast | Pancreatic | Cervix | Lung | Prostate | Ovarian | Adrenal | HeadNeck | Colon | Kidney | Stomach | Bladder | Thyroid |
| ----- | --------- | ------ | ---- | ------ | --------- | ------ | ---------- | ------ | ---- | -------- | ------- | ------- | -------- | ----- | ------ | ------- | ------- | ------- |
| 肝    | 食道      | 子宫   | 皮肤 | 睾丸   | 胆管      | 乳房   | 胰腺       | 子宫颈 | 肺   | 前列腺   | 卵巢    | 肾上腺  | 头颈     | 结肠  | 肾脏   | 胃      | 膀胱    | 甲状腺  |

#### 数据划分

考虑对数据集的划分，暂定为各个部位各取0.7，0.2，0.1分别作为训练集、验证集和测试集

然后将三个集合的json格式转化为coco格式利用mmedetection平台选取实例分割进行训练



### 终端参数

#### 训练参数

通用

超参（sparseinst使用的不是轮次epochs，而是迭代次数iters）

- 训练批处理大小 batch_size
- 训练轮次 max_epochs（sparseinst使用max_iters）
- 验证轮次 val_interval 
- 基础学习率 lr
- 权重衰减 weight_decay （一般不用改，要改也行）

```bash
# 训练批处理大小 train_dataloader.batch_size=1
# 训练轮次 train_cfg.max_epochs、验证轮次 train_cfg.val_interval
# 优化器：基础学习率 optimizer.lr、权重衰减 optimizer.weight_decay
python tools/train.py user/mask-rcnn.py --cfg-options optim_wrapper.optimizer.lr=0.02 optim_wrapper.optimizer.weight_decay=0.0001 train_cfg.max_epochs=12 train_cfg.val_interval=1 train_dataloader.batch_size=1
```

rtmdet

```bash
# max_epochs是两阶段总共的训练轮数，interval是验证轮数，[(90, 1)]代表从第90轮开始为第二阶段，第二阶段每轮都进行一次验证
python tools/train.py user/rtmdet-ins_tiny.py --cfg-options max_epochs=100 interval=10 train_cfg.dynamic_intervals="[(90, 1)]"
```

默认参数(solov2有个梯度裁剪clip_grad，不过还是别修改了，优化器1、3是SGD，2、4是Adam)

|                  | max_epochs/max_iters | batch_size | val_interval | lr      | dynamic_intervals | weight_decay |
| ---------------- | -------------------- | ---------- | ------------ | ------- | ----------------- | ------------ |
| mask-rcnn        | 12                   | 8          | 1            | 0.02    | -                 | 0.0001       |
| rtmdet-inst_tiny | 120                  | 8          | 10           | 0.004   | [100, 1]          | 0.05         |
| solov2_x         | 24                   | 8          | 1            | 0.01    | -                 | -            |
| sparseinst       | 120000               | 8          | 10000        | 0.00005 | -                 | 0.05         |



#### 测试参数

主要就是修改测试集文件

```bash
# example
python .\tools\test.py .\user\mask-rcnn.py .\work_dirs\mask-rcnn\epoch_12.pth --test-ann annotations/Bladder_coco.json 
```

假如说我选择Stomach这个部位的文件夹，可以通过代码将`annotations/Colon_coco.json`中的Colon替换为`Stomach`



### 测试结果json文件说明

```bash
# counts [T, R, K, A, M]
t = 10       # iou阈值从0.5到0.95，按0.05递增
r = 101      # recall召回率从0.到1.，按0.01递增   
k = 5        # 类别数
a = 4		# 预测面积，all、small、medium、large		
m = 3		# 最大检测数，不太懂	
# precision  # 不同recall下的精确率
# accuracy   # 准确率
# tpr        灵敏度
# fpr        特效度
```

![image-20230710225430977](assets/image-20230710225430977.png) 



获取每个iouThr对应的tp，fp，tn，fn

iouThr从0-1有对应的点（fpr，tpr），可以绘制roc曲线



### 模型使用过程中一些参考命令

```bash
# conda 环境激活
conda init
CALL conda.bat activate [env_name]

# 使用 tools/test.py 脚本存储检测结果
python tools/test.py ${CONFIG} ${MODEL_PATH} --out results.pkl

# 混淆矩阵
python tools/analysis_tools/confusion_matrix.py ${CONFIG}  ${DETECTION_RESULTS}  ${SAVE_DIR} --show

# 计算分离和遮挡目标的掩码的召回率
python tools/analysis_tools/coco_occluded_separated_recall.py results.pkl --out occluded_separated_recall.json --ann data/hist-cell/annotations/test_coco.json

# 得到 bbox 或 segmentation 的 json 格式文件
python tools/test.py configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py      checkpoint/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth

python tools/analysis_tools/coco_error_analysis.py results.segm.json results --ann=data/coco/annotations/instances_val2017.json --types='segm'
```