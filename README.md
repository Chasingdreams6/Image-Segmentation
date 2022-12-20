# 目录结构说明
-compare.py 训练与可视化代码
-datasetplit.py 划分数据集代码，只运行一次
-mydataset.py 定义数据集类
-unet.py 定义unet网络结构

# 运行说明
- data文件夹应与上述4个python代码在同一目录下，假设是./目录
- 首先运行datasetsplit.py一次，会在./目录下生成new_data文件夹，内部按照4：1的比例随机将数据集分成了训练集和验证集
- 运行compare.py，是训练的主要代码，内部有注释；
- compare会创建saves目录，里面每20个epoch保存model, scheduler和optimizer，以及loss和iou等数据
- 训练结束后会生成iou对比图, loss变化图和混淆矩阵，insight_epoch表示要查看混淆矩阵的epoch
- 每个epoch会输出loss和iou，以及当前epoch花费的时间
- 在RTX3060 Laptop GPU上，一个epoch平均需要37秒


