# 猫狗识别（OpenCV DNN）

基于 OpenCV DNN 模块加载预训练 MobileNetV2（ONNX）模型的猫狗二分类项目，
支持单张图片识别与批量测试，输出中文结果与置信度。

## 技术方案

- **模型**：MobileNetV2（ImageNet 1000 类预训练，ONNX 格式），来自 [ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet)
- **推理**：OpenCV `dnn.readNetFromONNX`，无需深度学习框架
- **二分类映射**：ImageNet 的狗类（索引 151~268）与猫类（索引 281~285）概率累加取大者；
  其余类别判定为 `unknown`（非猫狗），可缓解 `grey fox`、`necklace` 等相邻/干扰类别的误判

## 环境搭建（conda）

```bash
# 1. 创建并激活环境
conda create -n dogcat python=3.11 -y
conda activate dogcat

# 2. 安装依赖（国内清华镜像）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 模型与标签文件

已随项目提供，位于 `models/` 目录：

| 文件 | 说明 | 来源 |
| --- | --- | --- |
| `mobilenetv2-12.onnx` | 预训练模型（约 13.3MB） | `https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx` |
| `imagenet_classes.txt` | ImageNet 1000 类标签（每行一个类别） | 从 torchvision 包中提取（`torchvision/models/_meta.py`） |

> 国内网络直连 `raw.githubusercontent.com` 可能失败，github.com 主站与清华 PyPI 镜像通常可用。

## 数据集（开源）

批量测试目录结构要求（子目录名即真实标签）：

```
data/test/
├── cat/   # 猫图片
└── dog/   # 狗图片
```

### 方式一：Kaggle Cats vs Dogs（推荐，25000 张）

```bash
# 需先在 kaggle.com -> Settings -> API 生成 kaggle.json 并放置到 ~/.kaggle/
pip install kaggle -i https://pypi.tuna.tsinghua.edu.cn/simple
kaggle datasets download -d mustafaogan/cats-and-dogs-image-classification
unzip cats-and-dogs-image-classification.zip -d data/
# 按实际解压出的目录结构，将图片归入 data/test/cat 与 data/test/dog
```

### 方式二：下载样例图（快速验证，约 10 张）

```bash
python tools/download_samples.py   # 通过 Wikimedia Commons API 下载猫狗样例
```

## 使用方法

```bash
# 单张图片识别
python predict.py data/test/cat/cat1.jpg
python predict.py data/test/dog/dog1.jpg --show   # --show 弹窗显示（需要图形界面）

# 批量测试，输出分类准确率报表 + eval_report.csv 明细
python evaluate.py                  # 默认测试 data/test
python evaluate.py path/to/test     # 指定测试目录
python evaluate.py --limit 100      # 每类最多 100 张
```

## 项目结构

```
dog-and-cat/
├── config.py                 # 路径、猫狗类别索引映射、预处理参数
├── classifier.py             # 核心：模型加载、预处理、softmax、二分类映射
├── predict.py                # 单张图片识别入口
├── evaluate.py               # 批量测试与准确率报表
├── tools/download_samples.py # 样例图片下载工具（Wikimedia Commons）
├── models/                   # 预训练模型与标签文件
├── data/test/{cat,dog}/      # 测试图片（需自行准备）
├── eval_report.csv           # 批量测试明细（运行后生成）
└── requirements.txt
```

## 测试建议

1. **功能验证**：分别对猫、狗图片运行 `predict.py`，确认标签与置信度正常（0~100%）
2. **异常验证**：`predict.py 不存在的路径.jpg` 应输出中文错误且退出码非 0
3. **批量评估**：用 Kaggle 数据集抽样（`--limit 200`）跑 `evaluate.py`，
   参考准确率约 85%~95%（受样张质量影响）
4. **非猫狗图**：放入风景/人物图片，应识别为 `未知（非猫狗）`
