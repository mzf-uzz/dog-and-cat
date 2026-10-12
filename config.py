"""
项目配置文件：路径、ImageNet 类别到猫/狗二分类的映射、预处理参数
"""
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 预训练模型路径（MobileNetV2, ImageNet 1000 类, ONNX 格式）
MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenetv2-12.onnx")

# 数据集目录（数据集需自行下载，见 README.md）
DATA_DIR = os.path.join(BASE_DIR, "data")
TEST_DIR = os.path.join(DATA_DIR, "test")

# 批量测试结果输出路径
EVAL_REPORT_PATH = os.path.join(BASE_DIR, "eval_report.csv")

# ---------- ImageNet 类别索引到猫/狗的映射 ----------
# ImageNet 1000 类中：
#   151 ~ 268：各品种狗（共 118 类）
#   281 ~ 285：猫（虎斑猫、野猫、波斯猫、暹罗猫、埃及猫）
# 其余索引视为"非猫狗"，识别结果记为 unknown
DOG_INDICES = set(range(151, 269))
CAT_INDICES = set(range(281, 286))

# 中文标签
LABEL_ZH = {"cat": "猫", "dog": "狗", "unknown": "未知（非猫狗）"}

# ---------- 模型预处理参数 ----------
INPUT_SIZE = 224          # MobileNetV2 输入尺寸
# ImageNet 标准归一化参数（逐通道）
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
