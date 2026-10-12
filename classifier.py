"""
核心模块：加载 OpenCV DNN 模型，完成预处理、推理和猫狗二分类映射
"""
import cv2
import numpy as np

from config import (
    MODEL_PATH, INPUT_SIZE, MEAN, STD,
    DOG_INDICES, CAT_INDICES,
)

# ImageNet 1000 类英文标签（每行一个类别，按索引顺序）
LABEL_PATH = MODEL_PATH.replace("mobilenetv2-12.onnx", "imagenet_classes.txt")


def _load_class_names(path: str) -> list:
    """加载 ImageNet 类别标签文件，校验行数，失败时给出明确错误"""
    try:
        with open(path, encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
    except OSError as e:
        raise RuntimeError(f"无法读取类别标签文件 {path}: {e}") from e
    if len(names) != 1000:
        raise RuntimeError(f"类别标签数量异常，期望 1000，实际 {len(names)}: {path}")
    return names


CLASS_NAMES = _load_class_names(LABEL_PATH)


class CatDogClassifier:
    """猫狗二分类器（基于 OpenCV DNN 加载 ONNX 预训练模型）"""

    def __init__(self, model_path: str = MODEL_PATH):
        # 加载模型，失败时记录错误并抛出，避免后续推理出现难以定位的问题
        self.net = cv2.dnn.readNetFromONNX(model_path)
        if self.net.empty():
            raise RuntimeError(f"模型加载失败或文件为空: {model_path}")
        self.model_path = model_path

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """图像预处理：缩放 + ImageNet 归一化 + 通道顺序转换，返回 NCHW 张量"""
        resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        # BGR -> RGB，转浮点并归一化到 [0, 1]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # 逐通道减均值除标准差
        rgb = (rgb - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
        # HWC -> CHW，增加 batch 维度
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
        return blob

    def predict(self, image: np.ndarray) -> dict:
        """
        识别单张图像

        返回:
            {
                "label": "cat" | "dog" | "unknown",
                "confidence": float,        # 置信度 [0, 1]
                "raw_class": int,           # ImageNet 原始类别索引
                "raw_class_name": str,      # ImageNet 原始类别说明
            }
        """
        blob = self.preprocess(image)
        self.net.setInput(blob)
        logits = self.net.forward().flatten()

        # 模型输出为 logits，做数值稳定的 softmax 转为概率
        shifted = logits - logits.max()
        exp = np.exp(shifted)
        probs = exp / exp.sum()

        raw_idx = int(np.argmax(probs))

        # 二分类映射：不在单点 argmax 上硬判定，而是把落在
        # ImageNet 猫类/狗类集合内的概率分别累加，取分高者，
        # 可缓解 grey fox、necklace 等相邻/干扰类别的边界误判
        dog_score = float(probs[np.fromiter(DOG_INDICES, int)].sum())
        cat_score = float(probs[np.fromiter(CAT_INDICES, int)].sum())

        if dog_score > cat_score and dog_score > 0:
            label, confidence = "dog", dog_score
        elif cat_score > 0:
            label, confidence = "cat", cat_score
        else:
            label, confidence = "unknown", float(probs[raw_idx])

        return {
            "label": label,
            "confidence": confidence,
            "raw_class": raw_idx,
            "raw_class_name": CLASS_NAMES[raw_idx],
        }

    def predict_file(self, image_path: str) -> dict:
        """读取图片文件并识别，读取失败时抛出带路径的异常"""
        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f"无法读取图片（文件不存在或格式不支持）: {image_path}")
        return self.predict(image)

