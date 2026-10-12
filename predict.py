"""
单张图片猫狗识别

用法:
    python predict.py <图片路径> [--show]
示例:
    python predict.py data/test/cat/1.jpg
    python predict.py photo.jpg --show   # 弹窗显示结果（需要图形界面）
"""
import argparse
import sys

from classifier import CatDogClassifier
from config import LABEL_ZH


def main() -> int:
    parser = argparse.ArgumentParser(description="猫狗识别（OpenCV DNN）")
    parser.add_argument("image", help="待识别图片的路径")
    parser.add_argument("--show", action="store_true", help="弹窗显示识别结果")
    parser.add_argument("--model", default=None, help="自定义模型路径（可选）")
    args = parser.parse_args()

    try:
        classifier = CatDogClassifier(args.model) if args.model else CatDogClassifier()
        result = classifier.predict_file(args.image)
    except (RuntimeError, IOError) as e:
        # 关键错误统一记录到 stderr，退出码非 0 便于脚本调用方判断
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    print(f"图片: {args.image}")
    print(f"识别结果: {LABEL_ZH[result['label']]}  "
          f"(原始类别: {result['raw_class_name']})")
    print(f"置信度: {result['confidence']:.2%}")

    if args.show:
        import cv2
        image = cv2.imread(args.image)
        text = f"{LABEL_ZH[result['label']]} {result['confidence']:.1%}"
        # 根据结果着色：猫=橙、狗=绿、未知=灰
        color = {"cat": (0, 165, 255), "dog": (0, 200, 0), "unknown": (128, 128, 128)}
        cv2.putText(image, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color[result["label"]], 3)
        cv2.imshow("result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
