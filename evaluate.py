"""
批量测试：遍历测试目录统计猫狗识别准确率

目录结构要求（子目录名即真实标签）:
    data/test/
        cat/  *.jpg|png|jpeg|bmp|webp
        dog/  *.jpg|png|jpeg|bmp|webp

用法:
    python evaluate.py                  # 默认测试 data/test
    python evaluate.py path/to/test     # 指定测试目录
    python evaluate.py --limit 100      # 每类最多测试 100 张

输出:
    终端报表 + eval_report.csv 明细（每张图的识别结果）
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

from classifier import CatDogClassifier
from config import LABEL_ZH, TEST_DIR, EVAL_REPORT_PATH

# 支持的图片扩展名
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(test_dir: str, limit: int = 0) -> dict:
    """收集各子目录下的图片，返回 {真实标签: [图片路径, ...]}"""
    samples = defaultdict(list)
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"测试目录不存在: {test_dir}")
    for label in sorted(os.listdir(test_dir)):
        sub = os.path.join(test_dir, label)
        if not os.path.isdir(sub):
            continue
        for name in sorted(os.listdir(sub)):
            if os.path.splitext(name)[1].lower() in IMG_EXTS:
                samples[label].append(os.path.join(sub, name))
        if limit:
            samples[label] = samples[label][:limit]
    if not samples:
        raise FileNotFoundError(f"目录下没有找到子目录/图片: {test_dir}")
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="猫狗识别批量测试")
    parser.add_argument("test_dir", nargs="?", default=TEST_DIR,
                        help=f"测试目录（默认 {TEST_DIR}）")
    parser.add_argument("--limit", type=int, default=0,
                        help="每类最多测试张数（0 表示不限制）")
    parser.add_argument("--model", default=None, help="自定义模型路径（可选）")
    parser.add_argument("--report", default=EVAL_REPORT_PATH,
                        help=f"结果明细 CSV 输出路径（默认 {EVAL_REPORT_PATH}）")
    args = parser.parse_args()

    try:
        samples = collect_images(args.test_dir, args.limit)
        classifier = CatDogClassifier(args.model) if args.model else CatDogClassifier()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    # 统计：混淆矩阵 key=(真实标签, 预测标签)，unknown 表示模型认为不是猫狗
    confusion = defaultdict(int)
    total, correct = 0, 0
    rows = []

    for true_label, paths in samples.items():
        for path in paths:
            total += 1
            try:
                result = classifier.predict_file(path)
                pred, conf = result["label"], result["confidence"]
                raw = f"{result['raw_class_name']}(#{result['raw_class']})"
            except IOError as e:
                # 单张坏图不中断整个测试，记为 unknown 并打印警告
                print(f"[警告] {e}", file=sys.stderr)
                pred, conf, raw = "unknown", 0.0, ""
            confusion[(true_label, pred)] += 1
            if pred == true_label:
                correct += 1
            rows.append([path, true_label, pred, f"{conf:.4f}", raw])

    # 写明细 CSV
    with open(args.report, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["图片", "真实标签", "预测标签", "置信度", "ImageNet原始类别"])
        writer.writerows(rows)

    # 输出报表
    print(f"\n===== 批量测试结果（共 {total} 张）=====")
    print(f"{'真实标签':<10}{'识别正确':>8}{'总数':>8}{'准确率':>10}")
    for true_label in sorted(samples):
        n_correct = confusion[(true_label, true_label)]
        n_total = len(samples[true_label])
        rate = n_correct / n_total if n_total else 0
        print(f"{LABEL_ZH.get(true_label, true_label):<10}"
              f"{n_correct:>8}{n_total:>8}{rate:>10.2%}")
    print(f"\n总体准确率: {correct}/{total} = {correct / total:.2%}" if total
          else "\n没有测试样本")
    # 展示误判去向（含 unknown），便于分析
    print("\n混淆明细（真实 -> 预测: 张数）:")
    for (t, p), n in sorted(confusion.items()):
        mark = "  ✔" if t == p else "  ✘"
        print(f"  {LABEL_ZH.get(t, t)} -> {LABEL_ZH.get(p, p)}: {n}{mark}")
    print(f"\n明细已保存: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
