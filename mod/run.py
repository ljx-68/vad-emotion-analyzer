from mod.intent_core.main import Intent
from mod.intent_core.Document_Management import load_corpus_csv

from pathlib import Path
from mod.intent_core.ClauseStore_File import ClauseStore
import  os

os.chdir(Path(__file__).parent.parent)
from ctypes import c_double, c_int, c_char_p, POINTER, Structure
class NeuralNetwork(Structure):
    _pack_ = 1
    _fields_ = [
        ("W", POINTER(c_double)),  # 权重指针
        ("B", c_double * 3),       # 偏置数组
        ("Y", c_double * 3),       # 输出数组
        ("in_dim", c_int),         # 输入维度
        ("out_dim", c_int)         # 输出维度
    ]
class Node:
    def __init__(self, base_dir: str = "data/results", save_data: bool = True):
        self.intent = Intent()
        base: list[str] = [f"{Path(base_dir)}/v0.3_emotion.jsonl", f"{Path(base_dir)}/v0.3_unlabeled.jsonl"]
        self.save_data = save_data

        self.emotion_store = ClauseStore(
            base[0],
            key = "text",
            max_size = 100_000,
        )
        self.unlabeled_store = ClauseStore(
            base[1],
            key = "text",
            max_size = 200_000,
        )
        self.vocab_list = [
            "不", "没", "很", "太", "非常", "有点", "挺", "简直", "更",
            "喜欢", "讨厌", "好", "差", "开心", "伤心",
            "满意", "失望", "完美", "垃圾", "舒服", "恶心",
            "方便", "麻烦", "赞", "烂", "便宜", "贵", "棒", "爽"
        ]
        self.c_train_data = []
    def Data_Loading(self, start_row: int = 100, end_row: int = None):
        corpus = load_corpus_csv("ratings.csv", start_row = start_row, end_row = end_row)
        print(f"读到 {len(corpus)} 条")
        for i, item in enumerate(corpus, 1):
            self.uo_2(i, item, len(corpus))
        if self.save_data:
            self._flush()
            self._print_stats()
    def uo_2(self, i: int, item: dict, corpus: int)->None:
        try:
            tree = self.intent.process(item["text"])
            good_data = self.extract_for_c(tree)
            self.c_train_data.extend(good_data)
            if self.save_data:
                row = tree.to_dict()
                row["scores"] = item["scores"]
                row["timestamp"] = item["timestamp"]
                is_high_quality = any(
                    c.emotions and c.confidence >= 0.75
                    for c in tree.iter_leaves()
                )
                self.uo(is_high_quality, row)
        except Exception as e:
            print(f"处理失败: {e}")
        if i % 10 == 0:
            print(f"  已处理 {i}/{corpus}")

    def extract_for_c(self, tree) -> list[dict]:
        valid_samples = []
        for leaf in tree.iter_leaves():
            if leaf.emotions and leaf.confidence >= 0.75:
                text = leaf.text.strip()
                vad = leaf.vad
                if text and vad != [0.5, 0.5, 0.5]:
                    valid_samples.append({"text": text, "vad": vad})
        return valid_samples
    def uo(self, is_high_quality: bool, row: dict)->None:
        if is_high_quality:
            self.emotion_store.append(row)
        else:
            self.unlabeled_store.append(row)
    def _flush(self):
        self.emotion_store.flush()
        self.unlabeled_store.flush()

    def _print_stats(self):
        print("\n=== 有情绪库 ===")
        print(self.emotion_store.stats())
        print("\n=== 无情绪库 ===")
        print(self.unlabeled_store.stats())

if __name__ == "__main__":
    Node(save_data = False).Data_Loading(start_row = 0, end_row = 6000)
