import ctypes
import json
import time
from ctypes import c_double, c_int, c_char_p, POINTER, Structure, c_char
from pathlib import Path
from typing import List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent
DLL_PATH = str(BASE_DIR / "tfe_engine.pyd")
VOCAB_PATH = r"./Neuro-nutrients/final_vocab.json"
DATASET_PATH = r"./Neuro-nutrients/my_dataset_final.json"
MODEL_SAVE_PATH = str(BASE_DIR / "hybrid_model.bin")

class NeuralNetwork(Structure):
    _pack_ = 1
    _fields_ = [
        ("W", POINTER(c_double)),
        ("B", c_double * 3),
        ("Y", c_double * 3),
        ("in_dim", c_int),
        ("out_dim", c_int)
    ]


class CSample(Structure):
    _pack_ = 1
    _fields_ = [
        ("text", c_char * 256),
        ("VAD", c_double * 3)
    ]


class TextFeatureExtractor(Structure):
    _pack_ = 1
    _fields_ = [
        ("vocab", POINTER(c_char_p)),
        ("vocab_size", c_int),
        ("X", POINTER(c_double))
    ]


class CEngineTrainer:
    def __init__(self, dll_path: str = DLL_PATH, EPOCHS: int = 1000, LEARNING_RATE: float = 0.05):
        self.lib = ctypes.CDLL(dll_path)
        self.EPOCHS = EPOCHS
        self.LEARNING_RATE = LEARNING_RATE
        self._setup_c_functions()

    def _setup_c_functions(self) -> None:
        self.lib.tfe_network_init.argtypes = [POINTER(NeuralNetwork), c_int]
        self.lib.tfe_network_init.restype = c_int
        self.lib.save_weights.argtypes = [POINTER(NeuralNetwork), c_char_p]
        self.lib.save_weights.restype = c_int
        self.lib.load_weights.argtypes = [POINTER(NeuralNetwork), c_char_p]
        self.lib.load_weights.restype = c_int

        # 训练接口
        self.lib.train_batch_for_python.argtypes = [
            POINTER(NeuralNetwork), POINTER(c_double), POINTER(c_double), c_int, c_double
        ]
        self.lib.train_batch_for_python.restype = None

        # 纯 C 推理接口
        self.lib.tfe_extractor_init.argtypes = [POINTER(TextFeatureExtractor), POINTER(c_char_p), c_int]
        self.lib.tfe_extractor_init.restype = c_int
        self.lib.predict_for_python.argtypes = [
            POINTER(TextFeatureExtractor), POINTER(NeuralNetwork), POINTER(CSample), POINTER(c_double)
        ]
        self.lib.predict_for_python.restype = None

    @staticmethod
    def load_json(filepath: str | Path) -> Any:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def prepare_features(self, text: str, vocab_list: List[str]) -> List[float]:
        vocab_size = len(vocab_list)
        X = [0.0] * vocab_size
        for i, word in enumerate(vocab_list):
            if word in text:
                X[i] = 1.0
        return X

    def train(self, dataset: List[Dict[str, Any]], vocab_list: List[str]) -> None:
        vocab_size = len(vocab_list)
        batch_size = len(dataset)

        print(f"✅ 训练加载成功！当前词表维度: {vocab_size}，数据 {batch_size} 条")

        batch_X = (c_double * (batch_size * vocab_size))()
        batch_Y = (c_double * (batch_size * 3))()

        print("🧠 正在提取训练特征...")
        for b, item in enumerate(dataset):
            X = self.prepare_features(item["text"], vocab_list)
            Y = item["vad"]
            for i in range(vocab_size):
                batch_X[b * vocab_size + i] = X[i]
            for i in range(3):
                batch_Y[b * 3 + i] = Y[i]

        net = NeuralNetwork()
        init_res = self.lib.tfe_network_init(ctypes.byref(net), vocab_size)
        if init_res != 0:
            raise RuntimeError(f"C 网络初始化失败，错误码: {init_res}")

        print(f"🚀 C 引擎全速运转中 (训练 {self.EPOCHS} 轮)...")
        start_time = time.perf_counter()

        for epoch in range(self.EPOCHS):
            self.lib.train_batch_for_python(
                ctypes.byref(net), batch_X, batch_Y, batch_size, c_double(self.LEARNING_RATE)
            )

        end_time = time.perf_counter()
        print(f"✅ 训练完成！{self.EPOCHS} 轮总耗时: {end_time - start_time:.4f} 秒")

        save_res = self.lib.save_weights(ctypes.byref(net), MODEL_SAVE_PATH.encode('utf-8'))
        if save_res == 0:
            print(f"✅ 模型已成功保存为 {Path(MODEL_SAVE_PATH).name}")
        else:
            print("❌ 模型保存失败！")

    def predict(self, text: str, vocab_list: List[str]) -> List[float]:
        vocab_size = len(vocab_list)
        vocab_file = Path(VOCAB_PATH)
        dataset_file = Path(DATASET_PATH)
        vocab_list = CE.load_json(vocab_file)
        dataset = CE.load_json(dataset_file)
        CE.train(dataset, vocab_list)
        net = NeuralNetwork()
        self.lib.tfe_network_init(ctypes.byref(net), vocab_size)
        load_res = self.lib.load_weights(ctypes.byref(net), MODEL_SAVE_PATH.encode('utf-8'))
        if load_res != 0:
            raise RuntimeError(f"加载模型失败，错误码: {load_res}")
        ext = TextFeatureExtractor()
        vocab_array = (c_char_p * vocab_size)(*[v.encode("utf-8") for v in vocab_list])
        self.lib.tfe_extractor_init(ctypes.byref(ext), vocab_array, vocab_size)
        sample = CSample()
        sample.text = text.encode("utf-8")[:255]  # 防溢出
        sample.VAD[0] = 0.0
        sample.VAD[1] = 0.0
        sample.VAD[2] = 0.0

        out_y = (c_double * 3)()
        self.lib.predict_for_python(ctypes.byref(ext), ctypes.byref(net), ctypes.byref(sample), out_y)

        return [out_y[0], out_y[1], out_y[2]]



if __name__ == "__main__":
    CE = CEngineTrainer()
    vocab_file = Path(VOCAB_PATH)
    dataset_file = Path(DATASET_PATH)
    vocab_list = CE.load_json(vocab_file)
    dataset = CE.load_json(dataset_file)
    CE.train(dataset, vocab_list)
    print("\n" + "=" * 40)
    print("🔮 纯 C 引擎极速推理终端已启动")
    print("=" * 40)

    while True:
        user_text = input("请输入测试句子 (输入 q 退出): ")
        if user_text.lower() == 'q':
            break
        if not user_text.strip():
            continue

        vad = CE.predict(user_text, vocab_list)
        print(f"输入: {user_text}")
        print(f"预测 VAD: [V:{vad[0]:.3f}, A:{vad[1]:.3f}, D:{vad[2]:.3f}]\n")