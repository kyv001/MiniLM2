from torch.utils.data import Dataset
import torch
import numpy as np
import json
import os

class DatasetWithMask(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
            self,
            data_path: str,
            mask_path: str,
            max_length: int,
            subset_indexes: list[int] | None = None,
            used_indexes: list[int] | None = None):
        self.data_path = data_path
        self.mask_path = mask_path
        self.max_length = max_length
        data = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        self.n_lines = data.shape[0] // max_length
        self.data = data[:self.n_lines * max_length].reshape((self.n_lines, max_length))
        mask = np.memmap(self.mask_path, dtype=bool, mode="r")
        self.mask = mask[:self.n_lines * max_length].reshape((self.n_lines, max_length))
        self.n_lines = data.shape[0] // max_length
        self.data = data[:self.n_lines * max_length].reshape((self.n_lines, max_length))
        self.used_indexes = set(used_indexes or [])

        if subset_indexes is None:
            self.subset_indexes = set(range(self.n_lines)) # 默认使用整个数据集
        else:
            self.subset_indexes = set(subset_indexes)

        self.unused_indexes = list(self.subset_indexes - self.used_indexes) # 未使用的行索引

    def __len__(self) -> int:
        return len(self.unused_indexes) # 返回未使用的行数

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = index % len(self) # 确保索引在范围内
        abs_index = self.unused_indexes[index]
        line = self.data[abs_index]
        mask = self.mask[abs_index]
        self.used_indexes.add(abs_index)
        y = torch.from_numpy(line.copy()).type(torch.long) # 复制以防止torch报警告，同时转换类型以防止模型报错
        m = torch.from_numpy(mask.copy()).type(torch.long)
        return y, m

    def get_used_indexes(self) -> list[int]:
        return sorted(self.used_indexes)

def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    y_list, m_list = zip(*batch)
    y = torch.stack(y_list, dim=0)
    m = torch.stack(m_list, dim=0)
    return y, m

def from_file(config_path: str, max_lenth: int) -> tuple[DatasetWithMask, DatasetWithMask]: # 训练集、验证集
    # 根据配置文件读取数据集数据
    # 配置文件实例：
    # {
    #     "?path": "数据集本体相对路径",
    #     "path": "dataset.bin",
    #     "?mask_path": "掩码文件相对路径",
    #     "mask_path": "mask.bin",
    #     "?train": "训练集索引文件相对路径",
    #     "train": "train.lst",
    #     "?train_used": "训练集已用数据索引文件相对路径",
    #     "train_used": "train_used.lst"
    # }
    config = json.load(open(config_path))
    dir_path = os.path.dirname(config_path)
    path = os.path.join(dir_path, config["path"])
    mask_path = os.path.join(dir_path, config["mask_path"])

    train_indexes = None
    if "train" in config:
        train_path = os.path.join(dir_path, config["train"])
        train_indexes = [*map(int, open(train_path).read().split())]

    train_used_indexes = None
    if "train_used" in config:
        train_used_path = os.path.join(dir_path, config["train_used"])
        train_used_indexes = [*map(int, open(train_used_path).read().split())]
    train_dataset = DatasetWithMask(path, mask_path, max_lenth, subset_indexes=train_indexes, used_indexes=train_used_indexes)
    
    valid_indexes = train_indexes or [0]
    if "valid" in config:
        valid_path = os.path.join(dir_path, config["valid"])
        valid_indexes = [*list(map(int, open(valid_path).read().split()))]
    valid_dataset = DatasetWithMask(path, mask_path, max_lenth, subset_indexes=valid_indexes)
    return train_dataset, valid_dataset

if __name__ == '__main__':
    import sys
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer # type: ignore
    if len(sys.argv) < 3:
        print("Usage: python -m minilm2.llm.dataset_sft <tokenizer_path> <data_path>")
        exit(1)
    tokenizer_path = sys.argv[1]
    data_path = sys.argv[2]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(f"Loading dataset from {data_path}...")
    dataset, _ = from_file(data_path, 1024)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    for y, m in dataloader:
        x = ((y - 2) * (1 - m)) + 2
        for i in range(len(y[0])):
            if m[0][i].item():
                print("\033[31m", end="")
            else:
                print("\033[0m", end="")
            print(tokenizer.convert_ids_to_tokens([x[0][i].item()])[0], end="")
        try:
            input()
        except EOFError:
            break
        