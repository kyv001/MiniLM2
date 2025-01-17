from typing import Optional
from torch.utils.data import Dataset
import torch
import numpy as np
import json
import os

class SFTDataset(Dataset):
    def __init__(self, data_path: str, mask_path: str, max_lenth: int, unused_indexes: Optional[list[int]] = None):
        self.data_path = data_path
        self.mask_path = mask_path
        self.max_lenth = max_lenth
        data = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        self.n_lines = data.shape[0] // (max_lenth + 1)
        self.data = data[:self.n_lines * (max_lenth + 1)].reshape((self.n_lines, max_lenth + 1))
        mask = np.memmap(self.mask_path, dtype=np.bool, mode="r")
        self.mask = mask[:self.n_lines * (max_lenth + 1)].reshape((self.n_lines, max_lenth + 1))

        if unused_indexes is None:
            self.unused_indexes = list(range(self.n_lines)) # 默认所有行都未使用
        else:
            self.unused_indexes = unused_indexes
        self.used_indexes: list[int] = [] # 已使用的行索引

    def __len__(self) -> int:
        return len(self.unused_indexes) # 返回未使用的行数

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        abs_index = self.unused_indexes[index] # 从索引表中取出绝对索引
        line = self.data[abs_index]
        self.used_indexes.append(abs_index) # 记录已使用的行索引
        x = torch.from_numpy(line[:-1].copy()).type(torch.long) # 复制以防止torch报警告
        y = torch.from_numpy(line[1:].copy()).type(torch.long) # 同时转换类型以防止模型报错
        m = torch.from_numpy(self.mask[abs_index][1:].copy()).type(torch.long)
        return x, y, m

    def get_unused_indexes(self) -> list[int]:
        return list(set(self.unused_indexes) - set(self.used_indexes))

def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_list, y_list, m_list = zip(*batch)
    x = torch.stack(x_list, dim=0)
    y = torch.stack(y_list, dim=0)
    m = torch.stack(m_list, dim=0)
    return x, y, m

def from_file(config_path: str, max_lenth: int) -> SFTDataset: # SFT不需要验证集
    # 根据配置文件读取数据集数据
    # 配置文件实例：
    # {
    #     "?path": "数据集本体相对路径",
    #     "path": "dataset.bin",
    #     "?mask_path": "掩码文件相对路径",
    #     "mask_path": "mask.bin",
    #     "?train": "训练集索引文件相对路径",
    #     "train": "train.lst"
    # }
    config = json.load(open(config_path))
    dir_path = os.path.dirname(config_path)
    path = os.path.join(dir_path, config["path"])
    mask_path = os.path.join(dir_path, config["mask_path"])
    train_path = os.path.join(dir_path, config["train"])
    train_indexes = list(map(int, open(train_path).read().split()))
    train_dataset = SFTDataset(path, mask_path, max_lenth, unused_indexes=train_indexes)
    return train_dataset

if __name__ == '__main__':
    import sys
    from torch.utils.data import DataLoader
    from tokenizers import Tokenizer # type: ignore
    if len(sys.argv) < 3:
        print("Usage: python -m minilm2.llm.dataset_sft <tokenizer_path> <data_path>")
        exit(1)
    tokenizer_path = sys.argv[1]
    data_path = sys.argv[2]
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"Loading dataset from {data_path}...")
    dataset = from_file(data_path, 1024)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    for x, y, m in dataloader:
        for i in range(len(y[0])):
            if m[0][i].item():
                print("\033[31m", end="")
            else:
                print("\033[0m", end="")
            print(tokenizer.id_to_token(y[0][i].item()), end="")
        try:
            input()
        except EOFError:
            break
        