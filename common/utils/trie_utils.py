# 新增一個工具模組檔 e.g. unintentional-unalignment/common/utils/trie_utils.py
from dataclasses import dataclass, field
from typing import Dict, Optional, Iterable, List
import torch

@dataclass
class TrieNode:
    children: Dict[int, "TrieNode"] = field(default_factory=dict)
    is_end: bool = False

class TokenTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token_ids: Iterable[int]):
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = TrieNode()
            node = node.children[tid]
        node.is_end = True

    def allowed_first_tokens(self) -> List[int]:
        return list(self.root.children.keys())

    def allowed_next_tokens(self, prefix: Iterable[int]) -> List[int]:
        node = self.root
        for tid in prefix:
            if tid not in node.children:
                return []  # no completion from this prefix
            node = node.children[tid]
        return list(node.children.keys())

def build_item_trie_and_first_token_set(id2name: Dict[str, str], tokenizer, add_leading_space: bool = True):
    trie = TokenTrie()
    first_token_set = set()
    for sid, name in id2name.items():
        text = (" " + name) if add_leading_space else name
        # 注意：和你計算 seq logp 的 tokenizer 設定一致
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue
        trie.insert(token_ids)
        first_token_set.add(token_ids[0])
    return trie, sorted(first_token_set)
