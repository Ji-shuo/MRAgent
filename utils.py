from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
def extract_json_from_content(text: str):
    import json, re
    t = (text or "").strip()

    # 清掉 ```json ... ``` 围栏
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I | re.M).strip()

    # 优先定位 assistantfinal / final 标记后的 JSON
    m = re.search(r"(assistantfinal|final)\s*{", t, flags=re.I)
    if m:
        start = m.end() - 1  # 指向 '{'
        # 用栈配对花括号，拿到完整 JSON 块
        depth, i = 0, start
        while i < len(t):
            if t[i] == '{':
                depth += 1
            elif t[i] == '}':
                depth -= 1
                if depth == 0:
                    block = t[start:i + 1]
                    return json.loads(block)
            i += 1
        raise ValueError("Unbalanced braces after assistantfinal/final")

    # 退而求其次：抓取文本中的**最大**花括号块（而不是最后一个）
    # 仍然用配对法，避免拿到内部的小对象
    best = None
    stack = []
    for i, ch in enumerate(t):
        if ch == '{':
            stack.append(i)
        elif ch == '}' and stack:
            left = stack.pop()
            candidate = t[left:i + 1]
            # 选择最长的（更可能是顶层对象）
            if best is None or len(candidate) > len(best):
                best = candidate
    if best:
        return json.loads(best)

    raise ValueError(f"No JSON object found. head={t[:300]!r}")


def topk_answers_by_similarity(
        question_emb,
        answers_embs,
        id_list: List[str],
        k: int = 5,
        *,
        similarity: str = "dot",  # "cosine" 或 "dot"
        answer_texts: Optional[List[str]] = None,  # 可选：候选答案文本
) -> Tuple[List[str], List[float], np.ndarray, Optional[List[str]]]:
    """
    返回：
      - top_ids:       长度<=k 的 id 列表（按相关性从高到低，已按连字符前缀去重）
      - top_scores:    同上
      - top_embs:      形状 (<=k, d) 的 embedding 矩阵（与 top_ids 对齐）
      - top_texts:     （可选）长度<=k 的文本列表（若传入 answer_texts 才会返回）
    """
    assert len(id_list) == answers_embs.shape[0], "id_list 与 answers_embs 行数不一致！"
    if answer_texts is not None:
        # print(len(answer_texts))
        # print(answers_embs.shape[0])
        assert len(answer_texts) == answers_embs.shape[0], "answer_texts 长度需与 answers_embs 行数一致！"

    question_emb = question_emb.reshape(-1)

    if similarity == "cosine":
        scores = _cosine_sim(question_emb, answers_embs)  # (N,)
    elif similarity == "dot":
        scores = np.dot(question_emb, answers_embs.T)  # (N,)
    else:
        raise ValueError("similarity 只能是 'cosine' 或 'dot'")

    # —— 按“连字符前缀”去重（仅保留该前缀得分最高的样本）——
    # 例：D1:1-3、D1:1-7 的前缀都是 D1:1，只保留得分最高的那个
    def base_prefix(x: str) -> str:
        # 只切一次，左边为前缀
        return x.split('-', 1)[0]

    best_idx_by_prefix = {}   # prefix -> (idx, score)
    for i, (id_str, sc) in enumerate(zip(id_list, scores)):
        prefix = base_prefix(id_str)
        if (prefix not in best_idx_by_prefix) or (sc > best_idx_by_prefix[prefix][1]):
            best_idx_by_prefix[prefix] = (i, sc)

    # 取出所有“代表样本”的索引和分数
    dedup_indices = np.array([v[0] for v in best_idx_by_prefix.values()], dtype=int)
    dedup_scores  = np.array([v[1] for v in best_idx_by_prefix.values()], dtype=float)

    # 从这些代表里再取 top-k
    if dedup_indices.size == 0:
        # 无候选的极端情况
        return [], [], np.empty((0, answers_embs.shape[1])), ([] if answer_texts is not None else None)

    k_eff = min(k, dedup_indices.size)
    top_dedup_idx_unsorted = np.argpartition(-dedup_scores, kth=k_eff - 1)[:k_eff]
    order = np.argsort(dedup_scores[top_dedup_idx_unsorted])[::-1]
    top_dedup_idx = top_dedup_idx_unsorted[order]

    final_indices = dedup_indices[top_dedup_idx]
    final_scores  = dedup_scores[top_dedup_idx]

    top_ids = [id_list[i] for i in final_indices]
    top_scores = final_scores.tolist()
    top_embs = answers_embs[final_indices]

    top_texts = None
    if answer_texts is not None:
        top_texts = [answer_texts[i] for i in final_indices]

    return top_ids, top_scores, top_embs, top_texts

from nltk.stem import PorterStemmer
import re
from functools import lru_cache

_stemmer = PorterStemmer()
_WORD_SPLIT = re.compile(r'([A-Za-z]+)')

@lru_cache(maxsize=100_000)
def _stem_word(word: str) -> str:
    return _stemmer.stem(word.lower())

def lemmatize_keywords(
    keywords: List[str],
    model: str = "en_core_web_sm",
    keep_case: bool = False,
    exceptions: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    使用 PorterStemmer 对关键词进行词干提取（比 spaCy lemmatization 快得多）
    """
    out = []
    for kw in keywords:
        parts = _WORD_SPLIT.split(kw)
        buf = []
        for p in parts:
            if p.isalpha():
                stemmed = _stem_word(p)
                if keep_case:
                    if p.isupper():
                        stemmed = stemmed.upper()
                    elif p.istitle():
                        stemmed = stemmed.capitalize()
                buf.append(stemmed)
            else:
                buf.append(p)
        out.append("".join(buf))
    return out
