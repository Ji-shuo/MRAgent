# global_methods.py

import os
import time
from typing import List, Sequence, Optional

# OpenAI Python SDK v1.x
# pip install openai>=1.0.0
from openai import OpenAI
from openai._exceptions import OpenAIError, RateLimitError, APIStatusError


import os
os.environ["OPENAI_API_KEY"] = "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"

# 可选：供你现有代码调用
def set_openai_key(key_env: str = "OPENAI_API_KEY") -> None:
    """
    从环境变量里读取 OpenAI Key。你也可以在调用前手动设置 os.environ[key_env]。
    """

    api_key = os.getenv(key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"{key_env} is empty. Please export your OpenAI API key, e.g. "
            f'export {key_env}="sk-..."'
        )


def get_openai_embedding(
    texts: Sequence[str],
    model: str = "text-embedding-3-large",
    *,
    batch_size: int = 96,
    max_retries: int = 5,
    initial_backoff: float = 1.0,
    timeout: Optional[float] = 60.0,
) -> List[List[float]]:
    """
    将一批文本转成向量，返回与输入一一对应的二维数组（list of list of float）。

    参数
    ----
    texts : Sequence[str]
        要编码的文本列表。会按批发送，顺序保持不变。
    model : str
        OpenAI 向量模型名，常用：
        - "text-embedding-3-small"   # 1536 维，便宜
        - "text-embedding-3-large"   # 3072 维，质量更高
    batch_size : int
        每批请求的条数。根据你的速率/显存/超时情况调整。
    max_retries : int
        单批请求的最大重试次数（指数退避）。
    initial_backoff : float
        初始重试等待秒数，之后翻倍。
    timeout : Optional[float]
        单次 HTTP 请求超时（秒）。None 表示不设定。

    返回
    ----
    List[List[float]]
        形状 (len(texts), dim) 的嵌入列表。
    """
    if not isinstance(texts, (list, tuple)):
        raise TypeError("texts must be a list/tuple of strings")

    # 预处理：换行替换成空格，避免意外长度/格式问题
    clean_texts = [("" if t is None else str(t)).replace("\n", " ").strip() for t in texts]

    client = OpenAI(timeout=timeout)  # 读取 OPENAI_API_KEY 环境变量

    embeddings: List[List[float]] = []
    n = len(clean_texts)
    if n == 0:
        return embeddings

    # 分批请求，保持顺序
    for start in range(0, n, batch_size):
        batch = clean_texts[start : start + batch_size]

        # 指数退避重试
        attempt = 0
        backoff = initial_backoff
        while True:
            try:
                resp = client.embeddings.create(model=model, input=batch)
                # resp.data 的顺序与 input 一致
                for item in resp.data:
                    embeddings.append(item.embedding)
                break  # 成功则跳出重试循环

            except (RateLimitError, APIStatusError, OpenAIError, TimeoutError) as e:
                attempt += 1
                if attempt > max_retries:
                    # 将已拿到的结果与错误位置提示返回给调用方排查
                    raise RuntimeError(
                        f"OpenAI embedding request failed after {max_retries} retries "
                        f"at batch [{start}:{start+len(batch)}]: {e}"
                    ) from e
                # 等待后重试
                time.sleep(backoff)
                backoff *= 2.0  # 指数退避

    # 断言长度匹配（防守式）
    if len(embeddings) != n:
        raise RuntimeError(
            f"Embedding count mismatch: got {len(embeddings)} for {n} inputs."
        )
    return embeddings
