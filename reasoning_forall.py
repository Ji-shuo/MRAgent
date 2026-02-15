"""
Agent Memory (Episodic + Semantic) with Tool-Calling LLM
--------------------------------------------------------
- Single dispatcher tool: `memory_dispatcher(op, args)` to keep tool-count small.
- Episodic memory: graph-like KV (key nodes + event edges with tags and text).
- Semantic memory: aggregated summaries promoted by tag-usage thresholds.
- Retrieval: multi-step reasoning over memory (seed keys -> choose tags -> gather evidence -> answer).

This file is a minimal-yet-complete template. Fill backend_* stubs with your
own storage (FAISS/pgvector/Neo4j/etc.) or use the provided in-memory backend.

Run:
  pip install openai
  export OPENROUTER_API_KEY=...            # or OPENAI_API_KEY
  export BASE_URL=https://openrouter.ai/api/v1   # use OpenAI official by unsetting BASE_URL
  export MODEL=openai/gpt-4o               # or any model supporting tools
  python agent_memory.py

Notes:
- Keep tools count to 1 (dispatcher) to avoid context bloat and misrouting.
- Always include `tools` in *every* request (including after tool results).
- Force JSON-only outputs for Planner/Navigator/Synthesizer to simplify control.
"""

from tools import TOOLS
import os
import json
import re
from prompts2 import Prompts
from typing import List, Dict, Any, Set
from collections import defaultdict
from utils import extract_json_from_content, topk_answers_by_similarity
from LLM.llm_controller import LLM
from memory_controller import MemoryController
from memory_system import MemorySystem
from MRAgent import Agent
import config
from pathlib import Path
from data.get_data import get_data
from conv_embedding import embed_session, embed_question
import numpy as np
import logging
from log.logging_utils import per_sample_log
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar
from embed_translate import  embed_sample
from data.transform_keyword import transform_keyword
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
run_id_var = ContextVar("run_id", default="-")
logger = logging.getLogger(__name__)
class ContextFilter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        # 注入 run_id 到日志格式占位符
        record.run_id = run_id_var.get()
        return True

def setup_logging(level=logging.INFO, logfile="app.log"):
    fmt = "%(asctime)s %(levelname)s %(name)s [run_id=%(run_id)s] %(message)s"
    datefmt = "%H:%M:%S"

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt, datefmt))

    fileh = RotatingFileHandler(logfile, maxBytes=5_000_000, backupCount=3)
    fileh.setLevel(level)
    fileh.setFormatter(logging.Formatter(fmt, datefmt))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(fileh)

    # 让所有日志记录都带上 run_id
    root.addFilter(ContextFilter())

def set_run_id(value: str):
    run_id_var.set(value)


def get_graph_data():
    path = Path("conversation_list.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    session_list = []

    for j, (sample_id, sample) in enumerate(data.items()):
        if not isinstance(sample, dict):
            continue
        for i, (session_id, session) in enumerate(sample.items()):
            session_list.append((session_id, session))

        return session_list

import random

def question_format(dataset, qa):
    if dataset == "locomo":
        if qa.get("category") == 5:
            question = qa['question'] + " Select the correct answer: {} or {}. "
            if random.random() < 0.5:
                question = question.format('Not mentioned in the conversation', qa['adversarial_answer'])
                answer = {'a': 'Not mentioned in the conversation', 'b': qa['adversarial_answer']}
            else:
                question = question.format(qa['adversarial_answer'], 'Not mentioned in the conversation')
                answer = {'b': 'Not mentioned in the conversation', 'a': qa['adversarial_answer']}
        elif qa.get("category") == 2:
            question = qa['question']
        elif qa.get("category") == 1:
            question = qa['question'] + (" No extra explanations. ")
        elif qa.get("category") == 3:
            question = qa['question']# + (" No extra explanations in 'answer'. Give reasons with original text in 'reason'. ") #
        else:
            question = qa['question']
    else:
        question = qa['question']

    return question

def average_analyses(analyses: list[dict[str, float]]) -> dict[str, float]:
    sums = defaultdict(lambda: defaultdict(float))  # sums[cat][key]
    counts = defaultdict(lambda: defaultdict(int))  # counts[cat][key]

    for analysis in analyses:
        cat = analysis.get("category")
        if cat is None:
            continue
        for k, v in analysis.items():
            if k == "category":
                continue
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            sums[cat][k] += val
            counts[cat][k] += 1

    # 计算均值
    result: Dict[Any, Dict[str, float]] = {}
    for cat, kv in sums.items():
        result[cat] = {k: kv[k] / counts[cat][k] for k in kv}
    return result

def get_question(dataset, agent, question_list, sample_id, memory, result_path, evidence_path, analysis_path, cost_path, question_embeddings=None):
    # path = Path("question_list.json")
    logger.info(f"---------------{sample_id}-------------------")

    qa_list = question_list[sample_id]  # json.loads(path.read_text(encoding="utf-8"))
    # num = int(sample_id.split('-')[1])
    # if num < config.sample_id:
    #     return
    # if num > config.sample_id:
    #     return
    result_list = {}
    # with open(result_path, "w", encoding="utf-8") as f:
    # for j, (sample_id, qa_list) in enumerate(data_qa.items()):
    question_list = ""
    all_analyses = []
    with open(cost_path, "a", encoding="utf-8") as cf:
        cost = {"prompt_tokens": agent.llm.prompt_tokens, "completion_tokens": agent.llm.completion_tokens,
                "position": 2, "sample": sample_id}
        cf.write(json.dumps(cost, ensure_ascii=False, default=list) + "\n")

    for i, qa in enumerate(qa_list, start=1):
        if dataset == "locomo":
            if qa.get("category") != config.ca:
                continue


        logger.info(f"---------------question{i}-------------------")
        with open(analysis_path, "a", encoding="utf-8") as vf:
            vf.write(json.dumps(f"question{i}", ensure_ascii=False) + "\n")


        question = question_format(dataset, qa)
        evidence_list = qa.get("evidence")
        category = qa.get("category")

        # 对于 LM 数据集的 temporal-reasoning 类型问题，直接传递 question_date 作为时间锚点
        override_question_time = None
        if dataset == "LM" and category == "temporal-reasoning":
            question_date_raw = qa.get("question_date")  # 格式: "2023/05/30 (Tue) 23:40"
            if question_date_raw:
                # 提取日期部分并转换为 "YYYY-MM-DD, YYYY-MM-DD" 格式
                date_part = question_date_raw.split(" ")[0]  # "2023/05/30"
                date_formatted = date_part.replace("/", "-")  # "2023-05-30"
                override_question_time = f"{date_formatted}, {date_formatted}"
                logger.info(f"Temporal-reasoning question, using question_date as time anchor: {override_question_time}")

        if config.USE_EMBEDDING:
            question_emb = question_embeddings[i-1]
            # print(list(question_emb))
            results, evidence_pre, evidence_support, analysis = agent.answer_question(i, question, evidence_list, analysis_path, category, question_emb, override_question_time)
        else:
            results, evidence_pre, evidence_support, analysis = agent.answer_question(i, question, evidence_list, analysis_path, category, override_question_time=override_question_time)

        # accumulate(analysis)
        all_analyses.append(analysis)
        answer = qa.get("answer")


        with open(evidence_path, "a", encoding="utf-8") as ef:
            # record = {session_id: events}  # 组成一个字典
            ef.write(json.dumps(f"question{i}", ensure_ascii=False) + "\n")
            # print(evidence_pre)
            ef.write(json.dumps(evidence_pre, ensure_ascii=False, default=list) + "\n")
            ef.write(json.dumps(evidence_support, ensure_ascii=False, default=list) + "\n")
            ef.write(json.dumps(evidence_list, ensure_ascii=False, default=list) + "\n")
        evaluation = {"answer":answer,"prediction":results,"category":category,"evidence":evidence_list,
                      "question":qa.get("question"),"prediction_context":evidence_support,"sample":sample_id}
        # print(evaluation)
        with open(result_path, "a", encoding="utf-8") as f: 
            f.write(json.dumps(evaluation, ensure_ascii=False, default=list) + "\n")
            # break
    with open(cost_path, "a", encoding="utf-8") as cf:
        cost = {"prompt_tokens": agent.llm.prompt_tokens, "completion_tokens": agent.llm.completion_tokens,
                "position": 3, "sample":sample_id}
        cf.write(json.dumps(cost, ensure_ascii=False, default=list) + "\n")

    avg = average_analyses(all_analyses)
    with open(analysis_path, "a", encoding="utf-8") as vf:
        vf.write(json.dumps(avg, ensure_ascii=False, default=list) + "\n")
        vf.write(json.dumps([config.K1, config.K2], ensure_ascii=False, default=list) + "\n")
        # if i % 10 ==0:
        # cost = {"prompt_tokens":agent.llm.prompt_tokens, "completion_tokens":agent.llm.completion_tokens, "cost":agent.llm.cost,}
        # vf.write(json.dumps(cost, ensure_ascii=False, default=list) + "\n")



def split_batch_to_sessions(translate_path: str, keyword_path: str,
                            out_translate_path: str, out_keyword_path: str,
                            batch_size: int = 10):
    """
    将batch格式的translate和keyword文件拆分成单session格式

    Args:
        translate_path: 输入的batch格式translate文件路径
        keyword_path: 输入的batch格式keyword文件路径
        out_translate_path: 输出的单session格式translate文件路径
        out_keyword_path: 输出的单session格式keyword文件路径
        batch_size: 每个batch包含的session数量（用于解析）
    """
    import re
    from collections import defaultdict

    # 解析sentence id获取session编号 (D1:1-1 -> 1)
    def get_session_num(sentence_id: str) -> int:
        match = re.match(r'D(\d+):', sentence_id)
        if match:
            return int(match.group(1))
        return -1

    # 处理translate文件
    session_data = {}  # session_num -> session_data

    with open(translate_path, 'r', encoding='utf-8') as f:
        for line in f:
            batch = json.loads(line.strip())
            batch_key = list(batch.keys())[0]  # e.g., "session_1-session_10"
            data = batch[batch_key]

            # 获取batch的session范围
            parts = batch_key.split('-')
            start_session = int(parts[0].replace('session_', ''))
            end_session = int(parts[1].replace('session_', ''))

            # 初始化这个batch范围内的session
            for s in range(start_session, end_session + 1):
                session_data[s] = {
                    'conversation_time': data.get('conversation_time', ''),
                    'sentence': [],
                    'topics': {},  # 原始topic_id -> topic_text
                    'personal_sentences': [],
                    '_topic_set': set()  # 临时存储引用的topic ids
                }

            # 按session分配sentences，收集引用的topic ids
            for sent in data.get('sentence', []):
                session_num = get_session_num(sent.get('id', ''))
                if session_num in session_data:
                    session_data[session_num]['sentence'].append(sent)
                    for topic_id in sent.get('topic', []):
                        session_data[session_num]['_topic_set'].add(topic_id)

            # 为每个session重新编号topics (从t1开始)
            batch_topics = data.get('topics', {})
            for s in range(start_session, end_session + 1):
                old_to_new = {}  # 旧topic_id -> 新topic_id
                new_topics = {}
                for idx, old_tid in enumerate(sorted(session_data[s]['_topic_set']), start=1):
                    new_tid = f't{idx}'
                    old_to_new[old_tid] = new_tid
                    if old_tid in batch_topics:
                        new_topics[new_tid] = batch_topics[old_tid]
                session_data[s]['topics'] = new_topics

                # 更新sentences中的topic引用
                for sent in session_data[s]['sentence']:
                    if 'topic' in sent:
                        sent['topic'] = [old_to_new.get(t, t) for t in sent['topic'] if t in old_to_new]

                del session_data[s]['_topic_set']  # 删除临时字段

            # 按session分配personal_sentences
            for ps in data.get('personal_sentences', []):
                origin = ps.get('origin', '')
                session_num = get_session_num(origin + ':')  # origin格式是 "D1:1"
                if session_num == -1:
                    # 尝试直接从origin解析
                    match = re.match(r'D(\d+)', origin)
                    if match:
                        session_num = int(match.group(1))
                if session_num in session_data:
                    session_data[session_num]['personal_sentences'].append(ps)

    # 写入拆分后的translate文件
    os.makedirs(os.path.dirname(out_translate_path), exist_ok=True)
    with open(out_translate_path, 'w', encoding='utf-8') as f:
        for session_num in sorted(session_data.keys()):
            record = {f'session_{session_num}': session_data[session_num]}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # 处理keyword文件
    session_keywords = {s: {'sentence': []} for s in session_data.keys()}  # 确保每个session都有条目

    with open(keyword_path, 'r', encoding='utf-8') as f:
        for line in f:
            batch = json.loads(line.strip())
            for sent in batch.get('sentence', []):
                session_num = get_session_num(sent.get('sentence_id', ''))
                if session_num in session_keywords:
                    session_keywords[session_num]['sentence'].append(sent)

    # 写入拆分后的keyword文件（与translate行数一致）
    os.makedirs(os.path.dirname(out_keyword_path), exist_ok=True)
    with open(out_keyword_path, 'w', encoding='utf-8') as f:
        for session_num in sorted(session_keywords.keys()):
            f.write(json.dumps(session_keywords[session_num], ensure_ascii=False) + '\n')

    logging.info(f"Split {len(session_data)} sessions to {out_translate_path} and {out_keyword_path}")


def get_raw_text_sample(sample):
    pat = re.compile(r"^session_(\d+)$")  # 匹配 session_1 / session_2 / ...
    raw_text : Dict[str, dict] = defaultdict(dict)
    sample_id = sample.get("sample_id")
    conversation = sample.get("conversation")

    # 收集并按 session 编号排序
    session_keys = []
    for k in conversation.keys():
        m = pat.match(k)
        if m and isinstance(conversation.get(k), list):
            session_keys.append((int(m.group(1)), k))
    session_keys.sort()
    for idx, k in session_keys:
        turns = conversation.get(k, [])
        session_time = conversation.get(f"{k}_date_time") or ""
        session_id = f"D{idx}"

        # 逐条 turn 展开
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            speaker = (turn.get("speaker") or "UNKNOWN").strip()
            dia_id = (turn.get("dia_id")).strip()
            text = (turn.get("text") or "").strip()
            if not text:
                continue
            if turn.get("blip_caption") != None:
                raw_text[session_id].update({dia_id: f"{speaker}:{text} and shared {turn.get('blip_caption')}"})
            else:
                raw_text[session_id].update( {dia_id: f"{speaker}:{text}"})
        return raw_text

def get_raw_text():
    input_path = Path("locomo10.json")
    data = json.loads(input_path.read_text(encoding="utf-8"))

    pat = re.compile(r"^session_(\d+)$")  # 匹配 session_1 / session_2 / ...

    raw_text : Dict[str, dict] = defaultdict(dict)

    for i, sample in enumerate(data):
        if not isinstance(sample, dict):
            continue

        sample_id = sample.get("sample_id") or f"sample_{i}"
        conversation = sample.get("conversation")
        if not isinstance(conversation, dict):
            continue

        # 收集并按 session 编号排序
        session_keys = []
        for k in conversation.keys():
            m = pat.match(k)
            if m and isinstance(conversation.get(k), list):
                session_keys.append((int(m.group(1)), k))
        session_keys.sort()
        for idx, k in session_keys:
            turns = conversation.get(k, [])
            session_time = conversation.get(f"{k}_date_time") or ""
            session_id = f"D{idx}"

            # 逐条 turn 展开
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                speaker = (turn.get("speaker") or "UNKNOWN").strip()
                dia_id = (turn.get("dia_id")).strip()
                text = (turn.get("text") or "").strip()
                if not text:
                    continue
                if turn.get("blip_caption") != None:
                    raw_text[session_id].update({dia_id: f"{speaker}:{text} and shared {turn.get('blip_caption')}"})
                else:
                    raw_text[session_id].update( {dia_id: f"{speaker}:{text}"})
        return raw_text


def delete():
    file_name = "result_event.json"
    with open(file_name, "w", encoding="utf-8") as f:
        pass

    file_name = "result_key.json"
    with open(file_name, "w", encoding="utf-8") as f:
        pass

    file_name = "result_translate.json"
    with open(file_name, "w", encoding="utf-8") as f:
        pass

import pickle
def get_conv_embeddings(embedding_path):
    database = pickle.load(open(embedding_path, 'rb'))
    embeddings = database.get("embeddings")
    sentence_id = database.get("sentence_id")
    topic_embeddings = database.get("topic")
    # print(topic_embeddings)
    topic_id = database.get("topic_list")
    question_embeddings = database.get("question_embeddings")
    id2emb = {i: embeddings[r] for r, i in enumerate(sentence_id)}
    tid2emb = {i: topic_embeddings[r] for r, i in enumerate(topic_id)}
    # print(topic_id)
    return id2emb, question_embeddings, topic_id, topic_embeddings



def calculate(question_embeddings,conv_embedding):
    database = pickle.load(open(FILE_EMBEDDING, 'rb'))
    embeddings = database.get("embeddings")
    sentence_id = database.get("sentence_id")
    for emb in question_embeddings:
        # print(emb)
        top_ids, _, top_embs, top_texts = topk_answers_by_similarity(emb, embeddings,sentence_id,5)
        # print(f'question {top_ids} {top_texts}')
        # print(top_embs[0])
        # break


# def embed_sample(conversation, qa_list, embedding_path):
#     # sample_id = sample.get("sample_id")
#     # conversation = sample.get("conversation")
#     pat = re.compile(r"^session_(\d+)$")
#     retriever = config.RETRIEVER_TYPE
#     session_keys = []
#     sample_embedding = []
#     sample_sentence_id = []
#     print(conversation)
#     for k in conversation.keys():
#         m = pat.match(k)
#         if m and isinstance(conversation.get(k), list):
#             session_keys.append((int(m.group(1)), k))
#     session_keys.sort()
#     for idx, k in session_keys:
#         turns = conversation.get(k)
#         session_time = conversation.get(f"{k}_date_time")
#         session_id = f"D{idx}"
#         session_embedding, session_sentence_ids = embed_session(retriever, turns, k, session_time)
#         sample_embedding.append(session_embedding)
#         sample_sentence_id.extend(session_sentence_ids)
#     sample_embedding = np.vstack(sample_embedding)
#     # qa_list = sample.get("qa")
#     question_embeddings = embed_question(retriever, qa_list)
#
#     database = {'embeddings': sample_embedding,
#                 # 'date_time': session_time,
#                 'sentence_id': sample_sentence_id,
#                 'question_embeddings': question_embeddings}
#
#     with open(embedding_path, 'wb') as f:
#         pickle.dump(database, f)


def process_single_sample(sample_id, sample, dataset, question_list, raw_text_list, worker_id):
    """
    处理单个样本的函数，供线程池调用
    每个线程独立创建自己的 LLM、MemorySystem 等实例
    """
    thread_name = threading.current_thread().name
    logging.info(f"[Worker-{worker_id}] [{thread_name}] Processing sample {sample_id}")

    # 时间统计变量
    time_stats = {
        "sample_id": sample_id,
        "worker_id": worker_id,
        "conversation_processing_time": 0.0,
        "question_answering_time": 0.0,
        "total_time": 0.0
    }
    sample_start_time = time.time()

    try:
        # 每个线程独立创建实例
        llm = LLM()
        memory_system = MemorySystem()
        memory_controller = MemoryController(memory_system, llm)
        agent = Agent(llm, memory_system, memory_controller)

        # 每个样本独立的输出文件路径（避免文件写入冲突）
        cost_path = config.cost_template.format(dataset=dataset, sample_id=sample_id)

        with open(cost_path, "a", encoding="utf-8") as cf:
            cost = {"prompt_tokens": agent.llm.prompt_tokens, "completion_tokens": agent.llm.completion_tokens,
                    "position": 0, "sample": sample_id, "worker_id": worker_id}
            cf.write(json.dumps(cost, ensure_ascii=False, default=list) + "\n")

        with per_sample_log(sample_id=sample_id, dataset=dataset):
            logging.info(f"[Worker-{worker_id}] === Start processing sample {sample_id} ===")

            # ============ 开始计时：处理对话 ============
            conv_start_time = time.time()

            translate_path = config.translate_template.format(dataset=dataset, sample_id=sample_id)
            if not os.path.exists(translate_path):
                agent.translate_sample(sample, translate_path)
            else:
                logging.info(f"[Worker-{worker_id}] Translation for sample {sample_id} already exists, skipping.")

            keyword_path = config.keyword_template.format(dataset=dataset, sample_id=sample_id)
            keyword_path_t = config.keyword_template_t.format(dataset=dataset, sample_id=sample_id)
            if not os.path.exists(keyword_path):
                agent.extract_keyword_sample(keyword_path, translate_path)
            else:
                logging.info(f"[Worker-{worker_id}] Keyword for sample {sample_id} already exists, skipping.")

            # 检查是否需要拆分batch格式到单session格式
            need_split = False
            if os.path.exists(translate_path):
                with open(translate_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        first_key = list(json.loads(first_line).keys())[0]
                        if '-' in first_key and 'session_' in first_key:
                            need_split = True

            if need_split:
                split_batch_to_sessions(
                    translate_path=translate_path,
                    keyword_path=keyword_path,
                    out_translate_path=translate_path,
                    out_keyword_path=keyword_path,
                    batch_size=10
                )
            else:
                logging.info(f"[Worker-{worker_id}] Files for sample {sample_id} already in single-session format, skipping split.")

            # if not os.path.exists(keyword_path_t):
            #     transform_keyword(keyword_path, keyword_path_t)
            #     # pass
            # else:
            #     logging.info(f"[Worker-{worker_id}] Keyword_t for sample {sample_id} already exists, skipping.")

            embedding_path = config.embedding_template.format(dataset=dataset, sample_id=sample_id)
            if not os.path.exists(embedding_path):
                embed_sample(question_list[sample_id], translate_path, embedding_path)
            else:
                logging.info(f"[Worker-{worker_id}] Embedding for sample {sample_id} already exists, skipping.")

            raw_text = raw_text_list[sample_id]

            question_embeddings = None
            if config.USE_EMBEDDING:
                conv_embeddings, question_embeddings, topic_id_list, topic_embeddings = get_conv_embeddings(embedding_path)
                agent.store_raw_text(raw_text, conv_embeddings, topic_id_list, topic_embeddings)
            else:
                agent.store_raw_text(raw_text, topic_id_list, topic_embeddings)

            # ============ 结束计时：处理对话 ============
            conv_end_time = time.time()
            time_stats["conversation_processing_time"] = conv_end_time - conv_start_time
            logging.info(f"[Worker-{worker_id}] [TIME] Sample {sample_id} conversation processing time: {time_stats['conversation_processing_time']:.2f}s")

            result_path = config.result_template.format(dataset=dataset, sample_id=sample_id)
            evidence_path = config.evidence_template.format(dataset=dataset, sample_id=sample_id)
            analysis_path = config.analysis_template.format(dataset=dataset, sample_id=sample_id)
            cost_path_qa = config.cost_template.format(dataset=dataset, sample_id=sample_id)

            with open(cost_path, "a", encoding="utf-8") as cf1:
                cost = {"prompt_tokens": agent.llm.prompt_tokens, "completion_tokens": agent.llm.completion_tokens,
                        "position": 1, "sample": sample_id, "worker_id": worker_id}
                cf1.write(json.dumps(cost, ensure_ascii=False, default=list) + "\n")

            if not os.path.exists(result_path):
                agent.store_keyword(keyword_path, translate_path)

                # ============ 开始计时：回答问题 ============
                qa_start_time = time.time()
                get_question(dataset, agent, question_list, sample_id, memory_system, result_path, evidence_path, analysis_path, cost_path_qa, question_embeddings)
                qa_end_time = time.time()
                time_stats["question_answering_time"] = qa_end_time - qa_start_time
                logging.info(f"[Worker-{worker_id}] [TIME] Sample {sample_id} question answering time: {time_stats['question_answering_time']:.2f}s")
                # ============ 结束计时：回答问题 ============
            else:
                logging.info(f"[Worker-{worker_id}] Result for sample {sample_id} already exists, skipping.")

        # 计算总时间
        sample_end_time = time.time()
        time_stats["total_time"] = sample_end_time - sample_start_time
        logging.info(f"[Worker-{worker_id}] [TIME] Sample {sample_id} total time: {time_stats['total_time']:.2f}s")

        # 保存时间统计到文件
        time_stats_path = config.cost_template.format(dataset=dataset, sample_id=sample_id).replace("cost", "time_stats")
        os.makedirs(os.path.dirname(time_stats_path), exist_ok=True)
        with open(time_stats_path, "a", encoding="utf-8") as tf:
            tf.write(json.dumps(time_stats, ensure_ascii=False) + "\n")

        logging.info(f"[Worker-{worker_id}] === Finished processing sample {sample_id} ===")
        return {"sample_id": sample_id, "status": "success", "worker_id": worker_id, "time_stats": time_stats}

    except Exception as e:
        logging.error(f"[Worker-{worker_id}] Error processing sample {sample_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        # 记录到出错时为止的时间
        sample_end_time = time.time()
        time_stats["total_time"] = sample_end_time - sample_start_time
        time_stats["error"] = str(e)
        return {"sample_id": sample_id, "status": "error", "error": str(e), "worker_id": worker_id, "time_stats": time_stats}


def filter_samples_by_category(question_list, target_category):
    """
    根据 question 的 category 过滤样本

    Args:
        question_list: 问题列表字典 {sample_id: [questions]}
        target_category: 目标 category 字符串，如 "single-session-user"

    Returns:
        符合条件的 sample_id 列表
    """
    filtered_sample_ids = []
    for sample_id, qa_list in question_list.items():
        if qa_list and len(qa_list) > 0:
            # 取第一个问题的 category
            category = qa_list[0].get("category")
            if category == target_category:
                filtered_sample_ids.append(sample_id)
    return filtered_sample_ids


def distribute_samples_round_robin(sample_ids, num_workers):
    """
    轮询方式将样本分配给各个 worker

    Args:
        sample_ids: 样本ID列表
        num_workers: worker 数量

    Returns:
        分配结果字典 {worker_id: [sample_ids]}
    """
    distribution = {i: [] for i in range(num_workers)}
    for idx, sample_id in enumerate(sample_ids):
        worker_id = idx % num_workers
        distribution[worker_id].append(sample_id)
    return distribution


def main():
    """
    主函数 - 支持多线程并行处理

    命令行参数（通过 config 获取）：
    - --workers: 线程数（默认 4）
    - --start_sample: 起始 sample_id（用于断点续传）
    - --target_category: 目标 category（默认 "single-session-user"）
    """
    # 从 config 或命令行参数获取配置
    dataset = config.dataset
    datapath = config.datapath

    # 新增参数（需要在 config.py 中添加或直接使用默认值）
    num_workers = getattr(config, 'num_workers', 4)
    start_sample = getattr(config, 'start_sample', None)

    # category 映射
    category_dict = {0: "multi-session", 1: "single-session-user", 2: "temporal-reasoning"}

    # 优先使用 --target_category，否则使用 --ca 对应的 category
    target_category = getattr(config, 'target_category', None)
    if target_category is None:
        # 如果没有指定 target_category，使用 ca 参数对应的 category
        if hasattr(config, 'ca') and config.ca in category_dict:
            target_category = category_dict[config.ca]
        else:
            target_category = "single-session-user"  # 默认值

    logging.info(f"=== Configuration ===")
    logging.info(f"Dataset: {dataset}")
    logging.info(f"Target category: {target_category}")
    logging.info(f"Number of workers: {num_workers}")
    logging.info(f"Start sample: {start_sample}")
    logging.info(f"Max samples: {getattr(config, 'max_samples', None)}")

    # 加载数据
    conversation_list, question_list, raw_conversation_list, raw_text_list = get_data(dataset, datapath)

    # 根据 category 过滤样本
    if dataset == "LM":
        filtered_sample_ids = filter_samples_by_category(question_list, target_category)
        logging.info(f"Filtered {len(filtered_sample_ids)} samples with category '{target_category}'")
    else:
        # 对于其他数据集，使用所有样本
        filtered_sample_ids = list(conversation_list.keys())
        logging.info(f"Using all {len(filtered_sample_ids)} samples (dataset: {dataset})")

    # 断点续传：跳过已处理的样本
    if start_sample:
        try:
            start_idx = filtered_sample_ids.index(start_sample)
            filtered_sample_ids = filtered_sample_ids[start_idx:]
            logging.info(f"Resuming from sample {start_sample}, {len(filtered_sample_ids)} samples remaining")
        except ValueError:
            logging.warning(f"Start sample {start_sample} not found, processing all filtered samples")

    # 进一步过滤：跳过已存在结果文件的样本（断点续传核心逻辑）
    samples_to_process = []
    for sample_id in filtered_sample_ids:
        result_path = config.result_template.format(dataset=dataset, sample_id=sample_id)
        if not os.path.exists(result_path):
            samples_to_process.append(sample_id)
        else:
            logging.info(f"Skipping sample {sample_id} - result file already exists")

    logging.info(f"Samples to process after resume check: {len(samples_to_process)}")

    # 应用最大样本数量限制
    max_samples = getattr(config, 'max_samples', None)
    if max_samples is not None and max_samples > 0:
        original_count = len(samples_to_process)
        samples_to_process = samples_to_process[:max_samples]
        logging.info(f"Applied max_samples limit: {max_samples}, reduced from {original_count} to {len(samples_to_process)} samples")

    if not samples_to_process:
        logging.info("No samples to process. All done!")
        return

    # 轮询方式分配样本给各个 worker
    distribution = distribute_samples_round_robin(samples_to_process, num_workers)
    for worker_id, assigned_samples in distribution.items():
        logging.info(f"Worker-{worker_id}: {len(assigned_samples)} samples assigned")

    # 使用 ThreadPoolExecutor 并行处理
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}

        # 提交任务
        for sample_id in samples_to_process:
            # 计算该样本分配给哪个 worker
            worker_id = samples_to_process.index(sample_id) % num_workers
            sample = conversation_list[sample_id]

            future = executor.submit(
                process_single_sample,
                sample_id,
                sample,
                dataset,
                question_list,
                raw_text_list,
                worker_id
            )
            futures[future] = sample_id

        # 收集结果
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = result.get("status", "unknown")
                if status == "success":
                    logging.info(f"Sample {sample_id} completed successfully")
                else:
                    logging.error(f"Sample {sample_id} failed: {result.get('error', 'unknown error')}")
            except Exception as e:
                logging.error(f"Sample {sample_id} raised exception: {str(e)}")
                results.append({"sample_id": sample_id, "status": "exception", "error": str(e)})

    # 统计结果
    success_count = sum(1 for r in results if r.get("status") == "success")
    error_count = len(results) - success_count
    logging.info(f"=== Processing complete ===")
    logging.info(f"Total: {len(results)}, Success: {success_count}, Errors: {error_count}")

    # 汇总时间统计
    logging.info(f"=== Time Statistics Summary ===")
    all_time_stats = []
    total_conv_time = 0.0
    total_qa_time = 0.0
    total_sample_time = 0.0
    for r in results:
        ts = r.get("time_stats")
        if ts:
            all_time_stats.append(ts)
            total_conv_time += ts.get("conversation_processing_time", 0.0)
            total_qa_time += ts.get("question_answering_time", 0.0)
            total_sample_time += ts.get("total_time", 0.0)

    if all_time_stats:
        avg_conv_time = total_conv_time / len(all_time_stats)
        avg_qa_time = total_qa_time / len(all_time_stats)
        avg_total_time = total_sample_time / len(all_time_stats)
        logging.info(f"Average conversation processing time per sample: {avg_conv_time:.2f}s")
        logging.info(f"Average question answering time per sample: {avg_qa_time:.2f}s")
        logging.info(f"Average total time per sample: {avg_total_time:.2f}s")
        logging.info(f"Total conversation processing time: {total_conv_time:.2f}s")
        logging.info(f"Total question answering time: {total_qa_time:.2f}s")
        logging.info(f"Total processing time (all samples): {total_sample_time:.2f}s")

        # 打印每个样本的时间详情
        logging.info(f"=== Per-Sample Time Details ===")
        for ts in all_time_stats:
            logging.info(f"Sample {ts['sample_id']}: conv={ts['conversation_processing_time']:.2f}s, qa={ts['question_answering_time']:.2f}s, total={ts['total_time']:.2f}s")

    # 保存处理结果摘要
    summary_path = f"result/{dataset}/processing_summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(results),
            "success": success_count,
            "errors": error_count,
            "target_category": target_category,
            "time_summary": {
                "avg_conversation_processing_time": avg_conv_time if all_time_stats else 0,
                "avg_question_answering_time": avg_qa_time if all_time_stats else 0,
                "avg_total_time": avg_total_time if all_time_stats else 0,
                "total_conversation_processing_time": total_conv_time,
                "total_question_answering_time": total_qa_time,
                "total_processing_time": total_sample_time
            },
            "per_sample_time_stats": all_time_stats,
            "results": results
        }, f, ensure_ascii=False, indent=2)



def log_config(config_module, exclude=("API_KEY","OPENROUTER_URL")):
    logging.info("========== CONFIGURATION ==========")
    for name in dir(config_module):
        if not re.match(r'^[A-Z0-9_]+$', name):
            continue
        if any(kw in name.lower() for kw in ["key", "url", "secret", "password"]):
            # logging.info(f"{name} = [HIDDEN]")
            continue
        value = getattr(config_module, name)
        logging.info(f"{name} = {value}")
    logging.info("===================================")


# logging_utils.py
import os
import logging
from contextlib import contextmanager

import argparse, sys
_orig = argparse.ArgumentParser.parse_args
def _logged(self, *a, **k):
    print(f"[ARGPARSE] parse_args on id={id(self)} argv={sys.argv[1:]}")
    print("[ARGPARSE] actions:", [ac.option_strings or [ac.dest] for ac in self._actions])
    return _orig(self, *a, **k)



if __name__ == "__main__":
    # 初始化日志
    global_file_handler = logging.FileHandler(
        f"log/run_{config.DATASET}{config.ADDITIONAL_TK}{config.ADDITIONAL_RE}.log",
        encoding="utf-8"
    )
    stream_handler = logging.StreamHandler()

    logging.basicConfig(
        level=logging.INFO,  # 输出 INFO 及以上级别的日志
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[global_file_handler, stream_handler]
    )

    logging.info("=== Program start ===")
    log_config(config)
    # ⚠️ 关键：写完配置后，立刻把“总日志文件”handler卸掉
    root_logger = logging.getLogger()
    root_logger.removeHandler(global_file_handler)
    global_file_handler.close()
    argparse.ArgumentParser.parse_args = _logged
    main()
