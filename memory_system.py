import os
import json
import time
import uuid
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from openai import APIStatusError, APIConnectionError, APIResponseValidationError
import traceback
from openai import OpenAI
import requests
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import re
from typing import List, Dict, Any, Set
from collections import defaultdict
from config import USE_EMBEDDING
import config


def _norm_key(s: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in s).split())


def _simple_tokens(text: str) -> List[str]:
    return _norm_key(text).split()


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


class KeyNode:
    def __init__(self, key_id: str):
        self.key_id = key_id
        self.text = key_id
        self.tag_list = []
        self.tag_dict = {}

    def add_tag(self, tag, episode_id):
        if tag not in self.tag_list:
            self.tag_list.append(tag)
        if tag not in self.tag_dict:
            self.tag_dict[tag] = []
        self.tag_dict[tag].append(episode_id)

    def get_tag_link(self, tag):
        if tag not in self.tag_dict:
            return []
        return self.tag_dict[tag]

    def add_episode_tag(self, tag, episode_id):
        self.tag_dict[tag] = episode_id

    def get_tag_list(self):
        return self.tag_list


class SemanticEvent:
    def __init__(self, event_id: str, text: str, origin: str):
        self.event_id = event_id
        self.text = text
        self.tag_list = []
        self.tag_dict = {}
        self.origin = origin

    def add_tag(self, tag, semantic_id):
        self.tag_list.append(tag)
        if tag not in self.tag_dict:
            self.tag_dict[tag] = []
        self.tag_dict[tag].append(semantic_id)


class Topic:
    def __init__(self, topic_id: str, text: str):
        self.topic_id = topic_id
        self.text = text
        self.event_list = []


class PersonalEvent:
    def __init__(self, person: str, id: str, text: str, tag:str, origin: str):
        self.person = person
        self.personal_id = id
        self.text = text
        self.tag = tag
        self.origin = origin


class Persona:
    def __init__(self, person: str):
        self.person = person
        self.persona_dict: Dict[str, PersonalEvent] = defaultdict()
        self.tag_list = []
        self.tag_dict: Dict[str, List[PersonalEvent]] = defaultdict(list)

    def add_information(self, id: str, text: str, tag:str, origin: str):
        pe = PersonalEvent(self.person, id, text, tag, origin)
        self.persona_dict[id] = pe
        if tag not in self.tag_list:
            self.tag_list.append(tag)
        self.tag_dict[tag].append(pe)

    def get_tag_text(self, tag):
        pe_list = self.tag_dict[tag]
        text_list = []
        id_list = []
        for pe in pe_list:
            text_list.append(pe.origin+":"+pe.text)
            id_list.append(pe.origin)
        return text_list, id_list


class EpisodeEvent:
    def __init__(self, event_id: str, text: str, origin: str, embedding = None, time: str = None, conv_time: str = None, true_time: str = None, topic_list=None):
        self.event_id = event_id
        self.text = text
        self.time = time
        self.true_time = true_time
        self.tag_t = ""
        self.tag_list = []
        self.tag_dict = {}
        self.origin = origin
        self.embedding = embedding
        self.conversation_time = conv_time
        session_id = event_id.split(":")[0]
        self.topic_list = []
        for t in topic_list:
            self.topic_list.append(f"D{session_id}:"+t)

        # self.topic_list = topic_list

    def add_tag(self, tag, episode_id):
        self.tag_list.append(tag)
        if tag not in self.tag_dict:
            self.tag_dict[tag] = []
        self.tag_dict[tag].append(episode_id)


class Link:
    def __init__(self, key_id: str, event_id: str, event_type: str, tag: str):
        self.key_id = key_id
        self.event_id = event_id
        self.event_type = event_type
        self.tag = tag


class EventLink:
    def __init__(self, semantic_id: str, episode_id: list):
        self.semantic_id = semantic_id
        self.episode_id = episode_id

from datetime import datetime
class MemorySystem:
    def __init__(self):
        self.keys: Dict[str, KeyNode] = {}  # key -> meta {'aliases': set(), ...}
        self.semantic_events: Dict[str, SemanticEvent] = {}
        self.episode_events: Dict[str, EpisodeEvent] = {}
        self.semantic_links: Dict[str, Link] = {}  # id -> Edge
        self.episode_links: Dict[str, Link] = {}
        self.tag_list: List[str] = []
        # self.key_tags: Dict[str, List] = {}
        self.key_to_values: Dict[str, Set[tuple]] = defaultdict(set)
        self.key_to_semantic: Dict[str, Set[str]] = defaultdict(set)
        self.event_to_keys: Dict[str, Set[str]] = defaultdict(set)

        self.by_tag: Dict[str, List[str]] = {}  # tag -> [edge_id]
        self.by_key: Dict[str, List[str]] = {}  # key -> [edge_id]
        self.timeline: Dict[datetime, List[str]] = {} # -> event_id
        self.topic_to_event: Dict[str, List[str]] = {}
        self.persona_list: Dict[str, Persona] = {}
        self.topic_id_list: List[str] = []
        self.topic_embeddings = []
        self.topic_sentence_list = []

        self.semantic_by_tag: Dict[str, List[str]] = {}  # tag -> [sid]
        self.context_link: Dict[str, Set[str]] = defaultdict(set)
        self.raw_text: Dict[str,Dict[str,str]] = {}
        self.embeddings = {}
        self.semantic_number = 0
        self.episode_number = 0
        self.topic_dict: Dict[str, Topic] = {}
        self.tid2emb = {}

    # ----- Node ops -----

    def add_event_time(self, event_id, absolute_time):
        absolute_time = self._to_date(absolute_time)
        if config.USE_TIME:
            if absolute_time in self.timeline:
                self.timeline[absolute_time].append(event_id)
            else:
                self.timeline[absolute_time] = [event_id]

    def get_event_time(self, before_time=None, absolute_time=None):
        if before_time is not None:
            keys = sorted(k for k in self.timeline if k < before_time)
            return [ev for k in keys for ev in self.timeline[k]]

        if absolute_time is not None:
            return self.timeline[absolute_time]

        return []

    def get_time_event(self, start_date, end_date, include_date: bool = False):
        print(start_date)
        print(end_date)
        start_date = self._to_date(start_date) if start_date is not None else None
        end_date = self._to_date(end_date) if end_date is not None else None

        keys = sorted(self.timeline.keys())
        print(keys)
        take = lambda d: (start_date is None or d >= start_date) and (end_date is None or d <= end_date)

        if include_date:
            return [(d, eid) for d in keys if take(d) for eid in self.timeline[d]]
        else:
            return [eid for d in keys if take(d) for eid in self.timeline[d]]

    # add near the top of file
    from datetime import datetime, date
    def _years_from_timeline_keys(self):
        """从 timeline 的 key 提取出现过的年份集合"""
        from datetime import datetime, date
        keys = self.timeline.keys()
        yrs = set()
        for k in keys:
            if isinstance(k, date):
                yrs.add(k.year)
            elif isinstance(k, datetime):
                yrs.add(k.year)
            elif isinstance(k, str):
                try:
                    d = date.fromisoformat(k)
                    yrs.add(d.year)
                except Exception:
                    # 如果 timeline 里还有其它格式，可在这里扩展
                    pass
        return sorted(yrs)

    def _to_date(self, x):
        from datetime import datetime, date
        import re

        # 先处理 datetime，再处理 date（datetime 是 date 的子类）
        if isinstance(x, datetime):
            return x.date()
        if isinstance(x, date):
            return x
        if isinstance(x, (int, float)):  # unix timestamp (秒)
            return datetime.fromtimestamp(x).date()

        if isinstance(x, str):
            s = x.strip()

            # (A) 规范化含 00 的 ISO 风格：YYYY-00-00 / YYYY-00-DD / YYYY-MM-00
            m00 = re.fullmatch(r'(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})', s)
            if m00:
                y = int(m00.group('y'))
                mm = m00.group('m')
                dd = m00.group('d')
                if mm == '00':  # 月未知 → 01，日若未知也置 01
                    mm, dd = '01', '01'
                elif dd == '00':  # 日未知 → 01
                    dd = '01'
                s_norm = f'{y}-{mm}-{dd}'
                try:
                    return date.fromisoformat(s_norm)
                except Exception:
                    # 若仍非法（如 1963-02-31），退化为当月 1 号
                    mm_int = max(1, min(12, int(mm) if mm.isdigit() else 1))
                    return date(y, mm_int, 1)

            # (B) 直接尝试 ISO-8601 单日：YYYY-MM-DD
            try:
                return date.fromisoformat(s)
            except Exception:
                pass

            # (C) 年份区间：1914-1918 / 1914–1918 / 1914—1918
            m = re.fullmatch(r'(\d{4})\s*[–—-]\s*(\d{4})', s)
            if m:
                y1, y2 = int(m.group(1)), int(m.group(2))
                if y1 > y2:
                    y1, y2 = y2, y1
                start = date(y1, 1, 1)
                end = date(y2, 12, 31)
                return start + (end - start) // 2

            # (D) 日期区间：YYYY-MM-DD <sep> YYYY-MM-DD，sep 支持 to / ~ / – / —
            sep_pat = r'\s*(?:to|~|–|—)\s*'
            dm = re.fullmatch(r'(\d{4}-\d{2}-\d{2})' + sep_pat + r'(\d{4}-\d{2}-\d{2})', s)
            if dm:
                try:
                    d1 = date.fromisoformat(dm.group(1))
                    d2 = date.fromisoformat(dm.group(2))
                    if d1 > d2:
                        d1, d2 = d2, d1
                    return d1 + (d2 - d1) // 2
                except Exception:
                    pass

            # (E) 季度格式：YYYY-Q1/Q2/Q3/Q4
            qm = re.fullmatch(r'(\d{4})-Q([1-4])', s)
            if qm:
                y = int(qm.group(1))
                q = int(qm.group(2))
                # Q1=1月, Q2=4月, Q3=7月, Q4=10月，取季度中间月份的15日
                month = (q - 1) * 3 + 2  # Q1->2, Q2->5, Q3->8, Q4->11
                return date(y, month, 15)

            # (F) 简写年份格式：YY-MM-DD (如 23-03-19 -> 2023-03-19)
            ym = re.fullmatch(r'(\d{2})-(\d{2})-(\d{2})', s)
            if ym:
                yy = int(ym.group(1))
                mm = int(ym.group(2))
                dd = int(ym.group(3))
                # 假设 00-99 对应 2000-2099
                year = 2000 + yy if yy < 100 else yy
                try:
                    return date(year, mm, dd)
                except Exception:
                    return date(year, max(1, min(12, mm)), 1)

            # (G) 占位符格式：YYYY-MM-DD / YYYY-MM-XX / YYYY-XX-XX (含字母占位符)
            pm = re.fullmatch(r'(\d{4})-([A-Za-z0-9]{2})-([A-Za-z0-9]{2})', s)
            if pm:
                y = int(pm.group(1))
                mm_str = pm.group(2)
                dd_str = pm.group(3)
                # 如果月份是占位符（含字母），默认1月
                if not mm_str.isdigit():
                    return date(y, 1, 1)
                mm = int(mm_str)
                mm = max(1, min(12, mm))
                # 如果日期是占位符（含字母），默认1日
                if not dd_str.isdigit():
                    return date(y, mm, 1)
                dd = int(dd_str)
                try:
                    return date(y, mm, dd)
                except Exception:
                    return date(y, mm, 1)

            # (H) 其它常见格式
            for fmt in ("%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
                try:
                    return datetime.strptime(s, fmt).date()
                except Exception:
                    pass

        raise TypeError(f"Unsupported date-like type: {type(x)} -> {x!r}")

    def store_raw_text(self, raw_text, conv_embeddings=None, topic_id_list=None, topic_embeddings=None):
        self.raw_text = raw_text
        if USE_EMBEDDING:
            self.embeddings = conv_embeddings
            self.topic_id_list = topic_id_list
            self.topic_embeddings = topic_embeddings



    def get_tag_list(self, key_id):
        if key_id not in self.keys:
            return []
        return self.keys[key_id].get_tag_list()

    def get_all_tag(self):
        return self.tag_list

    def add_tag(self, tag, eid, key_id):
        if tag not in self.tag_list:
            self.tag_list.append(tag)
            self.by_tag[tag]=[eid]
        else:
            self.by_tag[tag].append(eid)
        if eid in self.event_to_keys:
            self.event_to_keys[eid].add(key_id)
        else:
            self.event_to_keys[eid] = {key_id}
        # print(self.event_to_keys[eid])

    def add_context_link(self, tarevent, souevent):

        if tarevent in self.context_link:
            self.context_link[tarevent].add(souevent)
        else:
            self.context_link[tarevent] = {souevent}

        if souevent in self.context_link:
            self.context_link[souevent].add(tarevent)
        else:
            self.context_link[souevent] = {tarevent}

    def add_topics(self, topic_sentences, eid_topic_dict, session_id):

        for ts in topic_sentences:
            tid = f"D{session_id}:"+ts
            ttext = topic_sentences[ts]
            topic = Topic(tid, ttext)
            self.topic_dict[tid] = topic
            self.topic_sentence_list.append(tid + ":" +ttext)
            print(self.topic_id_list)
            print(self.topic_sentence_list)
            print(tid)

            assert self.topic_id_list.index(tid) == self.topic_sentence_list.index(tid + ":" +ttext)


        # for i in range(len)

        for eid, topics in eid_topic_dict.items():
            for tid in topics:
                tid = f"D{session_id}:" + tid
                self.topic_dict.get(tid).event_list.append(eid)

        # for tid in self.topic_id_list:
        #     # tid = f"D{session_id}:" + tid
        #     self.topic_id_list.append(tid)
        #     self.topic_sentence_list.append(tid + ":" + )
        #
        # self.topic_embeddings = topic_embeddings


    def add_personal_information(self,pid,ptext,porigin,ptag,person):
        if person  in self.persona_list:
            pe = self.persona_list[person]
            pe.add_information(pid, ptext, ptag, porigin)
        else:
            pe = Persona(person)
            self.persona_list[person] = pe
            pe.add_information(pid,ptext,ptag,porigin)



    def event_by_tag(self, key: str, tag: str):
        if key not in self.keys:
            return ""
        links = self.keys[key].get_tag_link(tag)
        # print(links)
        text = []
        origin = []
        event_ids = []
        # print(self.episode_links)
        for link in links:
            # event_id = self.episode_links[link].event_id
            episode_event = self.episode_events[link]
            text.append(episode_event.event_id + ":" + episode_event.text)
            # print(episode_event.origin)
            origin.append(episode_event.origin)
            event_ids.append(episode_event.event_id)
        # print(f'event by tag{origin}')
        return text, origin, event_ids

    def query_conversation_time(self, event_id):
        pattern = re.compile(r'^(D\d+):(\d+)$')
        if pattern.match(event_id):
            return self.episode_events[event_id+"-1"].conversation_time
        return self.episode_events[event_id].conversation_time

    def query_event_keywords(self, event_id):
        keys = self.event_to_keys[event_id]
        key_candidates = []
        for k in keys:
            tag = self.get_tag_list(k)
            key_candidates.append({"key": k, "tags": tag})
        return key_candidates

    def query_personal_information(self, person):
        return {"person":person, "aspects":self.persona_list[person].tag_list}

    def query_personal_aspect(self, person, aspect):
        persona = self.persona_list[person]
        text_list, origin_list = persona.get_tag_text(aspect)
        return text_list, origin_list


    def query_event_context(self, event_id):
        context_id = self.context_link[event_id]
        event_text = []
        EVENT_RE = re.compile(r"^s\d+$")
        # EVENT_E_RE = re.compile(r"^D\d+$")
        origin_list = []
        pattern = re.compile(r'^(D\d+):(\d+)$')
        if not EVENT_RE.match(event_id):

            if pattern.match(event_id):
                event_origin = event_id
            else:
                event_origin = self.episode_events[event_id].origin

            m = pattern.match(event_origin)
            # if not m:
            #     raise ValueError(f"格式不对：{event_origin}，应为 'D<number>:<number>'")
            prefix, n = m.group(1), int(m.group(2))
            prev_n = n - 1
            next_n = n + 1
            prev_id = f"{prefix}:{prev_n}" if prev_n >= 1 else None
            next_id = f"{prefix}:{next_n}"
            if prev_id:
                event_text.append(prev_id + ":" + self.raw_text[f"{prefix}"].get(prev_id))
                origin_list.append(prev_id)
            event_text.append(event_origin + ":" + self.raw_text[f"{prefix}"].get(event_origin))
            origin_list.append(event_origin)
            if next_id in self.raw_text[f"{prefix}"]:
                event_text.append(next_id + ":" +self.raw_text[f"{prefix}"].get(next_id))
                origin_list.append(next_id)

        for event in context_id:
            if EVENT_RE.match(event):
                event_text.append(self.semantic_events[event].text)
                origin_list.append(self.semantic_events[event].origin)
            else:
                event_text.append(event+":"+self.episode_events[event].text)
                origin_list.append(self.episode_events[event].origin)
        return json.dumps(event_text, ensure_ascii=False), origin_list


    def query_semantic_information(self, key_id):
        semantic_ids = self.key_to_semantic[key_id]
        event_text = []
        origin_list = []
        for id in semantic_ids:
            event = self.semantic_events[id].text
            e_origin = self.semantic_events[event].origin
            if e_origin in origin_list:
                continue
            event_text.append(event)
            origin_list.append(self.semantic_events[event].origin)
        return json.dumps(event_text, ensure_ascii=False), origin_list

    def query_topic_events(self, topic_id):

        event_text = []
        events = self.topic_dict[topic_id].event_list
        origin_list = []
        for e in events:
            event = self.episode_events[e]
            e_origin = event.origin
            if e_origin in origin_list:
                continue
            event_text.append(event.event_id + ":" + event.text)
            origin_list.append(event.origin)
        return json.dumps(event_text, ensure_ascii=False), origin_list



    def query_semantic(self, key: str, tag: str):
        links = self.keys[key].get_tag_link(tag)
        text = ""
        for link in links:
            if link in self.semantic_links:
                event_id = self.semantic_links[link].event_id
                semantic_event = self.semantic_events[event_id]
                text = text + semantic_event.text
        return text

    def get_support_origin(self, evidence_support):
        support_origin = []
        if evidence_support:
            for e in evidence_support:
                if e in self.episode_events:
                    support_origin.append(self.episode_events[e].origin)
                else:
                    support_origin.append(e)
        return support_origin

