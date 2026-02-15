import json, re, pickle
from rag_utils import get_embeddings
import numpy as np
from pathlib import Path

FILE_EMBEDDING = "conversation_embedding.pkl"
RETRIEVER_TYPE = "dragon"
def embed_session(retriever_type, sentence_list, session_id, session_time):
    conversation = []
    sentence_id_list = []
    for sentence in sentence_list:
        conversation.append(sentence['speaker'] + ' said, \"' + sentence['text'] + '\"')
        sentence_id_list.append(sentence['dia_id'])
    # print(len(conversation))
    embeddings = get_embeddings(retriever_type, conversation, "context")
    assert embeddings.shape[0] == len(conversation), "Lengths of embeddings and dialogs do not match"
    return embeddings, sentence_id_list


def embed_question(retriever_type, qa_list):
    question_list = []
    for qa in qa_list:
        question = qa.get("question")
        question_list.append(question)
    embeddings = get_embeddings(retriever_type, question_list, 'query')
    return embeddings



if __name__ == '__main__':
    input_path = Path("locomo10.json")
    data = json.loads(input_path.read_text(encoding="utf-8"))

    pat = re.compile(r"^session_(\d+)$")  # 匹配 session_1 / session_2 / ...
    retriever = RETRIEVER_TYPE

    for i, sample in enumerate(data):
        sample_id = sample.get("sample_id")
        conversation = sample.get("conversation")
        if not isinstance(conversation, dict):
            continue

        # 收集并按 session 编号排序
        session_keys = []
        sample_embedding = []
        sample_sentence_id = []
        for k in conversation.keys():
            m = pat.match(k)
            if m and isinstance(conversation.get(k), list):
                session_keys.append((int(m.group(1)), k))
        session_keys.sort()
        for idx, k in session_keys:
            turns = conversation.get(k)
            session_time = conversation.get(f"{k}_date_time")
            session_id = f"D{idx}"
            session_embedding, session_sentence_ids = embed_session(retriever, turns, k, session_time)
            sample_embedding.append(session_embedding)
            sample_sentence_id.extend(session_sentence_ids)
        sample_embedding = np.vstack(sample_embedding)
        qa_list = sample.get("qa")
        question_embeddings = embed_question(retriever, qa_list)

        database = {'embeddings': sample_embedding,
                    # 'date_time': session_time,
                    'sentence_id': sample_sentence_id,
                    'question_embeddings': question_embeddings}

        with open(FILE_EMBEDDING, 'wb') as f:
            pickle.dump(database, f)
        break


