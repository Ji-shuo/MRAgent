import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["DISABLE_TORCHVISION"] = "1"
import time
import os, json
import torch
from tqdm import tqdm
from emb_openai import get_openai_embedding, set_openai_key



def save_eval(data_file, accs, key='exact_match'):

    
    if os.path.exists(data_file.replace('.json', '_scores.json')):
        data = json.load(open(data_file.replace('.json', '_scores.json')))
    else:
        data = json.load(open(data_file))

    assert len(data['qa']) == len(accs), (len(data['qa']), len(accs), accs)
    for i in range(0, len(data['qa'])):
        data['qa'][i][key] = accs[i]
    
    with open(data_file.replace('.json', '_scores.json'), 'w') as f:
        json.dump(data, f, indent=2)


# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def init_context_model(retriever):

    if retriever == 'dpr':
        from transformers import DPRConfig, DPRContextEncoder, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
        context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        context_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").cuda()
        context_model.eval()
        return context_tokenizer, context_model

    elif retriever == 'contriever':

        from transformers import AutoTokenizer, AutoModel
        context_tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        context_model = AutoModel.from_pretrained('facebook/contriever').cuda()
        context_model.eval()
        return context_tokenizer, context_model

    elif retriever == 'dragon':

        from transformers import AutoTokenizer, AutoModel
        context_tokenizer = AutoTokenizer.from_pretrained('facebook/dragon-plus-query-encoder')
        context_model = AutoModel.from_pretrained('facebook/dragon-plus-context-encoder').cuda()
        return context_tokenizer, context_model

    elif retriever == 'openai':

        set_openai_key()
        return None, None
    
    else:
        raise ValueError
    
def init_query_model(retriever):

    if retriever == 'dpr':
        from transformers import DPRConfig, DPRContextEncoder, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
        question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        question_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").cuda()
        question_model.eval()
        return question_tokenizer, question_model

    elif retriever == 'contriever':

        from transformers import AutoTokenizer, AutoModel
        question_tokenizer = context_tokenizer
        question_model = AutoModel.from_pretrained('facebook/contriever').cuda()
        question_model.eval()
        return question_tokenizer, question_model

    elif retriever == 'dragon':

        from transformers import AutoTokenizer, AutoModel
        context_tokenizer = AutoTokenizer.from_pretrained('facebook/dragon-plus-query-encoder')
        question_model = AutoModel.from_pretrained('facebook/dragon-plus-query-encoder').cuda()
        question_tokenizer = context_tokenizer
        return question_tokenizer, question_model

    elif retriever == 'openai':

        set_openai_key()
        return None, None
    
    else:
        raise ValueError


def get_embeddings(retriever, inputs, mode='context'):
    import sys, inspect, numpy as np

    frm = inspect.stack()[1]
    caller = f"{frm.filename}:{frm.lineno}"
    print(
        f"[get_embeddings] caller={caller}, retriever={repr(retriever)}, n_inputs={0 if inputs is None else len(inputs)}",
        flush=True)

    if mode == 'context':
        tokenizer, encoder = init_context_model(retriever)
    else:
        tokenizer, encoder = init_query_model(retriever)
    
    all_embeddings = []
    batch_size = 24
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size)):
            print(f"input len{len(inputs)}" )
            # print(inputs)
            if retriever == 'dpr':
                input_ids = tokenizer(inputs[i:(i+batch_size)], return_tensors="pt", padding=True,max_length=encoder.config.max_position_embeddings,)["input_ids"].cuda()
                embeddings = encoder(input_ids).pooler_output.detach()
                # print(embeddings.shape)
                all_embeddings.append(torch.nn.functional.normalize(embeddings, dim=-1))
            elif retriever == 'contriever':
                # Compute token embeddings
                ctx_input = tokenizer(inputs[i:(i+batch_size)], padding=True, truncation=True, return_tensors='pt',max_length=encoder.config.max_position_embeddings,)
                # print(ctx_input.keys())
                # input_ids = context_tokenizer(contexts, return_tensors="pt", padding=True)["input_ids"].cuda()
                outputs = encoder(**ctx_input)
                embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
                all_embeddings.append(torch.nn.functional.normalize(embeddings, dim=-1))
            elif retriever == 'dragon':
                ctx_input = tokenizer(inputs[i:(i+batch_size)], padding=True, truncation=True, return_tensors='pt',max_length=encoder.config.max_position_embeddings,).to(device)
                embeddings = encoder(**ctx_input).last_hidden_state[:, 0, :]
                # all_embeddings.append(torch.nn.functional.normalize(embeddings, dim=-1))
                all_embeddings.append(embeddings)
            elif retriever == 'openai':
                # 直接用你已有的封装：get_openai_embedding(List[str]) -> List[List[float]]
                print(inputs[i:(i+batch_size)])
                vecs = get_openai_embedding(_sanitize_inputs(inputs[i:(i+batch_size)]))  # 外部已 set_openai_key()
                embeddings = torch.tensor(vecs, dtype=torch.float32, device=device)
                # 是否需要归一化取决于你下游相似度的做法；这里与 DPR/Contriever 保持一致做 L2 归一化
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                all_embeddings.append(embeddings)
            else:
                raise ValueError

    return torch.cat(all_embeddings, dim=0).cpu().numpy()

def _sanitize_inputs(raw):
    cleaned = []
    bad_items = []
    for idx, x in enumerate(raw):
        # 允许字符串；如果是字典/其它类型，看你是否希望转成字符串

        if x is None or x == "":
            s = "None"
        elif isinstance(x, str):
            s = x.strip()
        elif isinstance(x, (int, float, bool)):
            s = str(x)
        elif x is None or x == "":
            s = "None"
        else:
            # 如果你愿意把复杂对象序列化成字符串：
            import json
            s = json.dumps(x, ensure_ascii=False)
            # 更稳妥：记录并跳过
            # bad_items.append((idx, type(x).__name__, x))
            # continue

        if s:  # 丢弃空串
            cleaned.append(s)
        else:
            bad_items.append((idx, "empty", x))

    if bad_items:
        # 提前暴露问题位置，方便你反查上游来源
        preview = ", ".join([f"#{i}:{t}" for (i, t, _) in bad_items[:10]])
        raise ValueError(f"Embedding input contains invalid/empty items at {preview} (and possibly more).")

    return cleaned


def get_context_embeddings(retriever, data, context_tokenizer, context_encoder, captions=None):

    context_embeddings = []
    context_ids = []
    batch_size = 24
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for i in tqdm(range(1,20), desc="Getting context encodings"):
        contexts = []
        if 'session_%s' % i in data:
            date_time_string = data['session_%s_date_time' % i]
            for dialog in data['session_%s' % i]:

                turn = ''
                # conv = conv + dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"' + '\n'
                try:
                    turn = dialog['speaker'] + ' said, \"' + dialog['compressed_text'] + '\"' + '\n'
                    # conv = conv + dialog['speaker'] + ': ' + dialog['compressed_text'] + '\n'
                except KeyError:
                    turn = dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"' + '\n'
                    # conv = conv + dialog['speaker'] + ': ' + dialog['clean_text'] + '\n'
                if "img_file" in dialog and len(dialog["img_file"]) > 0:
                    turn += '[shares %s]\n' % dialog["blip_caption"]
                contexts.append('(' + date_time_string + ') ' + turn)

                context_ids.append(dialog["dia_id"])
            with torch.no_grad():
                # print(input_ids.shape)
                if retriever == 'dpr':
                    input_ids = context_tokenizer(contexts, return_tensors="pt", padding=True)["input_ids"].cuda()
                    embeddings = context_encoder(input_ids).pooler_output.detach()
                    # print(embeddings.shape)
                    context_embeddings.append(torch.nn.functional.normalize(embeddings, dim=-1))
                elif retriever == 'contriever':
                    # Compute token embeddings
                    inputs = context_tokenizer(contexts, padding=True, truncation=True, return_tensors='pt')
                    print(inputs.keys())
                    # input_ids = context_tokenizer(contexts, return_tensors="pt", padding=True)["input_ids"].cuda()
                    outputs = context_encoder(**inputs)
                    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
                    context_embeddings.append(torch.nn.functional.normalize(embeddings, dim=-1))
                elif retriever == 'dragon':
                    ctx_input = context_tokenizer(contexts, padding=True, truncation=True, return_tensors='pt')
                    embeddings = context_encoder(**ctx_input).last_hidden_state[:, 0, :]
                    context_embeddings.append(torch.nn.functional.normalize(embeddings, dim=-1))
                elif retriever == 'openai':
                    # 一次性取整段 contexts 的向量（如需可自行再切 batch）
                    vecs = get_openai_embedding(contexts)
                    embeddings = torch.tensor(vecs, dtype=torch.float32, device=device)
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                    context_embeddings.append(embeddings)
                else:
                    raise ValueError

    # print(context_embeddings[0].shape[0])
    context_embeddings = torch.cat(context_embeddings, dim=0)
    # print(context_embeddings.shape[0])

    return context_ids, context_embeddings
