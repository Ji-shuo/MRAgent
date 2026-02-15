"""
aggregate_gemini_results.py

用法:
  python aggregate_gemini_results.py \
      --index index.json \
      --root result/LM \
      --out aggregated.jsonl
"""

import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/question_list_LM.json", help="包含id字典的JSON文件路径")
    parser.add_argument("--root", default="result/LM", help="jsonl 所在根目录（默认 result/LM）")
    parser.add_argument("--model", default="gemini", help="jsonl 所在根目录（默认 result/LM）")
    parser.add_argument("--file", default="0", help="jsonl 所在根目录（默认 result/LM）")
    # parser.add_argument("--out", required=True, help="聚合输出文件路径（.jsonl）")
    args = parser.parse_args()

    model = args.model
    file = args.file
    index_path = Path(args.index)
    root = Path(args.root)
    out_path = Path(f"result/LM/ss_result_{model}_{file}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取索引JSON
    with index_path.open("r", encoding="utf-8") as f:
        index_obj = json.load(f)

    # 统计信息
    found, missing = 0, 0

    with out_path.open("w", encoding="utf-8") as fout:
        for _id, items in index_obj.items():
            # 逐个 id 查找对应 jsonl
            jsonl_path = root / f"{_id}_result_{model}_{file}.jsonl"
            if jsonl_path.exists():
                found += 1
                with jsonl_path.open("r", encoding="utf-8") as fin:
                    for line in fin:
                        line = line.rstrip("\n")
                        if line:  # 跳过空行
                            fout.write(line + "\n")
            else:
                missing += 1
                # 默认写合法 JSON（推荐）
                fout.write(json.dumps({"id": _id}, ensure_ascii=False) + "\n")

                # 如果你非要不加引号的键名（不规范 JSON），用下面这一行替换上一行：
                # fout.write(f'{{id:"{_id}"}}\n')
    found = 0

    out_path_analysis = Path(f"result/LM/ss_analysis_{model}_{file}.jsonl")
    with out_path_analysis.open("w", encoding="utf-8") as fout:
        for _id, items in index_obj.items():
            # 逐个 id 查找对应 jsonl
            jsonl_path = root / f"{_id}_analysis_{model}_{file}.json" 
            if jsonl_path.exists():
                found += 1
                with jsonl_path.open("r", encoding="utf-8") as fin:
                    skip = False
                    wrote = False
                    for line in fin:
                        line = line.rstrip("\n")
                        if not line:
                            continue

                        # 如果行中包含 "question1"，则标记从下一行开始输出
                        if "question1" in line:
                            skip = True
                            continue

                        # 找到第一个非空且在 question1 之后的行
                        if skip:
                            fout.write(line + "\n")
                            wrote = True
                            break

                    # 如果没有找到符合条件的行，写默认 JSON
                    if not wrote:
                        fout.write(json.dumps({"id": _id}, ensure_ascii=False) + "\n")
            else:
                missing += 1
                # 默认写合法 JSON（推荐）
                fout.write(json.dumps({"id": _id}, ensure_ascii=False) + "\n")

                # 如果你非要不加引号的键名（不规范 JSON），用下面这一行替换上一行：
                # fout.write(f'{{id:"{_id}"}}\n')

    print(f"Done. Found: {found}, Missing: {missing}, Output: {out_path}")

    # 读取和统计 cost 文件
    cost_data = [] 
    cost_found = 0
    cost_filtered = 0
    cost_missing = 0

    for _id, items in index_obj.items():
        cost_path = root / f"{_id}_cost_{model}_{file}.json"
        if cost_path.exists():
            try:
                with cost_path.open("r", encoding="utf-8") as fin:
                    line = fin.readline().strip()
                    if line:
                        data = json.loads(line)
                        prompt_tokens = data.get("prompt_tokens", 0)
                        completion_tokens = data.get("completion_tokens", 0)
                        cost = data.get("cost", 0.0)

                        # 只统计 prompt_tokens >= 10000 的数据
                        if prompt_tokens >= 10000:
                            cost_data.append({
                                "id": _id,
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "cost": cost
                            })
                            cost_found += 1
                        else:
                            cost_filtered += 1
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error reading {cost_path}: {e}")
        else:
            cost_missing += 1

    # 计算平均值并输出统计结果
    if cost_data:
        avg_prompt = sum(d["prompt_tokens"] for d in cost_data) / len(cost_data)
        avg_completion = sum(d["completion_tokens"] for d in cost_data) / len(cost_data)
        total_cost = sum(d["cost"] for d in cost_data)

        # 输出统计结果到文件
        cost_stats_path = Path(f"result/LM/cost_stats_{model}_{file}.json")
        with cost_stats_path.open("w", encoding="utf-8") as fout:
            stats = {
                "total_files_processed": cost_found,
                "files_filtered_out": cost_filtered,
                "files_missing": cost_missing,
                "average_prompt_tokens": round(avg_prompt, 2),
                "average_completion_tokens": round(avg_completion, 2),
                "total_cost": round(total_cost, 4),
                "details": cost_data
            }
            json.dump(stats, fout, ensure_ascii=False, indent=2)

        print(f"\nCost statistics:")
        print(f"  Files processed (prompt_tokens >= 10000): {cost_found}")
        print(f"  Files filtered out (prompt_tokens < 10000): {cost_filtered}")
        print(f"  Files missing: {cost_missing}")
        print(f"  Average prompt_tokens: {avg_prompt:.2f}")
        print(f"  Average completion_tokens: {avg_completion:.2f}")
        print(f"  Total cost: {total_cost:.4f}")
        print(f"  Output: {cost_stats_path}")
    else:
        print("\nNo cost data found or all files filtered out (prompt_tokens < 10000)")



if __name__ == "__main__":
    main()