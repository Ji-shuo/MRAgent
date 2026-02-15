import json
from memory_controller import MemoryController

TOOLS= [
  {
  "type": "function",
  "function":{
    "name": "edges_by_tag",
    "description": "Follow memory graph edges filtered by a {tag, key} pair to retrieve related events under a topic. Do NOT repeat the same key–tag combination. ",
    "parameters": {
      "type": "object",
      "properties": {
        "tag": {
          "type": "string",
          "description": "Select a tag aligned with the related keyword. When exploring, choose at least one tag from each related keyword."
        },
        "key": {
          "type": "string",
          "description": "A key from keys_candidates (e.g., a person/entity/topic)."
        },
        "note": {
        "type": "string",
        "minLength": 8,
        "maxLength": 80,
        "description": "Short, verifiable decision note for next round. No step-by-step reasoning."
        }
      },
      "required": ["tag", "key", "note"],
    }
  }
  },
  {
  "type": "function",
  "function":{
    "name": "query_conversation_time",
    "description": "Return WHEN the conversation containing the event occurred (conversation time). This is not the exact real-world event time.",
    "parameters": {
      "type": "object",
      "properties": {
        "event_id": {
          "type": "string",
          "description": "Target event ID (e.g., D1:1)."
        }
      },
      "required": ["event_id"],
    }
  }},
  {
"type": "function",
  "function":{
    "name": "query_event_keywords",
    "description": "Return salient keywords for an event (entities, topics, times). Use when the event is related but vague. Often followed by query_event_context.",
    "parameters": {
      "type": "object",
      "properties": {
        "event_id": {
          "type": "string",
          "description": "Event suspected to be relevant but lacking clarity (e.g., D3:2)."
        }
      },
      "required": ["event_id"],
    }
  }},
  {
"type": "function",
  "function":{
    "name": "query_event_context",
    "description": "Return surrounding conversational context of an event (before/after turns) when evidence is related but incomplete.",
    "parameters": {
      "type": "object",
      "properties": {
        "event_id": {
          "type": "string",
          "description": "Event whose context is needed (e.g., D3:2)."
        }
      },
      "required": ["event_id"],
    }
  }},
  {
    "type": "function",
    "function": {
      "name": "query_personal_information",
      "description": "List available aspects or happened events for a given person (e.g., hobbies, achievement, preference).",
      "parameters": {
        "type": "object",
        "properties": {
          "person": {
            "type": "string",
            "description": "Name of the person. e.g. user"
          }
        },
        "required": ["person"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "query_personal_aspect",
      "description": "Return detailed personal information for one selected aspect of the person.",
      "parameters": {
        "type": "object",
        "properties": {
          "person": {
            "type": "string",
            "description": "Name of the person."
          },
          "aspect": {
            "type": "string",
            "description": "One aspect returned by query_personal_information."
          }
        },
        "required": ["person", "aspect"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "query_topic_events",
      "description": "Return detailed events under queried topic.",
      "parameters": {
        "type": "object",
        "properties": {
          "topic": {
            "type": "string",
            "description": "topic id (e.g. D1:t2)."
          },
        },
        "required": ["topic"]
      }
    }
  }
]


class ToolBridge:
  def __init__(self, memroy_controller: MemoryController):
    self.memroy_controller = memroy_controller

  def call(self, tool_call: list):

    tool_results = []
    origin_all = []
    new_evidence = []
    evidence_all = []
    tool_events_dict = dict()

    for item in tool_call:
      args = json.loads(item["function"].get("arguments"))
      op = item["function"].get("name")
      a = args  # .get("args", {})
      origin = []
      # out = memory_dispatcher(**args)
      try:
        if op == "edges_by_tag":
          out, origin, evidence = self.memroy_controller.event_by_tag(**a)
          origin_all.extend(origin)
          evidence_all.extend(evidence)
        elif op == "query_semantic":
          out = self.memroy_controller.query_semantic(**a)
        elif op == "query_conversation_time":
          out, origin = self.memroy_controller.query_conversation_time(**a)
          origin_all.append(origin)
        elif op == "query_event_keywords":
          out = self.memroy_controller.query_event_keywords(**a)
        elif op == "query_event_context":
          out, origin = self.memroy_controller.query_event_context(**a)
          origin_all.extend(origin)
        elif op == "query_semantic_information":
          out, origin = self.memroy_controller.query_semantic_information(**a)
          origin_all.extend(origin)
        elif op == "query_personal_information":
          out = self.memroy_controller.query_personal_information(**a)
        elif op == "query_personal_aspect":
          out, origin = self.memroy_controller.query_personal_aspect(**a)
          origin_all.extend(origin)
        elif op == "query_topic_events":
          out, origin = self.memroy_controller.query_topic_events(**a)
          origin_all.extend(origin)
        else:
          out = {"error": f"unknown op {op}"}
      except Exception as e:
        out = {"error": str(e)}
      if op == "query_conversation_time":
        tool_results.append({
          "role": "tool",
          "tool_call_id": item.get('id'),  # 对应配对
          "content": str(out),
        })
      else:
        tool_results.append({
          "role": "tool",
          "tool_call_id": item.get('id'),  # 对应配对
          "content": str(out),
          # "name": item.function.name
        })
      if op in tool_events_dict.keys():
        tool_events_dict[op].append(origin)
      else:
        tool_events_dict[op] = [origin]

      print(f"+++++++++++++++=toolresults{tool_results}+++++++++++++")

    return tool_results, origin_all, evidence_all, tool_events_dict

