# prompts.py
import json
import config
class Prompts:

    EVENT_KEYWORDS_SYSTEM_PROMPT = """You are going to answer a question with keyword and corresponding tags(fact summary).  For every tag of key, produce a relevance score in [0.0, 1.0] reflecting how useful it is for answering question:
    {
      "keyword": "Caroline",
      "tag_scores": {
        "Plan": 0.0-1.0,
        "Conference": 0.0-1.0
      },
    }"""


    EVALUATE_CANDIDATE_USER_PROMPT = """QUESTION:
            <<<
            {QUESTION}
            >>> 
            CANDIDATES: <<<
            {CANDIDATES}
            >>> """

    @classmethod
    def evaluate_candidate_user_prompt(cls, question: str, candidates: str) -> str:
        return cls.EVALUATE_CANDIDATE_USER_PROMPT.format(
            QUESTION=question,  CANDIDATES=candidates
        )

    TRANSLATE_SYSTEM_PROMPT = """You are a dialogue processor. Only output valid JSON.
TASK:
- For each sentence: ! Every sentence in the original text MUST be preserved.
  1. Replace ALL pronouns ("I", "you", "he", "she", "it", "they", "we", "this", "that", "these", "those", "the", "the xx", "one") with explicit entities, events or noun phrases from the conversation context, such as "the event"->"charity race".  
  2. Do NOT modify verbs, adjectives, or other words. Only replace pronouns. 
  3. Use a short concrete noun to describe what the speaker is talking about in "tag", e.g. Movie Preference, Hobbies. No more than two words. 
  4. If a sentence uses a relative time (e.g., 'yesterday', 'next Monday', 'in two days', 'last summer'), compute the absolute calendar date based on conversation_time and output 'YYYY-MM-DD'. If the sentence includes a precise absolute date already, output that date. If time refers to a period, output the midpoint date 'YYYY-MM-DD'. If no time is mentioned or mentioned time is BC date, output the conversation date (YYYY-MM-DD).
  5. If a sentence ends with a question, merge the question content into the next sentence that provides an answer to complete sentence information.
- Topics:derive at least ten concrete topics overall (short sentences). Assign topic IDs (t1..tn). In each sentence, fill 'topic' with a list of topic IDs that apply; use [] if none
- Personal information: extract person-related facts (preferences, roles, schedules, background, relations, attributes) into 'personal_sentences'. If a fact is already in a sentence, also duplicate a concise normalized version here.
- "id" in "sentence" is combination of "origin" and number. The "origin" must exactly correspond to the "dia_id" field in the "Dialogue". Do not create or invent new ids. 
Schema(One line of json):
{"conversation_time":"YYYY-MM-DD", "sentence":[{"id":"D1:1-1", "text":"sentence.", "tag":"short concrete tag","origin":"D1:1","topic": ["t1","t3"],"time":"YYYY-MM-DD"}],"topics":{"t1": "Nate plans the charity race route","t2": "Joanna discusses aquarium maintenance"},"personal_sentences":[{"id":"p1","text":"Nate enjoys long-distance running.","tag":"preference","origin":"D1:1","person": "Nate"}]}
"""

    TRANSLATE_PROMPT = """Dialogue:
        <<<
        {RAW_TEXT}
        >>>"""

    @classmethod
    def extract_translate_prompt(cls, raw_text: str) -> str:
        return cls.TRANSLATE_PROMPT.format(
            RAW_TEXT=raw_text,
        )


    KEYWORD_SYSTEM_PROMPT = """You are an information extraction system. Only output valid JSON.
- For each input sentence, extract 2–30 keywords DIRECTLY from the original text, such as "drew", "park", "lake sunrise". Do not invent , paraphrase, or generalize. Do not include inferred words unless they explicitly appear in the text.
- Keyword types to consider: entity | topic | verb | time | location | task | event | people.
- For each sentence, extract ALL words/phrases that match these types if they are explicitly present. 
- "sentence_id" must be same with "id" in TEXT. Do not create or invent new ids.
Schema(One line of json):
{"sentence":[{"sentence_id":"D1:1-1","keyword":["Coraline","park"]}]}"""

    # KEYWORD_SYSTEM_PROMPT = """You are an information extraction system. Only output valid JSON.
    # TASK 1: Keyword Extraction
    # - For each input sentence, extract 2–30 keywords DIRECTLY from the original text, such as "drew", "park", "lake sunrise". Do not invent , paraphrase, or generalize. Do not include inferred words unless they explicitly appear in the text.
    # - Keyword types to consider: entity | topic | verb | time | location | task | event | people.
    # - For each sentence, extract ALL words/phrases that match these types if they are explicitly present.
    # - Use a meaningful, concrete one-word tag describing the event, such as "plan", "meeting".
    # - "sentence_id" must be same with "id" in TEXT. Do not create or invent new ids.
    #
    #  Generate a JSON object strictly following the given schema, no extra text.
    # Schema:
    # {
    #   "sentence":[
    #     {
    #       "sentence_id":"D1:1-1",
    #       "keyword":["Coraline","park"],
    #       "tag":"plan"
    #     }
    #   ],
    # }
    #
    #     """


    KEYWORD_PROMPT = """TEXT:
        <<<
        {RAW_TEXT}
        >>>
        """

    @classmethod
    def extract_keyword_prompt(cls, raw_text: str, tag_list:str) -> str:
        return cls.KEYWORD_PROMPT.format(
            RAW_TEXT=raw_text
        )

    """   -  For time-related questions (e.g., "When…", "What date…"), call query_conversation_time, output the answer only as an absolute date or relative date grounded to query conversation time. Format must be: '7 May 2023', 'May 2023', '2023','The week/Sunday before 25 May 2023' and no extra word.
    """

    ANSWER_SORT_PROMPT = """You are a careful QA reasoner working over a memory of timestamped events. For every event in top_texts, produce a relevance score in [0.0, 1.0] reflecting how useful it is for answering question, do not make up event id:
    {
      "mode": "score",
      "relevance_scores": {
        "D1:1-1": 0.0-1.0,
      },
    } DO NOT output extra explanation."""

    ANSWER_SORT_PROMPT2 = """You are a careful QA reasoner working over a memory of timestamped events. For every event in top_texts, select at most 20 relevant events for answering question, do not make up event id:
        {
          "mode": "sort", 
          "events": ["D1:1","D1:2"]
        } DO NOT output extra explanation."""

    SYSTEM_TOOL_PROMPT = """ You are a diligent question-answering agent. You must gather and verify all relevant information before producing your final answer.
    You need to answer a question by searching key candidates and corresponding tags and personal information and similar_topic and similar_sentence. try to get more information with tools. call the tools with the proper argument. Do NOT describe the call in text or JSON. In each round, you must call as many relevant tools as possible, rather than skipping potential ones.
    If the question or answer is related to person, call query_personal_information and query_personal_aspect."""

    ANSWER_SYSTEM_PROMPT_FINAL = """
       You must answer the question with queried contents.
          Rules:
          -  For yes/no or binary questions, output 'Yes', 'No', 'Likely yes', 'Likely no'.    
          -  For "where / location / place" questions, the answer should be a concrete and specific place name. If no exact name is mentioned, describe it instead.
          -  For "what / which" questions, try to respond with one specific, concrete item directly asked for or descriptions of the answer. 
          -  For other questions, output only the minimal answer (key phrase or entity) without extra context.

           Format:
           - "answer": If the events already provide sufficient evidence to answer the question, then produce the final short answer, only asked part, not full sentence.   
           {
             "mode": "answer",
             "answer": "...", 
             "supports": ["D1:1","D1:2"],
             "confidence": 0.0-1.0
           }  """



#    -  If the question is about person, you can call query_personal_information (person is user). If the question is about some events, you could call query_topic_events.

    ANSWER_SYSTEM_TOOL_PROMPT = """You are a diligent question-answering agent. You always want to gather and verify all relevant information before producing your final answer.
You need to either answer a question or call tools to get more information with key candidates and corresponding tags and key_sentence. Write short answer with exact words from event whenever possible.
   Rules:
   -  For yes/no or binary questions, output 'Yes', 'No', 'Likely yes', 'Likely no'.
   -  For "where / location / place" questions, the answer must be a concrete and specific place name. If the sentence only provides a vague or ambiguous location, call query_event_keywords to further explore and identify a more specific place.
   -  For "what / which" questions, respond with one specific, concrete item (an event, subject, person, organization, place, or titled work) directly asked for—not a category, type, or class.
   -  For "how many" questions, the answer must be the number of tasks/objects, not the number of physical categories.
   -  For temporal questions (e.g., "How many days/weeks/months ago...", "How many days passed between..."), use "current_date" in the input as TODAY's date to calculate the time difference. Example: if current_date is "2023-02-01" and an event happened on "2023-01-25", then "7 days ago" is the answer.
   -  For other questions, output only the minimal answer (key phrase or entity) without extra context.
   -  There may be multiple answers, you should try to explore more relevant information.
    
    Decide ONE mode of:  
    - "answer": If the events already provide sufficient evidence to answer the question, then produce the final short answer, only asked part, not full sentence.  
   If the information is vague or incomplete, you may further query_personal_information, query_topic_events, query_semantic_information query_event_keywords or query_event_context.  
    {
      "mode": "answer",
      "answer": "...", 
      "supports": ["D1:1","D1:2"],
      "confidence": 0.0-1.0
    }
    
    - "navigate": If evidence is insufficient, you must immediately call the tools with the proper argument. Do NOT describe the call in text or JSON. In each round, you must call as many relevant tools as possible, rather than skipping potential ones. Only avoid calls that are clearly irrelevant."""

    ANSWER_SYSTEM_TOOL_PROMPT_AB = """You are a diligent question-answering agent. You always want to gather and verify all relevant information before producing your final answer.
    You need to either answer a question or call tools to get more information with key candidates and corresponding tags and key_sentence. Write short answer with exact words from event whenever possible.
       Rules:
       -  For yes/no or binary questions, output 'Yes', 'No', 'Likely yes', 'Likely no'.    
       -  For "where / location / place" questions, the answer must be a concrete and specific place name. If the sentence only provides a vague or ambiguous location, call query_event_keywords to further explore and identify a more specific place.
       -  For "what / which" questions, respond with one specific, concrete item (an event, subject, person, organization, place, or titled work) directly asked for—not a category, type, or class.
       -  For "how many" questions, the answer must be the number of tasks/objects, not the number of physical categories.
       -  For other questions, output only the minimal answer (key phrase or entity) without extra context.
       -  There may be multiple answers, you should try to explore more relevant information.

        Decide ONE mode of:  
        - "answer": If the events already provide sufficient evidence to answer the question, then produce the final short answer, only asked part, not full sentence.  
       If the information is vague or incomplete, you may further query_personal_information, query_topic_events, query_event_keywords or query_event_context.  
        {
          "mode": "answer",
          "answer": "...", 
          "supports": ["D1:1","D1:2"],
          "confidence": 0.0-1.0
        }

        - "navigate": If evidence is insufficient, you must immediately call the tools with the proper argument. Do NOT describe the call in text or JSON. In each round, you may make at most {CALL} tool calls. """

    ANSWER_SYSTEM_TOOL_PROMPT_AB = ANSWER_SYSTEM_TOOL_PROMPT_AB.replace("{CALL}", str(config.CALL))

    ANSWER_SYSTEM_TOOL_PROMPT3 = """You are a diligent question-answering agent. You always want to gather and verify all relevant information before producing your final answer.
    You need to answer a question or call tools to get more information with key candidates and corresponding tags and key_sentence. 
       Rules:
       -  For yes/no or binary questions, output 'Yes', 'No', 'Likely yes', 'Likely no'  only.    
       -  For "where / location / place" questions, the answer should be a concrete and specific place name. 
       -  For "what / which" questions, respond with one specific, concrete item (an event, subject, person, organization, place, or titled work) directly asked for.
       -  For other questions, output only the minimal answer (key phrase or entity) without extra context.
       -  Do NOT include any explanation, evidence, punctuation wrappers, or extra words in "answer".
       -  Prefer dataset-style canonical labels if possible (short noun phrases).
       -  There may be multiple answers, you should try to explore more relevant information.
       -  "reason" must be concise (<= 1-2 sentences). Do NOT restate the whole context, only key phase.

        Decide ONE mode of:
        - "answer": If the events already provide sufficient evidence to answer the question, then produce the final short answer, only asked part, not full sentence.  Double-check that your answer directly addresses the question asked
       If the information is vague or incomplete, you may further query_personal_information, query_topic_events, query_semantic_information query_event_keywords or query_event_context.  
        {
          "mode": "answer",
          "answer": "...", 
          "reason": "...",
          "supports": ["D1:1","D1:2"],
          "confidence": 0.0-1.0 
        }

        - "navigate": If evidence is insufficient, immediately call the tools with the proper argument. Do NOT describe the call in text or JSON. In each round, you must call as many relevant tools as possible, rather than skipping potential ones. Only avoid calls that are clearly irrelevant."""


    ANSWER_SYSTEM_PROMPT_FINAL2 = """
    You need to answer a question with queried contents. Write short answer with exact words from event whenever possible.
       Rules:
       -  For yes/no or binary questions, output 'Yes', 'No', 'Likely yes', 'Likely no'.    
       -  For "where / location / place" questions, the answer must be a concrete and specific place name. If the sentence only provides a vague or ambiguous location, call query_event_keywords to further explore and identify a more specific place.
       -  For "what / which" questions, respond with one specific, concrete item (an event, subject, person, organization, place, or titled work) directly asked for—not a category, type, or class.
       -  For other questions, output only the minimal answer (key phrase or entity) without extra context.
       - If you find answer relevant but not concrete, you can give than answer.

        Format:
        - "answer": If the events already provide sufficient evidence to answer the question, then produce the final short answer, only asked part, not full sentence.  

        {
          "mode": "answer",
          "answer": "...", 
          "supports": ["D1:1","D1:2"],
          "confidence": 0.0-1.0
        }  """

    """Requirements:
    - In navigate mode, you must have  multiple tool calls. 
    - For "what / which / about" questions, the final answer must contain a specific topic or subject, not just a restatement like "do some research". If no concrete topic is available in current evidence, you must switch to "navigate" mode and query event context/keywords and edges_by_tag with tags like "Research", "Topic", "Subject", or "Project".
    - Only after exhausting all relevant tool queries (event context, keywords, edges_by_tag across related keys and tags) and still finding no clear evidence, you may finally answer with "unknown" or "cannot be determined".
    """

    ANSWER_SYSTEM_PROMPT = """You need to answer a question with key candidates and corresponding tags and top events.Write short answer with exact words from event whenever possible.
       Rules:
       -  For time-related questions (e.g., "When…", "What date…"), call query_conversation_time, output the answer only as an absolute date or relative date grounded to query conversation time. Example: '7 May 2023', 'May 2023', '2023','The xx before 25 May 2023'.
       -  For yes/no or binary questions, output only 'Yes', 'No', 'Likely yes', 'Likely no'. Do not include explanations.    
       -  For "where / location / place" questions, the answer must be a concrete place name, do not output vague words like "hometown"
       -  For "what / which" questions, the answer must be a specific event, subject, project, topic, or entity directly related to the question.
       -  For other questions, output only the minimal answer (key phrase or entity) without extra context.

        Decide ONE mode of:
        - "answer": If the events already provide sufficient evidence to answer the question, then produce the final short answer, only asked part, not full sentence.  
       If the information is vague or incomplete, you may further query_semantic_information query_event_keywords or query_event_context.  
        {
          "mode": "answer",
          "answer": "...", 
          "supports": ["D1:1","D1:2"],
          "confidence": 0.0-1.0
        }

        - "navigate": If evidence is insufficient, DO NOT describe plans in JSON. INSTEAD, immediately call the tool memory_dispatcher with the proper arguments
          Allowed tool operations:
          - edges_by_tag(args: {tag, key, limit})  
            (Choose at least one tag from each related keyword! Do not repeat the same key–tag combination. Key-tag is selected from keys_candidates.)  
          - query_conversation_time(args: {event_id})  
            (Only returns the time when the conversation happened; exact event times still need to be analyzed.)  
          - query_event_keywords(args: {event_id})  
            (Returns keywords related to the event, useful when detected event is related but needs exploration. Also call query_event_context)  
          - query_event_context(args: {event_id})  
            (Returns surrounding context when the detected event is related but incomplete.)  
          - query_semantic_information(args: {key})
            (Returns summaries for the keyword.)

        Schema: 'function': {'arguments': '{"op": "edges_by_tag", "args": {"tag": "Future education plans", "key": "Caroline", "limit": 8}} 
        'function': {'arguments': '{"op": "query_conversation_time", "args": {"event_id": "D1:1"}}

        Requirements:
        - In navigate mode, you must have  multiple tool calls. 
        - Always replace vague or relative time expressions in the answer with the absolute time values provided by tools, such as "last year"-> "2023", "yesterday" -> "2023-08-05"
        - For "what / which / about" questions, the final answer must contain a specific topic or subject, not just a restatement like "do some research". If no concrete topic is available in current evidence, you must switch to "navigate" mode and query event context/keywords and edges_by_tag with tags like "Research", "Topic", "Subject", or "Project".
        - Only after exhausting all relevant tool queries (event context, keywords, edges_by_tag across related keys and tags) and still finding no clear evidence, you may finally answer with "unknown" or "cannot be determined".
        """


    ANSWER_USER_PROMPT = """TEXT:
        <<<
        {RAW_TEXT}
        >>>"""


    @classmethod
    def extract_answer_prompt(cls, raw_text: str) -> str:
        return cls.ANSWER_USER_PROMPT.format(
            RAW_TEXT=raw_text,
        )


    EVENT_SYSTEM_PROMPT = """Only output valid JSON.

TASK (Step 1 - Event Extraction):
- Split each conversation sentence into one or more events.   
- Do NOT paraphrase or shorten; replace pronouns(including I, you, the, those, these, this, they, we) with entity or people names in conversation.  
- Every original conversation sentence must map to at least one event. Do not delete any description of the event, including time, place, mood.  
- Output under key "sentence".  

TASK (Step 2 - Semantic Information):
- Summarize facts grouped by SUBJECT (each entity/item/activity/concept).  
- For each each entity/item/activity/concept, aggregate facts into one string.  
- Output under key "semantic_information".  

TASK (Step 3 - Context Links):
- If two sentences are related or describe the same situation, add a context link.  
- Output under key "context_links".  

Validation Rule:
- Count of conversation sentences = Count of covered "sentence" origins.  
- No information loss: time, place, actions, mood, objects must remain.  

Schema:
{
  "conversation_time":"YYYY-MM-DD HH:MM am/pm",
  "episode_event":[
    {
      "id":"e1",
      "text":"sentence after splitting.",
      "origin":"D1:1"
    }
  ],
  "semantic_information":[
    {
      "id":"s1",
      "entity":"Caroline",
      "information":"Caroline likes camping.",
      "origins":["D1:1","D1:3"]
    }
  ],
  "context_links":[
    {
      "source_event":"e1",
      "target_event":"e2",
      "relation":"context"
    }
  ]
}
    """

    EVENT_USER_PROMPT = """TEXT:
    <<<
    {RAW_TEXT}
    >>>"""

    @classmethod
    def extract_event_prompt(cls, raw_text: str) -> str:
        return cls.EVENT_USER_PROMPT.format(
            RAW_TEXT=raw_text,
        )

    EVENT_KEY_SYSTEM_PROMPT = """You are a keyword and relation extractor. Only output valid JSON. 
    Keyword types (all must be extracted for each event): entity | topic | predicate | time | location | task | event | people.
    Extract all keywords of all the above types for each episode event. 
    Link each extracted keyword to its corresponding event with a meaningful tag (one word) summarizing the event.
    Choose tag from existing tag list; only generate a new tag if no suitable one exists.
    Schema:
    {
      "keywords": [
        {
          "id": "Extracted keyword",
          "word": "Extracted keyword",
        }
      ],
    "episode_edges": [
        {
          "key_id": "Extracted keyword",
          "episode_id": "e1",
          "tag": "Tag"
        }
      ], 
    }
    Requirements:
    - key_id in edges must be in keywords. 
    - All relative times in episode events (e.g., 'yesterday') MUST be grounded into absolute ISO dates using the given CONVERSATION_TIME.
    - Every event (episode_id) must be connected to ALL its extracted keywords.
    - A keyword can be linked to multiple events if it is relevant.
    - The 'tag' field is not the keyword type. Instead, generate a meaningful word that summarizes the event, such as "Painting"
    """

    EVENT_KEY_USER_PROMPT = """TEXT:
        <<<
        {RAW_TEXT}
        >>>
        EXISTING_TAG:
        <<<
        {RAW_TAG}
        >>>
        """

    @classmethod
    def extract_event_key_prompt(cls, raw_text: str, tag_list:str) -> str:
        return cls.EVENT_KEY_USER_PROMPT.format(
            RAW_TEXT=raw_text, RAW_TAG=tag_list,
        )


    QUESTION_KEY_SYSTEM_PROMPT = """You are a keyword extractor. Only output valid JSON. 
    Given a question, extract keywords to find answers. Keyword types (all must be extracted for each question): entity | topic | predicate | time | location | task | event | people. For each keyword, also provide possible alternative expressions, including Synonyms, different form, different tense. Different tense of word is mandatory.
    If the question contains a time limit, return it in "question_time" as: "YYYY-MM-DD, YYYY-MM-DD". If no time info, set "question_time" as "". If no year, then write "MM-DD, MM-DD".If a single day, repeat the same date(e.g., "YYYY-MM-DD, YYYY-MM-DD").
    If no year appears, DO NOT guess or infer a year. Use only 'MM-DD, MM-DD'.
    Schema:
    {
      "question_time": "YYYY-MM-DD, YYYY-MM-DD or '' or MM-DD, MM-DD",
      "keywords": [
        {
          "id": "Extracted keyword",
          "alternatives": ["Possible tense", "Different Synonyms", "Different form"]
        }
      ] 
    }"""

    QUESTION_KEY_USER_PROMPT = """QUESTION:
        <<<
        {RAW_TEXT}
        >>>"""

    @classmethod
    def extract_question_key_prompt(cls, raw_text: str) -> str:
        return cls.QUESTION_KEY_USER_PROMPT.format(
            RAW_TEXT=raw_text,
        )

    # -------- 事实提炼 --------
    FACTS_SYSTEM = """You are an information extractor for a key–value memory system.
    Task: Refine raw text into minimal, self-contained facts while preserving as much information as possible (use the original wording whenever possible)
    Resolve pronouns using LOCAL CONTEXT ONLY (no external hint map).
    Ground deictic times (yesterday/today/next Monday…) to ISO dates using CONVERSATION_TIME.
    If a reference remains uncertain, keep the original token but record it in replacements with "ambiguous": true.
    Do not invent facts. One assertion per fact; keep negations, numbers, units, qualifiers; 30–220 chars.
    Return ONLY a single JSON object

    Output JSON only:
    {
      "facts": [
        {
          "id": "v1",
          "text": "Refined fact with resolved entities and grounded dates.",
          "origin": "Original snippet.",
          "replacements": [
            {"from":"he","to":"Bob Lee","type":"coref","ambiguous":false},
            {"from":"yesterday","to":"2025-08-25","type":"time","ambiguous":false}
          ]
        }
      ]
    }"""

    FACTS_USER_TMPL = """DIALOG:

    TEXT:
    <<<
    {RAW_TEXT}
    >>>

    Task:
    1) Split into atomic factual statements.
    2) Resolve pronouns from LOCAL CONTEXT ONLY.
    4) Preserve negations, numbers, units, qualifiers.
    5) Return ONLY the JSON per schema."""

    @classmethod
    def build_facts_prompt(cls, raw_text: str, reference_date: str, timezone_str: str) -> str:
        return cls.FACTS_USER_TMPL.format(
            REFERENCE_DATE=reference_date,
            TIMEZONE=timezone_str,
            RAW_TEXT=raw_text,
        )

    # -------- 多视角 key 生成 --------
    KEYS_SYSTEM = """You are a Query-Key Extractor for a key–value memory.
    TASK: From given values, extract discriminative ATOMIC keys and link them to values.
    Key types: entity | topic | predicate | time | location | task.
    Rules
    - One type per key; prefer base lemmas; short, specific heads.
    - Use snake_case inside key text (e.g., lgbtq_support_group).
    - Predicate: base verb or short verb phrase in lemma form
    - Time → absolute ISO dates (YYYY-MM-DD) or ranges (e.g., 1949-).
    - Drop generic/low-specificity keys (person, thing, do, event, info).
    Schema:
    {
      "values": [ { "id":"v1", "text":"..." } ],
      "keys":   [ { "key_id":"k1", "text":"is_capital_of", "type":"predicate", "key_quality_score": 0.92 } ],
      "edges":  [ { "key_id":"k1", "value_id":"v1" } ] 
    }"""

    KEYS_USER_TMPL = """
    VALUE_CANDIDATES:
    {VALUE_JSON}

    Task:
    1) Use VALUE_CANDIDATES as the values (do not re-extract).
    2) Generate multi-view keys (types above), deduplicate, score in [0,1].
    3) Produce edges linking keys to values.
    4) Return ONLY the JSON per schema."""

    @classmethod
    def build_keys_prompt(cls, raw_text: str, values: list[dict]) -> str:
        return cls.KEYS_USER_TMPL.format(
            RAW_TEXT=raw_text,
            VALUE_JSON=json.dumps(values or [], ensure_ascii=False, indent=2),
        )

    # -------- Questions → keys --------
    QUESTIONS_SYSTEM = """You are a Query-Key Extractor for a key–value memory.

    Task
    - For each question, extract a small set of discriminative atomic keys.

    Types
    - entity | topic | predicate | time | location | task

    Rules
    - One type per key; prefer base lemmas; short, specific heads.
    - Use snake_case inside key text (e.g., lgbtq_support_group).
    - Predicate: base verb or short verb phrase in lemma form
    - Time → absolute ISO dates (YYYY-MM-DD) or ranges (e.g., 1949-).
    - Drop generic/low-specificity keys (person, thing, do, event, info).

    Relations (AND/OR over keys)
    - Encode composition ONLY via relations using key_ids from keys.
    - Format: "relations": [["k1","k2"],["k3"]] ≡ (k1 AND k2) OR (k3).
    - AND cues: “and”, “both…and”, “as well as”, required co-occurrence (e.g., subject+predicate+topic/time).
    - OR cues: “or”, “either…or”.
    - 1–3 keys per AND-group; ≤4 max. If unclear, use one AND-group with the 1–3 most discriminative keys.

    Output (JSON only)
    {
      "question_key": [
        {
          "id": "q1",
          "keys": [
            {"key_id":"k1","text":"...","type":"entity|topic|predicate|time|location|task","key_quality_score":0.92}
          ],
          "relations":[["k1","k2"],["k3"]]
        }
      ]
    }

    Scoring & Limits
    - Deduplicate; score ∈ [0,1]; drop < 0.5.
    - Keep compact: ~4–12 keys per question.

    Do not add explanations, code fences, or extra text.
    """

    QUESTIONS_PROMPT = """QUESTION:
    <<<
    {QUESTION_TEXT}
    >>>
    Return ONLY the JSON object defined in QUESTIONS_SYSTEM.
    """

    @classmethod
    def build_questions_prompt(cls, question_text: str) -> str:
        return cls.QUESTIONS_PROMPT.format(
            QUESTION_TEXT=question_text.strip(),
        ).strip()




