#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Main server file for Rehan AI
# -------------------------
# IMPORTS

import torch
import json
import os
import re
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# -------------------------
# CONFIGURATION

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LORA_PATH = "./app/lora-llama3/Training3/checkpoint-814"
CHAT_JSON_PATH = "chats.json"

MAX_NEW_TOKENS = 400
MAX_CONTEXT_TURNS = 80
BOOTSTRAP_TURNS = 80

# -------------------------
# LOAD BOOTSTRAP MEMORY

bootstrap_memory = []

# -------------------------
# LOAD MODEL ON STARTUP

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model (4-bit)...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    device_map="auto",
)

print("Loading LoRA checkpoint...")
model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
)

model.eval()
print("Model ready.")

# -------------------------
# FASTAPI APP

app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str 
    name: str
    message: str 

# Per-user in-memory conversation store
memory = {} 

# -------------------------
# UI ENDPOINT

@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta charset="UTF-8">
<title>Rehan AI</title>
<style>
    body {
        margin: 0;
        background: #0f0f0f;
        color: #e5e5e5;
        font-family: system-ui;
        min-height: 100vh;
        display: flex;
        justify-content: center;
    }

    .screen {
        width: 100%;
        max-width: 700px;
        min-height: 100vh;
        display: none;
        flex-direction: column;
    }

    .active {
        display: flex;
    }

    #name-screen {
        align-items: center;
        justify-content: center;
        gap: 20px;
    }

    input {
        background: #1a1a1a;
        border: none;
        color: white;
        padding: 14px;
        border-radius: 10px;
        font-size: 16px;
        width: 260px;
        outline: none;
    }

    button {
        background: #444;
        border: none;
        color: white;
        padding: 14px 24px;
        border-radius: 10px;
        cursor: pointer;
        font-size: 15px;
    }

    button:hover {
        background: #666;
    }

    #chat {
        flex: 1;
        overflow-y: auto;
        padding: 20px;
    }

    .msg {
        max-width: 70%;
        margin-bottom: 14px;
        padding: 12px 16px;
        border-radius: 12px;
        white-space: pre-wrap;
    }

    .user {
        background: #1e1e1e;
        align-self: flex-end;
    }

    .bot {
        background: #2a2a2a;
        align-self: flex-start;
    }

    #input-bar {
        display: flex;
        padding: 16px;
        border-top: 1px solid #222;
        background: #0f0f0f;
    }

    @media (max-width: 600px) {
        #chat {
            padding: 12px;
        }

        .msg {
            max-width: 85%;
            font-size: 15px;
        }

        input, button {
            font-size: 14px;
        }
    }

    #msg-input {
        flex: 1;
    }
</style>
</head>

<body>

<!-- NAME SCREEN -->
<div id="name-screen" class="screen active">
    <h2>Hey 👋 What’s your name?</h2>
    <input id="name-input" placeholder="Your name..." />
    <button onclick="startChat()">Continue</button>
</div>

<!-- CHAT SCREEN -->
<div id="chat-screen" class="screen">
    <div id="chat"></div>
    <div id="input-bar">
        <input id="msg-input" placeholder="Type something..." />
        <button onclick="send()">Send</button>
    </div>
</div>

<script>
let USER_ID = crypto.randomUUID();
let SESSION_ID = crypto.randomUUID();
let USER_NAME = "";
const chat = document.getElementById("chat");

function startChat() {
    const name = document.getElementById("name-input").value.trim();
    if (!name) return;

    USER_NAME = name;
    document.getElementById("name-screen").classList.remove("active");
    document.getElementById("chat-screen").classList.add("active");
}

function addMessage(text, cls, sender="") {
    const div = document.createElement("div");
    div.className = "msg " + cls;
    div.textContent = sender ? `${sender}: ${text}` : text;
    chat.appendChild(div);
}

async function send() {
    const input = document.getElementById("msg-input");
    const text = input.value.trim();
    if (!text) return;

    addMessage(text, "user");
    input.value = "";

    const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            session_id: SESSION_ID,
            name: USER_NAME,
            message: text,
        })
    });

    const data = await res.json();
    addMessage(data.reply, "bot", "Rehan");
}

document.getElementById("msg-input").addEventListener("keydown", e => {
    if (e.key === "Enter") send();
});
</script>

</body>
</html>
"""

#ALL OF THE MEMORY BELOW IS STATIC AND DOES NOT CHANGE DURING A SESSION

SELF_MEMORY ="""
CANONICAL SELF MEMORY (always true):

- Name: Rehan
- Gender: male
- Orientation: straight

Principles:
- Hates cheating
- Values honesty
- Avoids emotional chaos
- Does not overshare personal history

Background (do NOT expand beyond this):
- Had an unstable childhood
- Became independent early
- Does not like discussing past details

Communication style:
- Short to medium replies
- Reactive, not monologues
- Uses slang naturally (gng, damn, holy damn)
- Uses emojis sparingly, not every line

Hard rules:
- Do NOT invent trauma, relationships, crushes, or life events
- If unsure, say "idk", "not really", or ask instead
"""

#PERSON MEMORY STORE

PEOPLE_MEMORY = {
    "Hana": {
        "public_facts": {
            "name": "Hana",
            "country": "India",
        },
        "private_facts": {
            "romantic_history": "Never had a romantic or crush-based relationship with Rehan",
        },
        "behavior_profile": {
            "bond": "emotional",
            "tone": "gentle",
            "boundaries": [
                "avoid sarcasm",
                "be reassuring",
                "do not imply romantic history"
            ]
        }
    },

    "Elena": {
        "public_facts": {
            "name": "Elena",
        },
        "private_facts": {},
        "behavior_profile": {
            "bond": "friendly",
            "tone": "playful",
            "boundaries": [
                "no heavy flirting unless initiated",
                "no personal assumptions"
            ]
        }
    }
}

# BUILD MEMORY STRINGS

def build_public_facts(name: str):
    person = PEOPLE_MEMORY.get(name)
    if not person:
        return ""

    lines = []
    for k, v in person.get("public_facts", {}).items():
        lines.append(f"- {k}: {v}")

    return "\n".join(lines)

def build_private_facts(name: str):
    person = PEOPLE_MEMORY.get(name)
    if not person:
        return ""

    lines = []
    for k, v in person.get("private_facts", {}).items():
        lines.append(f"- {k}: {v}")

    return "\n".join(lines)

def build_behavior_profile(name: str):
    person = PEOPLE_MEMORY.get(name)
    if not person:
        return ""

    bp = person["behavior_profile"]
    lines = [
        f"- Bond: {bp['bond']}",
        f"- Tone: {bp['tone']}",
    ]

    for b in bp.get("boundaries", []):
        lines.append(f"- {b}")

    return "\n".join(lines)

# -------------------------
# SESSION MEMORY STORAGE

MEMORY_DIR = "/app/memory_store"
os.makedirs(MEMORY_DIR, exist_ok=True)

def load_session(session_id: str):
    path = os.path.join(MEMORY_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_session(session_id: str, data: dict):
    path = os.path.join(MEMORY_DIR, f"{session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# -------------------------
# CHAT ENDPOINT

def extract_candidate_facts(message: str):
    facts = {}
    msg = message.lower()

    birthday_match = re.search(
        r"\bmy\s+(?:birthday|bday)\s+is\s+(.+)",
        msg
    )
    if birthday_match:
        facts["birthday"] = birthday_match.group(1).strip()

    country_match = re.search(
        r"\bi\s+am\s+from\s+(.+)",
        msg
    )
    if country_match:
        facts["country"] = country_match.group(1).strip()

    return facts

# -------------------------
# CHAT HANDLER

@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id
    name = req.name.strip()
    user_msg = req.message.strip()

    # -------------------------
    # LOAD OR INIT SESSION

    session = load_session(session_id)

    if session is None:
        session = {
            "session_id": session_id,
            "name": name,
            "conversation": [],

            "facts": {
                "known": {
                    "name": name,
                },
                "private": {},
                "candidates": {},
            }
        }

    # -------------------------
    # EXTRACT CANDIDATE FACTS

    extracted_candidate_facts = extract_candidate_facts(user_msg)

    PROMOTABLE_FACTS = {
        "birthday",
        "country",
        "city",
        "age",
        "profession",
        "exam",
    }    
    
    PROMOTION_THRESHOLD = 2

    for key, value in extracted_candidate_facts.items():
        cand = session["facts"]["candidates"].setdefault(
            key,
            {"value": value, "count": 0}
        )

        if cand["value"] == value:
            cand["count"] += 1
        else:
            cand["value"] = value
            cand["count"] = 1


    for key, data in list(session["facts"]["candidates"].items()):
        if data["count"] >= PROMOTION_THRESHOLD:
            session["facts"]["private"][key] = data["value"]
            del session["facts"]["candidates"][key]
 
    # -------------------------
    # SAVE USER MESSAGE
 
    session["conversation"].append({
        "role": "user",
        "content": user_msg
    })

    # ------------------------- 
    # BUILD MESSAGES

    system_prompt = {
    "role": "system",
    "content": f"""
        You are Rehan.

        Your identity is fixed:
            - Name: Rehan
            - Gender: male
            - Orientation: straight

        You are chatting casually on Instagram.

        ====================
        KNOWN PUBLIC FACTS ABOUT THE USER:
            {build_public_facts(name)}

        ====================
        BEHAVIOR RULES FOR THIS USER:
            {build_behavior_profile(name)}

        ====================
        PRIVATE FACTS:
            {build_private_facts(name)}
            {json.dumps(session["facts"]["private"], indent=2)}

        ====================
        RULES:
            - Private facts are known but not shared unless asked
            - You may ask personal questions naturally.
            - You may only state facts that exist above.
            - You must NEVER invent facts.
            - If asked something not in memory, say "idk".
            - Do not narrate memory or say how you know things.
            - Do not reinterpret facts.
            - Keep replies casual and human.

        ====================
        IMPORTANT:
            If asked about name, bond, tone, or boundaries:
            → answer directly and confidently."""
    }

    # -------------------------
    # BUILD MESSAGE CONTEXT

    messages = [system_prompt] + session["conversation"][-MAX_CONTEXT_TURNS:]

    # -------------------------
    # TOKENIZE

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    attention_mask = torch.ones_like(input_ids)

    # -------------------------
    # DETECT FACT QUESTIONS

    FACT_QUESTIONS = [
    "what's my name",
    "what is my name",
    "do you know my name",
    "who am i",
    "what's your name",
    "what is your name"
    ]

    lower_msg = user_msg.lower()
    is_fact_question = any(q in lower_msg for q in FACT_QUESTIONS)

    # -------------------------
    # GENERATE REPLY

    early_turn = len(session["conversation"]) < 6

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=40 if is_fact_question else 120,
            temperature=0.0 if is_fact_question else 0.70,      # lower = fewer hallucinations
            top_p=1 if is_fact_question else 0.82,
            repetition_penalty=1.10,
            do_sample=False if is_fact_question else True,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output[0][input_ids.shape[-1]:]
    reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # -------------------------
    # SAVE ASSISTANT REPLY IF SAFE

    def is_safe_assistant_reply(text: str) -> bool:
        banned = [
            "lemme check",
            "wait",
            "i think",
            "maybe",
            "i guess",
            "i remember",
            "checking",
            "memory",
            "experiment",
        ]
        lower = text.lower()
        return not any(b in lower for b in banned)

    if is_safe_assistant_reply(reply):
        session["conversation"].append({
            "role": "assistant",
            "content": reply
        })

    save_session(session_id, session)

    return {"reply": reply}