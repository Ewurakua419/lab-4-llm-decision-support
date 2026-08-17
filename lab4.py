# %% [markdown]
# # Lab 4: LLMs and Prompt Engineering for Decision Support
# 
# **Duration:** 2 weeks [30 Jul - 13 Aug, 2026]
# **Due Date:** 13th August, 2026
# **Format:** Jupyter Notebook / Google Colab + external APIs + GitHub version control
# **Grading:** This is a graded lab.
# 
# **Student Name:** Ewurakua Amoah
# **Student ID:** 74492028
# 
# ---
# 
# ### Objective
# 
# In the previous labs you *trained* models. In this lab you will *use* a model that someone
# else spent millions of dollars training — a **Large Language Model (LLM)** — and learn that
# getting good results out of one is an engineering discipline of its own: **prompt
# engineering**.
# 
# You will build a **decision support system for a microfinance loan officer**. Given a pile of
# free-text loan application letters, your system will:
# 
# 1. **Summarize** each application into a short, factual brief,
# 2. **Extract** specific structured data points (JSON) that a downstream system could store,
# 3. Produce a **decision-support recommendation** — while keeping the human firmly in the loop.
# 
# Just as importantly, you will **evaluate** the LLM's output for quality, reliability, and
# appropriateness: Does it hallucinate? Is it consistent across runs? Should it be trusted to
# make the final call?
# 
# ---
# 
# ### Choosing an API provider
# 
# You need an LLM API with a **free tier**. Recommended options (pick ONE):
# 
# | Provider | Free tier | Notes |
# |---|---|---|
# | **Groq** (recommended) | Yes, generous | OpenAI-compatible API, very fast, open models (Llama) |
# | **Google Gemini** | Yes | `google-generativeai` package |
# | **Hugging Face Inference API** | Yes, limited | Many open models |
# | OpenAI / Anthropic | Paid | Fine if you already have credits |
# 
# The notebook's example code uses the **OpenAI-compatible chat format** (works with Groq and
# OpenAI directly; Gemini users adapt the call in one place). Everything else in the lab is
# provider-agnostic.

# %% [markdown]
# ---
# ### Part 0: Repository and API-key setup
# 
# 1. Create a **public** repository named `lab-4-llm-decision-support` and save this notebook
#    inside it.
# 2. Sign up with your chosen provider and create an **API key**.
# 3. **NEVER hard-code or commit your API key.** This is a graded requirement.
#    - Locally: put it in a `.env` file and add `.env` to `.gitignore`.
#    - Colab: use the Secrets panel (key icon) and read it with `google.colab.userdata`.
# 4. Add a `requirements.txt`: `openai python-dotenv pandas matplotlib`.
# 5. Commit and push after **each Part** — we will check for incremental commits.
# 
# > **A leaked key in your commit history = resubmission + penalty.** Keys can be scraped from
# > public repos within minutes.

# %%
# API-key setup — DO NOT hard-code your key in this cell.

import os

# --- Local (with a .env file) ---
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ["GROQ_API_KEY"]

# --- Google Colab (Secrets panel) ---
# from google.colab import userdata
# API_KEY = userdata.get("GROQ_API_KEY")

# TODO: set API_KEY using ONE of the methods above.

# OpenAI-compatible client (works for Groq and OpenAI; Gemini users see their docs):
from openai import OpenAI

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1",   # remove this line if using OpenAI itself
)
MODEL = "llama-3.3-70b-versatile"                # or your provider's model name

print("Client ready.")

# %% [markdown]
# ---
# # Section 1 — Talking to an LLM Programmatically
# 
# Before building anything, understand the anatomy of an API call: **messages and roles**
# (`system`, `user`, `assistant`), and the **generation parameters** (`temperature`,
# `max_tokens`).

# %% [markdown]
# ### Part 1.1 — Your first API call

# %%
# TODO: Write a helper function you will reuse for the WHOLE lab:
#
def ask_llm(user_prompt, system_prompt="You are a helpful assistant.",
             temperature=0.7, max_tokens=500):
     response = client.chat.completions.create(
         model=MODEL,
         messages=[
             {"role": "system", "content": system_prompt},
             {"role": "user",   "content": user_prompt},
         ],
         temperature=temperature,
         max_tokens=max_tokens,
     )
     
     return response.choices[0].message.content,response.usage
#
# TODO: Call it once with a simple question and print the answer.
a,b=ask_llm(user_prompt="how many countries are there in africa?")
print(a)

print(f"Usage: {b}")
# TODO: Print response.usage as well — how many tokens did your call consume?


# %% [markdown]
# **Student Reasoning — Anatomy of a call**
# *1. What is the difference between the `system` and `user` roles? Give an example of
# something that belongs in each.*
# *2. What is a token, roughly? Why do API providers bill per token rather than per request?*
# 
# > **Answer:** [Double-click to edit]
# > 1. the user role is the human input being requested while the system role is the more foundational input that guides the llm's answer
# > a system role can have a prompt where it says "keep under 500 words"
# > user role would have, "what is the nasme of the tallest mountain in the world?"
# > 2. Token is the basic unit of data a model uses. The token is the budget as requests can use a varying amount of tokens so charging by token makes sure that the cost matches the ammount of proccessing power used.

# %% [markdown]
# ### Part 1.2 — Temperature: the randomness dial

# %%
# TODO: Ask the SAME question 5 times at temperature=0.0 and 5 times at temperature=1.2.
#   A good test question: "Suggest a name for a savings product for market traders in Accra."

a1,usage1=(ask_llm(temperature=0.0, user_prompt="Suggest a name for a savings product for market traders in Accra"))
a2,usage2=(ask_llm(temperature=0.0, user_prompt="Suggest a name for a savings product for market traders in Accra"))
a3,usage3=(ask_llm(temperature=0.0, user_prompt="Suggest a name for a savings product for market traders in Accra"))
a4,usage4=(ask_llm(temperature=0.0, user_prompt="Suggest a name for a savings product for market traders in Accra"))
a5,usage5=(ask_llm(temperature=0.0, user_prompt="Suggest a name for a savings product for market traders in Accra"))


b1,usage6=(ask_llm(temperature=1.2, user_prompt="Suggest a name for a savings product for market traders in Accra"))
b2,usage7=(ask_llm(temperature=1.2, user_prompt="Suggest a name for a savings product for market traders in Accra"))
b3, usage8=(ask_llm(temperature=1.2, user_prompt="Suggest a name for a savings product for market traders in Accra"))
b4,usage9=(ask_llm(temperature=1.2, user_prompt="Suggest a name for a savings product for market traders in Accra"))
b5,usage10=(ask_llm(temperature=1.2, user_prompt="Suggest a name for a savings product for market traders in Accra"))

# TODO: Print all 10 answers, grouped by temperature.
print("Temp: 0.0")
print(a1)
print(" ")
print(a2)
print(" ")
print(a3)
print(" ")
print(a4)
print(" ")
print(a5)
print(" ")

print("Temp: 1.2")
print(b1)
print(" ")
print(b2)
print(" ")
print(b3)
print(" ")
print(b4)
print(" ")
print(b5)
print(" ")

# %% [markdown]
# **Student Reasoning — Temperature**
# *What did you observe at each temperature? For the loan decision-support system you are about
# to build, which temperature regime is appropriate, and why?*
# 
# > **Answer:** [Double-click to edit]
# > at a higher temperature it gave longer and more detailed answers.
# > I think i would use the higher temperature

# %% [markdown]
# ---
# # Section 2 — The Dataset: Loan Application Letters
# 
# Run the next cell to load **six loan application letters** submitted to a (fictional)
# microfinance institution in Ghana, plus **gold-standard extraction labels** for three of them
# (you will use these for evaluation in Section 4).
# 
# Read at least two letters fully before moving on — you cannot engineer prompts for text you
# have not read.

# %%
LETTERS = {
"L001": """Dear Sir/Madam,
My name is Akosua Mensah and I have been selling provisions at Makola Market for 12 years.
I am applying for a loan of GHS 8,000 to buy a deep freezer and expand into frozen foods.
My current stall makes about GHS 900 profit each month. I have saved GHS 2,500 with your
susu scheme over the past two years and I have never missed a contribution. I can repay
GHS 450 monthly over 20 months. My sister, a teacher, will stand as my guarantor.
Thank you for considering my application.""",

"L002": """Hello,
I am Kwame Boateng, a commercial driver in Kumasi. I need GHS 25,000 urgently to repair my
trotro engine and settle some personal debts. Business has been slow but it will surely
pick up after the festive season. I can pay back whenever the money comes. I do not have
collateral at the moment but God willing everything will be fine. Please help me quickly.""",

"L003": """Dear Loan Committee,
I am Efua Darko, owner of Darko Fashions, a registered dressmaking business in Takoradi
(registration no. BN-2019-4482). I employ three apprentices. I request GHS 15,000 to
purchase two industrial sewing machines and fabric stock ahead of the Christmas season.
Last year my December revenue alone was GHS 22,000; monthly profit averages GHS 2,800.
I hold a fixed deposit of GHS 5,000 with GCB which I can pledge. Proposed repayment:
GHS 1,100 monthly for 15 months. Attached are my sales records for the past 18 months.""",

"L004": """Good day,
My name is Yaw Owusu. I want a loan for my poultry farm at Nsawam. The amount is GHS 12,000
for feed and 500 new layers. I started the farm last year. Sometimes I make good money,
around GHS 1,500 in a good month, but bird flu affected us in March and I lost many birds.
I am rebuilding now. I can repay in 18 months. My uncle has agreed to guarantee the loan
with his taxi.""",

"L005": """Dear Manager,
I am writing on behalf of the Adenta Women's Weaving Cooperative (14 members). We seek
GHS 30,000 to buy a bulk order of yarn directly from the factory, cutting out middlemen and
raising our margins from 15% to about 35%. The cooperative has operated for 6 years and
holds GHS 9,000 in our group account. We propose repayment of GHS 2,000 monthly over
16 months, backed by our group savings and joint liability agreement.""",

"L006": """Hi,
This is Kofi. I saw your advert. I want GHS 50,000 to start a car washing business, a
provision shop, and also import phones from Dubai. I am 22 and full of energy. I have not
started any of these yet but my friends say I am very business minded. I will pay back in
one year when the businesses are booming. No collateral but I am trustworthy.""",
}

# Gold-standard labels for three letters (for Section 4 evaluation):
GOLD = {
  "L001": {"applicant_name": "Akosua Mensah", "amount_ghs": 8000,  "purpose": "buy deep freezer / expand into frozen foods",
           "monthly_profit_ghs": 900,  "has_collateral_or_guarantor": True,  "repayment_months": 20},
  "L003": {"applicant_name": "Efua Darko",    "amount_ghs": 15000, "purpose": "industrial sewing machines and fabric stock",
           "monthly_profit_ghs": 2800, "has_collateral_or_guarantor": True,  "repayment_months": 15},
  "L006": {"applicant_name": "Kofi",          "amount_ghs": 50000, "purpose": "car wash, provision shop, phone imports",
           "monthly_profit_ghs": None, "has_collateral_or_guarantor": False, "repayment_months": 12},
}

print(f"{len(LETTERS)} letters loaded.")

# %% [markdown]
# ---
# # Section 3 — Prompt Engineering for the Decision Support System
# 
# You will now build the three components of the system, iterating on your prompts as you go.
# **Keep every major prompt version** — Section 3.4 asks you to commit your prompt templates
# and document how they evolved.

# %% [markdown]
# ### Part 3.1 — Component 1: Summarization
# Turn a rambling letter into a 3-4 sentence factual brief a busy loan officer can scan.

# %%
# TODO: Write SUMMARY_PROMPT_V1 — your first, naive attempt (e.g. just "Summarize this:").
#   Run it on L002 and L006. Read the output critically.
SUMMARY_PROMPT_V1=f"summarize this:"
V1l002, V1Ul002=ask_llm(user_prompt=f"{SUMMARY_PROMPT_V1} {LETTERS['L002']}")
V1l006, V1Ul006=ask_llm(user_prompt=f"{SUMMARY_PROMPT_V1} {LETTERS['L006']}")

print(" ")
print("============ First attempt: SUMMARY_PROMPT_V1 ==============")
print(" ")
print("=========== Letter: l002 ============")
print(" ")
print(V1l002)
print(" ")
print(" ================= Letter: l006 ===========")
print(" ")
print(V1l006)
# TODO: Now write SUMMARY_PROMPT_V2 as a proper template with:
#   - a system prompt giving the LLM a ROLE (e.g. "You are an assistant to a microfinance
#     loan officer...") and constraints (factual, neutral, no invented details, 3-4 sentences)
#   - a user prompt template like: f"Summarize this loan application:\n\n{letter_text}"
#   Run V2 on the same two letters at temperature=0.
SUMMARY_PROMPT_V2={"system_p":f"""You are an assistant to a micro finance loan officer,
 ensure that you are factual, neutral, no invented details
 Use between 3-4 sentences.""", "user_p": f"Summarize this loan application"}

V2l002, V2Ul002=ask_llm(system_prompt=SUMMARY_PROMPT_V2["system_p"], user_prompt=f"{SUMMARY_PROMPT_V2['user_p']} {LETTERS['L002']}",temperature=0)
V2l006, V2Ul006=ask_llm(system_prompt=SUMMARY_PROMPT_V2["system_p"], user_prompt=f"{SUMMARY_PROMPT_V2['user_p']} {LETTERS['L006']}",temperature=0)
print(" ")
print("========= Second attempt: SUMMARY_PROMPT_V2 ===========")
print(" ")
print("====== Letter: l002 ========")
print(" ")
print(V2l002)

print(" ")
print("======== Letter:l006 ==========")
print(" ")
print(V2l006)

# TODO: Compare V1 vs V2 outputs side by side. Keep both prompt versions in this notebook.

# %% [markdown]
# **Student Reasoning — Summarization prompts**
# *1. What concrete problems did V1's output have that V2 fixed? Quote examples.*
# *2. Why is "no invented details" an essential instruction in this application? What is this
# failure mode called in the LLM literature?*
# 
# > **Answer:** [Double-click to edit]
# >
# > 1.V1's output was not professional and was rather general and not tailored to the use case, for example L002  did not have collateral but the model did not explicitly state that rather it summarized it generally, however a loan officer would be looking to see if there are items such as collateral which the applicant can provide, however in V2; it is explicitly mentioned that there is no collateral and even suggests that the loan officer performs extra checks. 
# >
# > 2. This ensures that the model does not add any details due to any commonality it has realized in other texts, such as saying that l002 is not trustworthy because they do not have a paymenyt plan.
# > This faliure mode is called bias

# %% [markdown]
# ### Part 3.2 — Component 2: Structured extraction (JSON)
# Downstream software cannot read prose. Extract the fields in `GOLD` as strict JSON.

# %%
# TODO: Write EXTRACT_PROMPT — a template that instructs the model to return ONLY a JSON
#   object with EXACTLY these keys:
#     applicant_name (string), amount_ghs (number), purpose (string),
#     monthly_profit_ghs (number or null), has_collateral_or_guarantor (boolean),
#     repayment_months (number or null)
#   Techniques to use:
#     - explicit schema in the prompt
#     - ONE worked example (few-shot) using a letter you write yourself (not from LETTERS!)
#     - "If a field is not stated in the letter, use null. Do not guess."
#     - temperature=0
import json
import pandas as pd

EXTRACT_PROMPT={"sys_prompt":""" Your task is to extract information from the  text given at user_prompt, not example input. 
Format everything as a JSON object with EXACTLY these keys:
Schema(required field with types):
    "applicant_name": "string", 
    "amount_ghs": "number", 
    "purpose": "string",
    "monthly_profit_ghs":"number" or null, 
    "has_collateral_or_guarantor" : "boolean",
    "repayment_months" :"number" or null

    Techniques to use:

    Example input:
    Hello my name is Ama Forson and I would like GHS 2000 to build a new vegetable shop.
    Currently I make GHS 1200 monthly. I do not have any collateral but I can pay you back in 25 months.


    Perfect example output for the exapmle input:
    {
    "applicant_name": "Ama Forson",
    "amount_ghs":2000,
    "purpose": "To build a new vegetable shop",
    "monthly_profit_ghs":1200 , 
    "has_collateral_or_guarantor": False,
    "repayment_months": 25 },

    
   
    
    if field is not stated in data, use null, do not guess """}

# TODO: Write extract_fields(letter_text) that calls the LLM, strips any ```json fences,
#   json.loads() the result, and returns a dict. Handle parse failures gracefully
#   (return None and print a warning).
def extract_fields(letter_text,temp=0.7):
    result,u_result=ask_llm(user_prompt=f"Process this text denoted by backticks ```{letter_text}`` strip any ```json fences", system_prompt=EXTRACT_PROMPT["sys_prompt"],temperature=0) 
    try:
        r_load=json.loads(result)
    except json.JSONDecodeError as e:
        r_load=None
        print("None value in JSON load")
    return r_load
dict_list=[]
for letter in LETTERS:
    dict_list.append(extract_fields(letter_text=LETTERS[letter]))

letters_df=pd.DataFrame(dict_list)
print(letters_df)
# TODO: Run it on ALL SIX letters; collect results into a pandas DataFrame (one row per
#   letter) and display it.

# %% [markdown]
# **Student Reasoning — Structured extraction**
# *1. Why must the few-shot example NOT come from the six letters you are processing?*
# *2. Why "use null, do not guess" — what did the model do without that instruction?*
# *3. Why is temperature=0 the right choice for extraction but arguably not for creative tasks?*
# 
# > **Answer:** [Double-click to edit]
# >
# > 1. It would have caused the model to memorize not learn ftom the pattern we were trying to define, this would lead to struggles with new inputs
# >
# > 2. it can hallucinate or speculate based on information it already knows
# > 3. temperature=0 gives the basic info which is helpful when we just need facts without too muv=ch detail. For creative tasks, more details are preferred so a higher temp is favorable

# %% [markdown]
# ### Part 3.3 — Component 3: The decision-support brief
# Combine everything: for each letter, produce a recommendation brief for the loan officer —
# strengths, risks, missing information, and a suggested next step. The system must
# **support** the decision, not **make** it.

# %%
# TODO: Write BRIEF_PROMPT — it receives the letter AND your extracted JSON, and must output:
#     1. Strengths (bullet points, grounded in the letter)
#     2. Risks / red flags (bullet points)
#     3. Missing information the officer should request
#     4. Suggested next step (e.g. "invite for interview", "request documents",
#        "flag for senior review") — NOT "approve" or "reject".
#   Give the model an explicit instruction that final decisions are made by humans.
BRIEF_PROMPT={"user_p":"""Take this letter and JSON output and give a brief """,
              "sys_p": """You are the assistant to a loan officer for a microfinance company, final decisions are made by humans, take in a letter and an extracted JSON and output:
1. Strengths (bullet points, grounded in the letter)
   2. Risks / red flags (bullet points)
    3. Missing information the officer should request
   4. Suggested next step (e.g. "invite for interview", "request documents", "flag for senior review") — NOT "approve" or "reject". ensure that it is clear and straightforward"""}#get clarification on what exactly counts as sys prompt and which is user prompt
# TODO: Generate briefs for ALL SIX letters. Print the briefs for L001, L002, and L006 —
#   three very different applications.
i=0
briefs=[]
for letter in LETTERS:
   letter_r, letter_u=ask_llm(user_prompt=f"{BRIEF_PROMPT['user_p']} letter: {LETTERS[letter]}, JSON: {dict_list[i]}", system_prompt=BRIEF_PROMPT["sys_p"])
   briefs.append(letter_r)

   i+=1
print("====Briefs====")
print("")
print("")
print("===L001===")
print("")
print(briefs[0])
print("")
print("")
print("===L002===")
print("")
print(briefs[1])
print("")
print("")
print("===L006===")
print("")
print(briefs[5])
print("")
print("")
print("===L003===")
print("")
print(briefs[2])
print("")
print("")

# %% [markdown]
# **Student Reasoning — Decision support**
# *1. Compare the briefs for L003 (strong application) and L006 (weak application). Did the
# system identify the right strengths and red flags in each?*
# *2. Why did we forbid the model from outputting "approve"/"reject"? Give one practical and
# one ethical reason.*
# 
# > **Answer:** [Double-click to edit]
# > 1.  The system identified the right sreengths for L003 highlighting her clear plan and transpareny. It also mentions good red flags
# > For L006 its strengths were very optimistic. Its weaknesses were quite accurate as well
# >
# > 2. This is because it may decide to be more biased against weaker applicants who might just not know how to appropriately present themselves. In practice, some

# %% [markdown]
# ### Part 3.4 — Commit your prompt templates
# Prompts ARE code. Save your final `SUMMARY_PROMPT`, `EXTRACT_PROMPT`, and `BRIEF_PROMPT` into
# a separate file `prompts.py` (or `prompts.md`) in your repository and commit it with a
# message describing how the prompts evolved. Paste your commit hash below.
# 
# > **Commit hash:** [paste here]
# > f58a89f9d267a444b2c45efa03235051bfc225a7

# %% [markdown]
# ---
# # Section 4 — Evaluation: Quality, Reliability, Appropriateness
# 
# An impressive demo is not a trustworthy system. Now measure it.

# %% [markdown]
# ### Part 4.1 — Extraction accuracy against gold labels

# %%
# TODO: For the three letters in GOLD, compare your extracted DataFrame to the gold values
#   field by field. Compute per-field accuracy across the three letters
#   (name matching can be case-insensitive; numbers must match exactly).
index=[0,2,5]
letternames=["L001",
    "L002",
    "L006"]

data={"field":["applicant_name",
    "amount_ghs",
    "purpose",
    "monthly_profit_ghs", 
    "has_collateral_or_guarantor" ,
    "repayment_months"],

    "L001":[],
    "L002":[],
    "L006":[],
    "Accuracy":[]
    
    }

i=0
for key in GOLD.keys():# move per letter
    match=False
    accuracy=0
    for field in GOLD[key]:#per field in each letter
        match=False
        checkrow= letters_df.iloc[index[i]]
        checkfeild=checkrow[field]
        if field=="applicant_name" or field== "purpose":
            check= str(checkfeild).lower().strip()
            if check== str(GOLD[key][field]).lower().strip():
                match=True
                

        elif checkfeild==GOLD[key][field]:
                match=True
                


        data[letternames[i]].append(match)
    
    i+=1#iterate through both index and letternames
    if i==len(index):
        break

for key in range(len(data["field"])):
    if data["L001"][key]==False:
        data["Accuracy"].append(0)
    elif data["L002"][key]==False:
        data["Accuracy"].append(0)
    elif data["L006"][key]==False:
        data["Accuracy"].append(0)

    else:
        data["Accuracy"].append(1)
     

df=pd.DataFrame(data)
df


    
    
# TODO: Display a small table: rows = fields, columns = L001 / L003 / L006 / accuracy.

# %% [markdown]
# ### Part 4.2 — Reliability: is the system consistent?

# %%
# TODO: Run extract_fields() on letter L004 FIVE times at temperature=0 and FIVE times at
#   temperature=1.0.
valid_count0=0
identical_count0=0
result_temp0=[]
for i in range(5):
    result=extract_fields(LETTERS['L004'], temp=0)
    result=json.dumps(result, sort_keys=True)
    result_temp0.append(result)
    if result!=None:
        valid_count0+=1

unique0=set(result_temp0)#create a set of unique values
for unique in unique0:
    for k in range(len(result_temp0)):
        if unique==result_temp0[k]:
            identical_count0+=1
identical_count0=len(result_temp0)-len(unique0)

valid_count1=0
result_temp1=[]
identical_count1=0
for i in range(5):
    result=extract_fields(LETTERS['L004'], temp=1.0)
    result=json.dumps(result, sort_keys=True)
    result_temp1.append(result)
    if result!=None:
        valid_count1+=1

unique1=set(result_temp1)#create a set of unique values
for unique in unique0:
    for k in range(len(result_temp1)):
        if unique==result_temp1[k]:
            identical_count1+=1

print(f"Out of the 5 runs at temperature = 0 , {valid_count0} produced a valid JSON")
print(f"Out of the 5 runs at temperature = 0 , {identical_count0} were identical")

print("")
print(f"Out of the 5 runs at temperature = 1 , {valid_count1} produced a valid JSON")
print(f"Out of the 5 runs at temperature = 1 , {identical_count1} were identical")
# TODO: For each temperature, report how many of the 5 runs produced (a) valid JSON and
#   (b) identical values across runs. A simple approach: json.dumps(result, sort_keys=True)
#   and count unique strings.

# %% [markdown]
# ### Part 4.3 — Hallucination probing

# %%
# TODO: Design TWO adversarial tests and run them:
#   Test 1 — Ask your summarizer a question about a detail that is NOT in a letter
#     (e.g. "What is the applicant's credit score?"). Does it admit the information is
#     absent, or does it invent one?
#   Test 2 — Feed your extractor an EMPTY or IRRELEVANT text (e.g. a weather report).
#     Does it return nulls, or does it fabricate an applicant?
test1, usage1test=ask_llm(system_prompt=SUMMARY_PROMPT_V2['system_p'], user_prompt=f" letter is {LETTERS['L001'] } what is their favorite color")
print("===TEST 1===")
print(test1)#pass

test2_test=" The. itsy bitsy spider went up the water spout down came the rain and washed the spider out, out came the sun and dried up all the rain so the itsy bitsy spider climbed up the spout again"
test2=extract_fields(test2_test)
print("===TEST 2===")
print(test2)
# TODO: Record the outputs verbatim below and label each PASS or FAIL.
#output
#test 1- pass
#test 2-[pass]
#There is no information provided about Akosua Mensah's favorite color in the given text. The letter is focused on her loan application and provides details about her business and repayment plan. The loan officer will likely review her application based on her business proposal and financial information. The decision to approve or reject the loan will be based on her creditworthiness and ability to repay the loan.
#Expecting value: line 1 column 1 (char 0)
#===TEST 2===
#None

# %% [markdown]
# **Student Reasoning — Evaluation results**
# *1. Report your extraction accuracy. Which field was hardest for the model and why?*
# *2. What did the reliability experiment show about temperature and production systems?*
# *3. Did your system hallucinate under probing? If yes, how could the prompt (or the system
# design around it) reduce the risk?*
# 
# > **Answer:** [Double-click to edit]
# > 1. The hardest field was the purpose field. I believe it was because that could be summarized many different ways, so the main concept might be the same but it can be rewrittem in many ways.
# > 
# > 2. The production system was more consistent at a higher temperature, this showed that systems are more reliable at higher temperatures. This does not mean that the results are accurate but it will reproduce the same information
# >
# > 3. It did not hallucinate, it returned a proper error message for the first test and for the second, did not return anything as the promt had nothing to extract.

# %% [markdown]
# ### Part 4.4 — Appropriateness: should this system exist?
# No code in this part — just judgment, which is the scarcest skill in AI for business.

# %% [markdown]
# **Student Reasoning — Appropriateness**
# *1. Letters L002 and L006 would likely be declined. If the bank fully automated decisions
# with your system, who could be unfairly harmed, and how? Consider applicants who write
# poorly in English but run solid businesses.*
# *2. Loan letters contain personal data. What are the implications of sending them to a
# third-party API in another country? What would you check before deploying this at a real
# Ghanaian microfinance institution?*
# *3. Name TWO concrete safeguards you would build around this system in production (think:
# human review points, logging, appeal processes, monitoring).*
# 
# > **Answer:** [Double-click to edit]
# > 1. People who have no experience in formal loan applications would be greatly affected. This would be very detrimental as some potential borrowers might fit the criteria but might not be knowledgable on how to structure or fill a proper loan form. People starting out would also be negatively affected as they might not know beforehand to have collateral but if informed they could figure a way to get some. Some new people might have the right idea and direction but would be affected as they do not currently have anything to show for it, discouraging new enterprises.
# >
# > 2. If you send the loan letters to foreign API's personal information like bank statements, private identifying information can be leaked or sold by the foreign company and be used for malicious reasons. You have to checkthe data privay policy of the host company and the data laws of the host company's country.
# >
# > 3. Content moderation filters, which can be used to scan input and catch any information that contains hate speech, or illegal requests. This would cacth any weird information that a person enters in their loan letter before it gets to the API. 
# > 
# > Appeal options where rejected applicant would have an option to request a human review.

# %% [markdown]
# ---
# # Section 5 — Reflection
# 
# *Answer in a few sentences each:*
# 
# 1. **Prompting as engineering:** How is iterating on a prompt similar to and different from
#    iterating on the model hyperparameters you tuned in Lab 3?
# 2. **Trust:** After your Section 4 evaluation, would you trust this system to run unattended?
#    What single evaluation result most influenced your answer?
# 3. **Cost and scale:** Estimate (from your `response.usage` numbers) the tokens needed to
#    process 1,000 applications per month. What does that imply for provider choice?
# 4. **Looking back at the course:** You have now used classical ML (Lab 2), trained neural
#    networks (Lab 3), and used a foundation model via API (Lab 4). For a task like this one,
#    why does calling an API beat training your own model — and when would it not?
# 
# > **Answer:** [Double-click to edit]
# >
# > 1. Iterating a prompt also helps you test the consistency of the output, like iterating on model hyperparameters which helps you get the right temperature to get the best answers. Iterating a prompt adjusts the response of an already existing model but for the hyper parameters it is training the model to fix errors. 
# >
# > 2. I would trust teh system after the hallucination tests. It showed me that if the wrong information or prompt is given it could give an appropriate error message and flag for human review. 
# >
# > 3. It should take about 485 * 1000 which is about 485000 tokens monthly. This means that we need a provider with a good fee structure whether it is a subscription, or based on the load used monthly.
# >
# > 4. Calling an API is great when we do not have a lot of training data and the information coming in is generic not custom, since most loan letters are not common or publicly available it might be better to use an API. However if you needed a highly customized model it would be better to use a self trained network.

# %% [markdown]
# ---
# ### Submission checklist
# 
# - [ ] All cells run top-to-bottom with no errors (`Kernel -> Restart & Run All`).
# - [ ] **No API key anywhere in the notebook or the commit history.**
# - [ ] Every **Student Reasoning** box is filled in with full sentences.
# - [ ] `prompts.py` / `prompts.md` committed with your final prompt templates.
# - [ ] Evaluation tables and adversarial test outputs visible in the saved notebook.
# - [ ] Notebook pushed to `lab-4-llm-decision-support` with incremental commits.
# - [ ] Repository link submitted to the course portal.
# - [ ] AI Declaration form in Repository.


