#####################################################################
# Extract clinical related information from UMSLE-style questions.
#####################################################################

import re
import fire
import os
import json
from tqdm import tqdm
from . import shared
import logging

# Configure logging to print to console
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROMPT_EXTRACT_CLINICAL_INFORMATION = """You are a very strong clinician with broad medical expertise. You will be given a medical exam question. You must complete this task in TWO STRICT PHASES.
## PHASE 1 — EXTRACTION
Your job is to extract ONLY the patient-specific, clinically meaningful facts that are necessary to answer the question correctly.
Follow these rules EXACTLY:
1. Each list item must contain exactly ONE complete, standalone piece of information. If two facts are inseparable, combine them into ONE item.
2. ONLY extract information that describes:
   - the patient
   - symptoms or complaints
   - timing or course
   - physical exam findings
   - labs, imaging, or test results
   - relevant medical history, medications, exposures, or context
3. DO NOT extract any non-clinical or meta-level information, including but not limited to:
   - the task being asked (e.g. “the question asks for the most appropriate next step”)
   - the goal of the question (diagnosis, management, next step, prognosis, etc.)
   - instructions, stems, or selection guides
   - answer choice descriptions or summaries
   - references to “options,” “choices,” or letters (A)–(F)
   ❌ Examples of FORBIDDEN extractions:
   - “The question asks for the most appropriate next step in management.”
   - “Answer options include pulmonary function tests or CT scan.”
   - “Which of the following is the best treatment…”
4. ONLY extract facts that would remain true even if the question prompt and all answer choices were completely removed.
5. Each extracted item must be understandable on its own without needing the original question text.
6. Assign each item an importance weight from 0 to 10 based on how critical it is for answering the question correctly.
   - Add the weight using double square brackets: [[weight]]
7. After each item, include a brief justification for the weight.
   - Wrap the justification in double curly braces: {{explanation}}

!!!IMPORTANT!!!:
- Do NOT attempt to answer the question in this phase.
- Do NOT include reasoning about diagnoses, management, or next steps.
- Do NOT restate or paraphrase the question prompt itself.

## PHASE 2 — ANSWERING
After completing <conditions>, answer the question using ONLY the extracted items. Rules for answering:
1. You may ONLY use information explicitly listed in <conditions>. You may NOT introduce new facts or assumptions.
2. You MUST base your answer primarily on the highest-weighted items.
3. If multiple items conflict, prioritize those with higher weights.
4. Briefly justify your answer by referencing the most influential items by number and weight.
   - Do NOT introduce new information.
5. When selecting an answer, quote the EXACT text of the chosen option from the question, including its letter (e.g. “(D) Discontinuation of lisinopril”).
   - Wrap the quoted option in double square brackets.
6. Wrap your final answer between <correct_answer> and </correct_answer>

## OUTPUT FORMAT
<conditions>
1. ...
2. ...
</conditions>

<correct_answer>
[[(B) Myocarditis]]
Justification: brief justification referencing item numbers and weights.
</correct_answer>

## YOUR TURN
Question:
{question}
"""

def parse_clinical_conditions(text: str):
   """
   Parse clinical conditions with weights and explanations.
    
   Returns list of dicts with: condition, weight, explanation
   """
   results = []
    
   # Split into rows
   rows = text.strip().split('\n')
    
   for row in rows:
      if not row.strip():
         continue
            
      # Pattern to capture: number, item text, weight [[]], explanation {{}}
      pattern = r'^(\d+)\.\s+(.+?)\s+\[\[(\d+)\]\]\s+\{\{(.+?)\}\}$'
      match = re.match(pattern, row.strip())
        
      if match:
         results.append({
                'condition': match.group(2).strip(),
                'weight': int(match.group(3)),
                'explanation': match.group(4).strip()
         })
      else:
         logger.warning(f"Could not parse row: {row}")
   return results

def collect_conditions_with_weights(response_text: str):
   conditions_text = shared.extract_between_tags(response_text, '<conditions>', '</conditions>')
   if not conditions_text:
      logger.warning("No information found between <conditions> tags.")
      return []
   return parse_clinical_conditions(conditions_text)


def parse_answer_justification(text: str) -> dict:
   """
   Extract answer from [[]] and justification text.
    
   Returns dict with: answer, justification
   """
   # Extract answer from double square brackets
   answer_match = re.search(r'\[\[\s*(.+?)\s*\]\]', text)
   if answer_match:
      answer = answer_match.group(1).strip()
   else:
      answer = None
    
   # Extract justification - handle with or without "Justification:" label
   # First try to find text after "Justification:"
   justification_match = re.search(r'Justification:\s*(.+)', text, re.DOTALL)
    
   if justification_match:
      justification = justification_match.group(1).strip()
   else:
      # If no "Justification:" label, get everything after the [[answer]]
      after_answer = re.search(r'\]\]\s*(.+)', text, re.DOTALL)
      justification = after_answer.group(1).strip() if after_answer else None
   
   if answer is None or justification is None:
      logger.warning(f"Could not find complete answer or justification in text: {text}")
      return {}
   return {
        'answer': answer,
        'justification': justification
   }

def collect_final_answer_and_justification(response_text: str) -> dict:
   answer_text = shared.extract_between_tags(response_text, '<correct_answer>', '</correct_answer>')
   if not answer_text:
      logger.warning(f"No information found between <correct_answer> tags from {response_text}")
      return {}
   return parse_answer_justification(answer_text)

def verify_llm_answer(llm_answer: str, correct_answer: str) -> bool:
   # Normalize both answers by stripping whitespace, converting to lowercase, and remove all brackets
   def normalize_answer(ans: str) -> str:
      ans = ans.strip().lower()
      ans = re.sub(r'[\[\]\(\)]', '', ans)  # Remove all brackets
      return ans
   return normalize_answer(llm_answer) == normalize_answer(correct_answer)

def generate_and_extract_information(
      question: str,
      client: shared.ai.Client,
      model_name: str = shared.AI_FIREWORKS_MODEL,
):
   """Generate extraction prompt and extract information using LLM."""
   prompt = PROMPT_EXTRACT_CLINICAL_INFORMATION.format(question=question)
   response = shared.llm_generate(
      user_prompt=prompt,
      model_name=model_name,
      client=client,
      temperature=0.6,
      top_p=1,
      max_tokens=4090,
   )
   conditions = collect_conditions_with_weights(response)
   answer_info = collect_final_answer_and_justification(response)
   return conditions, answer_info

def extract_information(
        input_file: str,
        output_file: str,
        model_name: str = shared.AI_FIREWORKS_MODEL,
   ):
   """Extract clinical information from USMLE-style questions."""
   # Load questions
   if not os.path.exists(input_file):
      raise FileNotFoundError(f"Input file {input_file} does not exist.")
   
   with open(input_file, 'r') as f:
      questions = json.load(f)
   
   logger.info(f"Loaded {len(questions)} questions from {input_file}")

   # get client
   client = shared.get_client()

   # Loop through questions and extract information using LLM
   results = []
   failed_extraction_count = 0
   mismatched_answer_count = 0
   
   for idx, item in tqdm(enumerate(questions)):
      ques = item['question']
      conditions, answer_info = generate_and_extract_information(
         question=ques,
         client=client,
         model_name=model_name,
      )
      # If extraction incomplete, rerun
      extraction_failed = False
      if len(conditions) == 0 or len(answer_info) == 0:
         logger.warning(f"Incomplete extraction for question index {idx}. Rerunning...")
         # attempt 3 more times
         for attempt in range(3):
            conditions, answer_info = generate_and_extract_information(
               question=ques,
               client=client,
               model_name=model_name,
            )
            if len(conditions) > 0 and len(answer_info) > 0:
               logger.info(f"Successful extraction on attempt {attempt + 1} for question index {idx}.")
               break
         else:
            # All 3 attempts failed
            extraction_failed = True
            failed_extraction_count += 1
            logger.error(f"Failed extraction after 3 attempts for question index {idx}.")
      
      # If the final answer does not match the correct answer, try rerunning
      correct_answer = item.get('correct_response', '').strip()
      if not extraction_failed and answer_info.get("answer") is not None and correct_answer and not verify_llm_answer(answer_info['answer'], correct_answer):
         logger.warning(f"Mismatch answer for question index {idx}. Rerunning...")
         for attempt in range(3):
            conditions, answer_info = generate_and_extract_information(
               question=ques,
               client=client,
               model_name=model_name,
            )
            if answer_info.get('answer') is not None and verify_llm_answer(answer_info['answer'], correct_answer):
               logger.info(f"Successful answer match on attempt {attempt + 1} for question index {idx}.")
               break
         else:
            # All 3 attempts failed to match answer
            mismatched_answer_count += 1
            logger.error(f"Mismatched answer after 3 attempts for question index {idx}.")
         
      # add all extracted info to item
      if len(conditions) > 0 and len(answer_info) > 0 and verify_llm_answer(answer_info['answer'], correct_answer):
         item['llm_extracted_info'] = conditions
         item['llm_final_answer'] = answer_info
         results.append(item)
      if (idx + 1) % 5 == 0:
         logger.info(f"Processed {idx + 1} questions.")
   # Save results
   shared.save_json(results, output_file)
   # print summary
   logger.info("=" * 20 + " EXTRACTION SUMMARY " + "=" * 20)
   logger.info(f"Saved {len(results)} records to {output_file}")
   logger.info(f"Failed extractions after 3 attempts: {failed_extraction_count}")
   logger.info(f"Mismatched answers after 3 attempts: {mismatched_answer_count}")

# python -m pipeline.extract --input_file=data/cardiology_usmle_questions.json --output_file=data/cardiology_usmle_extracted_v1.json
if __name__ == "__main__":
   fire.Fire(extract_information)