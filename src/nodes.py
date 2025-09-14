"""Node functions for Question Generation (Q-Gen) and Answer Generation (A-Gen).

This module splits the original monolithic prompt into two focused prompts:
1. Q-Gen: Generates stage-specific questions when USER_ANSWER is empty
2. A-Gen: Generates checklists and next questions when USER_ANSWER is provided

Contains all prompt templates and stage definitions in one place.
"""

import json
import random
from typing import Dict, List, Any, Optional

# ------ Context Engineering Functions ------

def load_scenario_examples() -> Dict[str, Dict]:
    """Load scenario examples for context engineering"""
    try:
        with open('data/scenario_examples.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load scenario examples: {e}")
        return {}

def get_scenario_context(state: str, scenario: str) -> str:
    """Generate context-enhanced scenario using few-shot examples"""
    examples_db = load_scenario_examples()
    
    if state not in examples_db:
        return f"SCENARIO (focus): {scenario}"
    
    # Find most relevant example based on scenario keywords
    scenario_lower = scenario.lower()
    stage_examples = examples_db[state]  # Now directly a list
    
    # Score examples based on keyword overlap
    scored_examples = []
    for example in stage_examples:
        example_scenario = example['scenario'].lower()
        # Simple keyword matching
        overlap = sum(1 for word in scenario_lower.split() if word in example_scenario)
        scored_examples.append((overlap, example))
    
    # Get top 3 most relevant examples (sort by overlap score)
    scored_examples.sort(key=lambda x: x[0], reverse=True)
    relevant_examples = [ex[1] for ex in scored_examples[:3]]
    
    # Build context with examples
    context = f"SCENARIO (focus): {scenario}\n\n"
    context += f"STAGE: {state}\n\n"
    context += "RELEVANT EXAMPLES:\n"
    
    for i, example in enumerate(relevant_examples, 1):
        context += f"{i}. Scenario: '{example['scenario']}' → Question: '{example['question']}'\n"
        context += f"   Reasoning: {example['reasoning']}\n\n"
    
    return context

# ------ Stage Definitions ------

STAGE_DEFINITIONS = {
    "M": "MASSIVE HEMORRHAGE CONTROL: Locate any life-threatening bleeding (extremity, junctional, axial) and decide if tourniquet, packing + direct pressure are required. Focus ONLY on bleeding control methods – NEVER ask about chest seals, occlusive dressings, or respiratory concerns.",
    "A": "AIRWAY MANAGEMENT: Confirm if airway is patent or obstructed; decide on suction, positioning or airway adjunct. Focus ONLY on airway patency – do NOT ask about bleeding, circulation, or other concerns.",
    "R": "RESPIRATION ASSESSMENT: Assess breathing and ventilation; identify any respiratory problems or breathing difficulties. Focus ONLY on breathing assessment – do NOT ask about bleeding, airway, or circulation.",
    "C": "CIRCULATION ASSESSMENT: Assess perfusion (pulse quality, skin colour/temp, mental state) and decide if fluids or blood products are required. Focus ONLY on circulation and shock – do NOT ask about airway, breathing, or bleeding.",
    "H": "HYPOTHERMIA PREVENTION: Prevent heat loss and screen for head injury or altered consciousness. Focus ONLY on temperature and head injury – do NOT ask about other vital signs.",
    "P": "PAIN MANAGEMENT: Assess pain level and decide if analgesia is required. Focus ONLY on pain assessment and analgesia – do NOT ask about other medical concerns.",
    "A2": "ANTIBIOTICS: Determine if a penetrating wound mandates antibiotics and note allergy status. Focus ONLY on antibiotic needs – do NOT ask about other treatments.",
    "W": "WOUND REASSESSMENT: Re-inspect the casualty for missed injuries, burns or eviscerations; verify all dressings and seals. Focus ONLY on wound inspection – do NOT ask about other assessments.",
    "S": "SPLINTING: Identify fractures/dislocations, immobilise, and re-check distal pulse after splinting. Focus ONLY on fracture management – do NOT ask about other injuries."
}

# Q-Gen System Prompt (Question Generation Only)
Q_SYS = (
    "You are a clinical assistant generating MARCH-PAWS protocol questions. You MUST:\n"
    "- PRIMARY FOCUS: Use the STATE_DEFINITION to understand what this stage requires\n"
    "- SECONDARY FOCUS: Use the SCENARIO for context about the patient's condition\n"
    "- Generate ONLY a concise, stage-specific question for the CURRENT_STATE\n"
    "- Ask for observed facts/signs/symptoms — not plans or actions\n"
    "- Avoid multi-part 'if ... then ...' questions; ask a single, concise question\n"
    "- Prefer yes/no or short factual answers (e.g., presence/absence, quantity, location)\n"
    "- NEVER ask 'Is it necessary to...' or 'Should you...' - these are procedural questions\n"
    "- Ask about OBSERVABLE CONDITIONS, not treatment decisions\n"
    "- Focus on what can be OBSERVED or MEASURED in this stage\n"
    "- Examples of GOOD questions: 'Is there active bleeding?', 'Is the airway clear?', 'Is the casualty breathing normally?'\n"
    "- Examples of BAD questions: 'Is it necessary to pack the wound?', 'Should you apply a tourniquet?', 'What actions are required?'\n"
    "\n"
    "QUESTION GENERATION PROCESS:\n"
    "1. READ the STATE_DEFINITION carefully - what does this stage focus on?\n"
    "2. CONSIDER the SCENARIO - what context is relevant?\n"
    "3. GENERATE a question that asks about observable conditions for this stage\n"
    "\n"
    "Return ONLY a simple question string, no JSON, no additional text."
)

# Q-Gen User Template
Q_USER = (
    "CURRENT_STATE: {state}\n"
    "STATE_DEFINITION: {state_definition}\n"
    "SCENARIO (focus): {scenario}\n\n"
    "Generate a stage-specific question for {state} focusing ONLY on {state} requirements.\n"
    "Consider the scenario context when generating the question."
)

# A-Gen System Prompt (Answer Generation with Checklist)
A_SYS = (
    "You are a clinical assistant following the MARCH-PAWS protocol. You MUST:\n"
    "- Only use the provided EXCERPTS to answer\n"
    "- If the excerpts don't support an answer, reply EXACTLY: I cannot advise — no relevant content found\n"
    "- If you reply with that refusal line, do not include any other text, items, or citations\n"
    "- PRIMARY FOCUS: Analyze the USER_RESPONSE in detail\n"
    "- SECONDARY FOCUS: Match user's response with retrieved content\n"
    "- CONTEXT: Use scenario and stage definition for understanding\n"
    "- OUTPUT: Generate actionable clinical advice based on what user reported\n"
    "- Do NOT generate questions - that's handled separately\n"
    "\n"
    "CLINICAL ADVICE PROCESS:\n"
    "1. READ the USER_RESPONSE carefully - what exactly did they say?\n"
    "2. FIND matching content in EXCERPTS that relates to what they reported\n"
    "3. COMBINE user's report + retrieved content + stage requirements\n"
    "4. GENERATE specific clinical actions based on this combination\n"
    "\n"
    "STAGE FOCUS:\n"
    "- Focus on the current stage's specific requirements\n"
    "- Use the provided excerpts to generate relevant clinical advice\n"
    "- Ensure citations are appropriate for the current stage context\n"
    "\n"
    "ADVICE FORMAT:\n"
    "- Create 2-4 specific checklist items based on user's response\n"
    "- Each checklist item should be actionable and specific\n"
    "- Include relevant citations for each item\n"
    "- Focus on what the user reported and what actions are needed\n"
    "\n"
    "CHECKLIST EXAMPLES:\n"
    "  * User: 'Yes, there is active bleeding from chest' → ['Apply direct pressure to chest wound', 'Monitor for signs of shock']\n"
    "  * User: 'Airway is clear, patient speaking' → ['Confirm airway remains patent', 'Continue monitoring airway status']\n"
    "  * User: 'Pulse 110 bpm, weak, skin pale' → ['Establish IV access immediately', 'Begin fluid resuscitation', 'Monitor vital signs closely']\n"
    "  * User: 'No bleeding observed' → ['No life-threatening bleeding detected - continue to airway assessment', 'Continue MARCH-PAWS protocol']\n"
    "\n"
    "\n"
    "CITATIONS: Use exact paragraph citations like [ChX §Y-Z, p.X–Y]\n"
    "Return strict JSON with keys: checklist (list of strings), citations (list of strings)."
)

# A-Gen User Template
A_USER = (
    "CURRENT_STATE: {state}\n"
    "STATE_DEFINITION: {state_definition}\n"
    "SCENARIO: {scenario}\n"
    "USER_ANSWER: {user_answer}\n\n"
    "REFERENCE CATALOG:\n"
    "{catalog}\n\n"
    "EXCERPTS:\n"
    "{excerpts}\n\n"
    "INSTRUCTIONS:\n"
    "- PRIMARY FOCUS: Read the USER_ANSWER carefully - what exactly did they report?\n"
    "- SECONDARY FOCUS: Find matching content in EXCERPTS that relates to what they reported\n"
    "- CONTEXT: Use SCENARIO and STATE_DEFINITION for understanding\n"
    "- OUTPUT: Generate clinical advice based on user's response + retrieved content\n"
    "- Do NOT generate questions - that's handled separately\n"
    "- Use exact paragraph citations like [ChX §Y-Z, p.X–Y]."
)


def build_q_prompt(state: str, scenario: str) -> str:
    """Build Q-Gen prompt for question generation with context engineering.
    
    Parameters
    ----------
    state : str
        Current MARCH-PAWS state (M, A, R, C, H, P, A2, W, S)
    scenario : str
        Medical scenario description
        
    Returns
    -------
    str
        Complete prompt for question generation with few-shot examples
    """
    state_definition = STAGE_DEFINITIONS.get(state, "")
    
    # Get context-enhanced scenario with few-shot examples
    scenario_context = get_scenario_context(state, scenario)
    
    return f"{Q_SYS}\n\n{Q_USER.format(state=state, state_definition=state_definition, scenario=scenario_context)}"


def build_a_prompt(state: str, scenario: str, user_answer: str, catalog: str, excerpts: str) -> str:
    """Build A-Gen prompt for answer generation with checklist.
    
    Parameters
    ----------
    state : str
        Current MARCH-PAWS state (M, A, R, C, H, P, A2, W, S)
    scenario : str
        Medical scenario description
    user_answer : str
        User's response to the current question
    catalog : str
        Formatted catalog of retrieved excerpts
    excerpts : str
        Formatted excerpts with content
        
    Returns
    -------
    str
        Complete prompt for answer generation
    """
    state_definition = STAGE_DEFINITIONS.get(state, "")
    
    # Truncate scenario to keep prompt focused
    scenario_truncated = scenario[:120] + "..." if len(scenario) > 120 else scenario
    
    return f"{A_SYS}\n\n{A_USER.format(state=state, state_definition=state_definition, scenario=scenario_truncated, user_answer=user_answer, catalog=catalog, excerpts=excerpts)}"


def get_next_state_definition(current_state: str) -> str:
    """Get the definition for the next state in MARCH-PAWS sequence.
    
    Parameters
    ----------
    current_state : str
        Current state
        
    Returns
    -------
    str
        Definition for the next state, or empty string if at end
    """
    # MARCH-PAWS state sequence
    state_sequence = ["M", "A", "R", "C", "H", "P", "A2", "W", "S"]
    
    try:
        current_index = state_sequence.index(current_state)
        if current_index < len(state_sequence) - 1:
            next_state = state_sequence[current_index + 1]
            return STAGE_DEFINITIONS.get(next_state, "")
    except ValueError:
        pass
    
    return ""
