#!/usr/bin/env python3
"""
Quality Evaluator for MARCH-PAWS System

Evaluates question quality, citation accuracy, and semantic matching.
"""

import json
import re
from typing import Dict, List, Any, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from nltk.corpus import wordnet
    from nltk import download
    # Download WordNet data if not already present
    try:
        download('wordnet', quiet=True)
    except:
        pass
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False

try:
    import spacy
    # Try to load English model
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        # Fallback to basic model
        try:
            nlp = spacy.load("en_core_web_md")
            SPACY_AVAILABLE = True
        except OSError:
            SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False

class QualityEvaluator:
    """Evaluates quality of questions, citations, and semantic matching"""
    
    def __init__(self):
        self.semantic_model = None
        self.citation_data = {}
        self.stage_definitions = {}
        self.actionable_verbs = self._expand_actionable_verbs()
        self._load_data()
    
    def _load_data(self):
        """Load citation data and stage definitions"""
        # Load citation data
        try:
            with open('data/tc4-02.1_sections.jsonl', 'r') as f:
                for line in f:
                    data = json.loads(line)
                    citation_id = data['id']
                    self.citation_data[citation_id] = data
            print(f"✅ Loaded {len(self.citation_data)} citation records")
        except Exception as e:
            print(f"❌ Error loading citation data: {e}")
        
        # Load stage definitions
        try:
            from src.nodes import STAGE_DEFINITIONS
            self.stage_definitions = STAGE_DEFINITIONS
            print(f"✅ Loaded {len(self.stage_definitions)} stage definitions")
        except Exception as e:
            print(f"❌ Error loading stage definitions: {e}")
        
        # Load semantic model
        try:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Loaded semantic model")
        except Exception as e:
            print(f"❌ Error loading semantic model: {e}")
    
    def _expand_actionable_verbs(self) -> set:
        """Expand actionable verbs using WordNet synonyms"""
        base_verbs = {
            'apply', 'monitor', 'check', 'assess', 'establish', 'begin', 'administer',
            'document', 'observe', 'verify', 'prepare', 'maintain', 'evaluate',
            'inspect', 'confirm', 'record', 'provide', 'ensure', 'continue',
            'seek', 'cover', 'obtain', 'secure', 'control', 'manage', 'perform',
            'conduct', 'implement', 'execute', 'accomplish', 'achieve'
        }
        
        expanded_verbs = base_verbs.copy()
        
        if WORDNET_AVAILABLE:
            try:
                for verb in base_verbs:
                    # Get synonyms for each verb
                    synsets = wordnet.synsets(verb, pos=wordnet.VERB)
                    for synset in synsets:
                        for lemma in synset.lemmas():
                            expanded_verbs.add(lemma.name().lower())
                print(f"✅ Expanded actionable verbs to {len(expanded_verbs)} terms")
            except Exception as e:
                print(f"⚠️  Could not expand verbs with WordNet: {e}")
        else:
            print("⚠️  WordNet not available, using base verb set")
        
        return expanded_verbs
    
    def _is_actionable_with_spacy(self, text: str) -> bool:
        """Use SpaCy to detect actionable items through dependency parsing"""
        if not SPACY_AVAILABLE:
            return False
        
        try:
            doc = nlp(text)
            
            # Check for imperative verbs (root of sentence)
            for token in doc:
                if (token.pos_ == "VERB" and 
                    (token.dep_ == "ROOT" or token.dep_ == "ccomp") and
                    token.lemma_.lower() in self.actionable_verbs):
                    return True
            
            # Check for passive voice with action verbs
            for token in doc:
                if (token.pos_ == "VERB" and 
                    token.dep_ == "ROOT" and
                    "pass" in token.tag_.lower() and
                    token.lemma_.lower() in self.actionable_verbs):
                    return True
            
            # Check for modal verbs with action verbs
            for token in doc:
                if (token.pos_ == "AUX" and token.lemma_.lower() in ["should", "must", "need"]):
                    for child in token.children:
                        if (child.pos_ == "VERB" and 
                            child.lemma_.lower() in self.actionable_verbs):
                            return True
            
            return False
        except Exception:
            return False
    
    def evaluate_question_quality(self, question: str, stage: str, scenario: str) -> Dict[str, Any]:
        """Evaluate question quality based on multiple criteria"""
        
        if not question or question == "Unable to formulate question for the current state.":
            return {
                "quality_score": 0.0,
                "issues": ["Empty or failed question generation"],
                "suggestions": ["Check Q-Gen prompt and LLM response"]
            }
        
        issues = []
        suggestions = []
        score = 1.0
        
        # 1. Check if question is stage-appropriate
        stage_definition = self.stage_definitions.get(stage, "")
        if stage_definition:
            semantic_similarity = self._calculate_semantic_similarity(question, stage_definition)
            if semantic_similarity < 0.25:
                issues.append(f"Question not semantically aligned with stage definition (similarity: {semantic_similarity:.3f})")
                suggestions.append("Question should focus more on stage-specific requirements")
                score -= 0.05
        
        # 2. Check for procedural questions (bad)
        procedural_patterns = [
            r"is it necessary to",
            r"should you",
            r"what actions are required",
            r"what should be done"
        ]
        
        for pattern in procedural_patterns:
            if re.search(pattern, question.lower()):
                issues.append(f"Contains procedural question pattern: '{pattern}'")
                suggestions.append("Ask about observable conditions, not treatment decisions")
                score -= 0.4
        
        # 3. Check for observable conditions (good)
        observable_patterns = [
            r"is there",
            r"are there",
            r"does the",
            r"is the",
            r"can you see",
            r"is visible",
            r"appears to be"
        ]
        
        has_observable = any(re.search(pattern, question.lower()) for pattern in observable_patterns)
        if has_observable:
            score += 0.2
        else:
            issues.append("Question doesn't ask about observable conditions")
            suggestions.append("Ask about what can be observed or measured")
            score -= 0.2
        
        # 4. Check question length and clarity
        if len(question) < 10:
            issues.append("Question too short")
            suggestions.append("Provide more specific details")
            score -= 0.2
        elif len(question) > 150:
            issues.append("Question too long")
            suggestions.append("Keep question concise and focused")
            score -= 0.1
        
        # 5. Check for multi-part questions (bad)
        if question.count('?') > 1:
            issues.append("Multi-part question detected")
            suggestions.append("Ask single, focused question")
            score -= 0.3
        
        # 6. Check scenario relevance (increased weight)
        if scenario:
            scenario_similarity = self._calculate_semantic_similarity(question, scenario)
            if scenario_similarity < 0.3:  # Lowered threshold
                issues.append("Question not relevant to scenario")
                suggestions.append("Consider scenario context when generating questions")
                score -= 0.05  # Further reduced penalty
        
        return {
            "quality_score": max(0.0, min(1.0, score)),
            "issues": issues,
            "suggestions": suggestions,
            "semantic_similarity_to_stage": semantic_similarity if stage_definition else 0.0,
            "semantic_similarity_to_scenario": scenario_similarity if scenario else 0.0,
            "has_observable_conditions": has_observable,
            "question_length": len(question)
        }
    
    def evaluate_citation_accuracy(self, citations: List[str]) -> Dict[str, Any]:
        """Evaluate citation accuracy and relevance"""
        
        if not citations:
            return {
                "accuracy_score": 0.0,
                "valid_citations": 0,
                "total_citations": 0,
                "issues": ["No citations provided"],
                "suggestions": ["Generate relevant citations from retrieved content"]
            }
        
        valid_citations = 0
        issues = []
        suggestions = []
        
        # Import citation mapping function
        from src.utils import map_citation_format
        
        for citation in citations:
            # Use the same mapping logic as the orchestrator
            mapped_id = map_citation_format(citation)
            
            if mapped_id and mapped_id in self.citation_data:
                valid_citations += 1
            else:
                issues.append(f"Citation not found in database: {citation}")
                suggestions.append("Verify citation format and existence")
        
        accuracy_score = valid_citations / len(citations) if citations else 0.0
        
        # Add repetition penalty for excessive duplicate citations (only if >50% are duplicates)
        unique_citations = len(set(citations))
        duplicate_ratio = (len(citations) - unique_citations) / len(citations) if citations else 0
        
        # Only penalize if more than 50% are duplicates (allowing some contextual reuse)
        if duplicate_ratio > 0.5:
            duplicate_penalty = 0.1 * duplicate_ratio  # Penalty proportional to duplication
            accuracy_score -= duplicate_penalty
            issues.append(f"Excessive duplicate citations: {duplicate_ratio:.1%} are duplicates")
            suggestions.append("Reduce citation repetition - same citation should be used for different contexts")
        else:
            duplicate_penalty = 0
        
        if accuracy_score < 0.8:
            suggestions.append("Improve citation generation from retrieved content")
        
        return {
            "accuracy_score": max(0.0, accuracy_score),
            "valid_citations": valid_citations,
            "total_citations": len(citations),
            "unique_citations": unique_citations,
            "duplicate_penalty": duplicate_penalty,
            "issues": issues,
            "suggestions": suggestions
        }
    
    def evaluate_checklist_quality(self, checklist: List[str], user_response: str, stage: str) -> Dict[str, Any]:
        """Evaluate checklist quality and relevance"""
        
        if not checklist:
            return {
                "quality_score": 0.0,
                "issues": ["Empty checklist"],
                "suggestions": ["Generate actionable checklist items"]
            }
        
        issues = []
        suggestions = []
        score = 1.0
        
        # Check if checklist items are actionable (using expanded verb list)
        actionable_verbs = self.actionable_verbs
        
        actionable_count = 0
        for item in checklist:
            # Try SpaCy semantic detection first, fallback to keyword matching
            is_actionable = (self._is_actionable_with_spacy(item) or 
                           any(word in item.lower() for word in actionable_verbs))
            
            if is_actionable:
                actionable_count += 1
            else:
                issues.append(f"Non-actionable item: {item}")
                suggestions.append("Make checklist items specific and actionable")
                score -= 0.05  # Reduced penalty
        
        actionable_ratio = actionable_count / len(checklist)
        if actionable_ratio < 0.7:
            score -= 0.3
        
        # Check relevance to user response
        if user_response:
            response_similarity = self._calculate_semantic_similarity(
                ' '.join(checklist), user_response
            )
            if response_similarity < 0.3:
                issues.append("Checklist not relevant to user response")
                suggestions.append("Base checklist on what user reported")
                score -= 0.3
        
        # Check stage appropriateness (lowered threshold)
        stage_definition = self.stage_definitions.get(stage, "")
        if stage_definition:
            stage_similarity = self._calculate_semantic_similarity(
                ' '.join(checklist), stage_definition
            )
            if stage_similarity < 0.25:  # Further lowered threshold
                issues.append("Checklist not aligned with stage requirements")
                suggestions.append("Focus checklist on stage-specific actions")
                score -= 0.1  # Reduced penalty
        
        return {
            "quality_score": max(0.0, min(1.0, score)),
            "actionable_items": actionable_count,
            "total_items": len(checklist),
            "actionable_ratio": actionable_ratio,
            "issues": issues,
            "suggestions": suggestions,
            "response_similarity": response_similarity if user_response else 0.0,
            "stage_similarity": stage_similarity if stage_definition else 0.0
        }
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if not self.semantic_model or not text1 or not text2:
            return 0.0
        
        try:
            embeddings = self.semantic_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def evaluate_complete_response(self, question: str, checklist: List[str], citations: List[str], 
                                 user_response: str, stage: str, scenario: str) -> Dict[str, Any]:
        """Evaluate complete response quality"""
        
        question_eval = self.evaluate_question_quality(question, stage, scenario)
        citation_eval = self.evaluate_citation_accuracy(citations)
        checklist_eval = self.evaluate_checklist_quality(checklist, user_response, stage)
        
        # Dynamic weighting based on content richness (placeholder for now)
        # TODO: Add excerpts parameter when available
        content_richness = 0.5  # Default moderate richness
        
        # REBALANCED: Reduce citation weight, increase question weight
        base_citation_weight = 0.25  # Reduced from 0.3
        citation_weight = base_citation_weight + (content_richness * 0.15)  # Reduced from 0.2
        
        # Adjust other weights proportionally
        remaining_weight = 1.0 - citation_weight
        question_weight = remaining_weight * 0.6  # Increased from 0.5
        checklist_weight = remaining_weight * 0.4  # Reduced from 0.5
        
        # Calculate overall score (dynamically weighted)
        overall_score = (
            question_eval["quality_score"] * question_weight +
            citation_eval["accuracy_score"] * citation_weight +
            checklist_eval["quality_score"] * checklist_weight
        )
        
        return {
            "overall_score": overall_score,
            "question_evaluation": question_eval,
            "citation_evaluation": citation_eval,
            "checklist_evaluation": checklist_eval,
            "recommendations": {
                "question": question_eval.get("suggestions", []),
                "citations": citation_eval.get("suggestions", []),
                "checklist": checklist_eval.get("suggestions", [])
            }
        }

def main():
    """Demo the quality evaluator"""
    evaluator = QualityEvaluator()
    
    # Demo with sample data
    question = "Is there any life-threatening external bleeding from the extremities, junctional areas, or axial sites?"
    checklist = ["Apply direct pressure to chest wound", "Monitor for signs of shock"]
    citations = ["Ch6 §6-4, p.35–35", "Ch8 §8-1, p.41–41"]
    user_response = "Yes, there is active bleeding from the chest wound"
    stage = "M"
    scenario = "Gunshot wound to the chest with difficulty breathing and visible bleeding."
    
    print("🧪 QUALITY EVALUATOR DEMO")
    print("=" * 50)
    
    evaluation = evaluator.evaluate_complete_response(
        question, checklist, citations, user_response, stage, scenario
    )
    
    print(f"Overall Score: {evaluation['overall_score']:.3f}")
    print(f"Question Score: {evaluation['question_evaluation']['quality_score']:.3f}")
    print(f"Citation Score: {evaluation['citation_evaluation']['accuracy_score']:.3f}")
    print(f"Checklist Score: {evaluation['checklist_evaluation']['quality_score']:.3f}")
    
    print("\nRecommendations:")
    for category, recs in evaluation['recommendations'].items():
        if recs:
            print(f"  {category}: {recs}")

if __name__ == "__main__":
    main()
