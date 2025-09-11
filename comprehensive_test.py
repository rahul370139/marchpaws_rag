#!/usr/bin/env python3
"""
Comprehensive Test Suite for MARCH-PAWS RAG System

This script tests the system across all 9 MARCH-PAWS states with different
medical scenarios and evaluates performance metrics.
"""

import sys
import os
import json
import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

sys.path.append('src')

from orchestrator import Orchestrator

@dataclass
class TestResult:
    """Container for individual test results"""
    scenario: str
    state: str
    question: str
    user_answer: str
    response: Dict[str, Any]
    success: bool
    error: str = None
    response_time: float = 0.0
    checklist: List[str] = None
    citations: List[str] = None
    has_checklist: bool = False
    has_citations: bool = False
    question_quality_score: float = 0.0
    answer_quality_score: float = 0.0

@dataclass
class ScenarioTest:
    """Container for complete scenario testing"""
    scenario_name: str
    scenario_description: str
    user_answers: List[str]  # Answers for each state
    expected_states: List[str]  # Expected state sequence
    results: List[TestResult] = None
    overall_success: bool = False
    completion_rate: float = 0.0

class MARCHPAWSTester:
    """Comprehensive tester for MARCH-PAWS RAG system"""
    
    def __init__(self):
        """Initialize the tester with orchestrator"""
        try:
            self.orchestrator = Orchestrator(
                bm25_path='data/window_bm25_index.pkl',
                embeddings_path='data/window_embeddings.npy',
                metadata_path='data/window_metadata.json'
            )
            print("✅ Orchestrator initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize orchestrator: {e}")
            raise
    
    def define_test_scenarios(self) -> List[ScenarioTest]:
        """Define comprehensive test scenarios covering different injury types"""
        
        scenarios = [
            # Scenario 1: Chest Injury (Gunshot)
            ScenarioTest(
                scenario_name="Chest Gunshot",
                scenario_description="Gunshot wound to the chest with active bleeding",
                user_answers=[
                    "Yes, there is active bleeding from the chest wound",
                    "No, the airway is clear and patient is breathing",
                    "Yes, breathing is labored and shallow",
                    "Weak pulse, skin is pale and cool",
                    "No signs of hypothermia, no head injury",
                    "Patient reports severe pain (8/10)",
                    "Yes, penetrating wound requires antibiotics, no known allergies",
                    "Chest wound is bleeding, no other injuries visible",
                    "No fractures detected, patient can move all limbs"
                ],
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"]
            ),
            
            # Scenario 2: Arm Burn
            ScenarioTest(
                scenario_name="Arm Burn",
                scenario_description="Severe burns to the right arm",
                user_answers=[
                    "No active bleeding, but severe burns present",
                    "Airway is clear, patient is breathing normally",
                    "Breathing is normal, no respiratory distress",
                    "Strong pulse, good circulation to unaffected areas",
                    "No hypothermia, no head injury",
                    "Severe pain from burns (9/10)",
                    "Yes, burns require antibiotics, no allergies",
                    "Severe burns on right arm, no other injuries",
                    "No fractures, but burns may affect mobility"
                ],
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"]
            ),
            
            # Scenario 3: Head Injury
            ScenarioTest(
                scenario_name="Head Injury",
                scenario_description="Blunt force trauma to the head with altered consciousness",
                user_answers=[
                    "No external bleeding visible",
                    "Airway is clear but patient is unconscious",
                    "Breathing is irregular and shallow",
                    "Weak and irregular pulse",
                    "Patient is unconscious, possible head injury",
                    "Patient is unconscious, cannot assess pain",
                    "No penetrating wounds, no antibiotics needed",
                    "Head injury visible, no other external injuries",
                    "No fractures detected"
                ],
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"]
            ),
            
            # Scenario 4: Leg Fracture
            ScenarioTest(
                scenario_name="Leg Fracture",
                scenario_description="Open fracture of the left femur with bleeding",
                user_answers=[
                    "Yes, there is bleeding from the open fracture",
                    "Airway is clear, patient is conscious and breathing",
                    "Breathing is normal",
                    "Weak pulse due to blood loss",
                    "No hypothermia, no head injury",
                    "Severe pain from fracture (10/10)",
                    "Yes, open fracture requires antibiotics, no allergies",
                    "Open fracture of left femur, no other injuries",
                    "Obvious fracture of left femur, needs splinting"
                ],
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"]
            ),
            
            # Scenario 5: Abdominal Evisceration
            ScenarioTest(
                scenario_name="Abdominal Evisceration",
                scenario_description="Penetrating abdominal wound with organ evisceration",
                user_answers=[
                    "No external bleeding, but internal bleeding suspected",
                    "Airway is clear, patient is conscious",
                    "Breathing is shallow due to pain",
                    "Rapid weak pulse, signs of shock",
                    "No hypothermia, no head injury",
                    "Severe abdominal pain (9/10)",
                    "Yes, penetrating wound requires antibiotics, no allergies",
                    "Abdominal evisceration visible, no other injuries",
                    "No fractures detected"
                ],
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"]
            ),
            
            # Scenario 6: Minor Injury (Control Case)
            ScenarioTest(
                scenario_name="Minor Injury",
                scenario_description="Minor laceration to the hand",
                user_answers=[
                    "No, only minor bleeding from small cut",
                    "Airway is clear, breathing normally",
                    "Breathing is normal",
                    "Strong pulse, good circulation",
                    "No hypothermia, no head injury",
                    "Mild pain (3/10)",
                    "No, minor injury doesn't require antibiotics",
                    "Small cut on hand, no other injuries",
                    "No fractures detected"
                ],
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"]
            )
        ]
        
        return scenarios
    
    def test_single_interaction(self, scenario: str, user_answer: str = None) -> TestResult:
        """Test a single interaction with the system"""
        start_time = time.time()
        
        try:
            response = self.orchestrator.run_step(scenario, user_answer)
            response_time = time.time() - start_time
            
            success = (
                response is not None and 
                not response.get('refusal', False) and
                'error' not in response
            )
            
            # Extract checklist and citations
            checklist = response.get('checklist', []) if response else []
            citations = response.get('citations', []) if response else []
            
            # Evaluate quality scores
            question_quality = self.evaluate_question_quality(response.get('question', ''), response.get('state', '')) if response else 0.0
            answer_quality = self.evaluate_answer_quality(checklist, citations) if response else 0.0
            
            return TestResult(
                scenario=scenario,
                state=response.get('state', 'Unknown') if response else 'Error',
                question=response.get('question', 'No question generated') if response else 'Error',
                user_answer=user_answer or "Initial query",
                response=response or {},
                success=success,
                response_time=response_time,
                checklist=checklist,
                citations=citations,
                has_checklist=len(checklist) > 0,
                has_citations=len(citations) > 0,
                question_quality_score=question_quality,
                answer_quality_score=answer_quality
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                scenario=scenario,
                state="Error",
                question="Error occurred",
                user_answer=user_answer or "Initial query",
                response={},
                success=False,
                error=str(e),
                response_time=response_time
            )
    
    def test_complete_scenario(self, scenario_test: ScenarioTest) -> ScenarioTest:
        """Test a complete scenario through all MARCH-PAWS states"""
        print(f"\n🧪 Testing Scenario: {scenario_test.scenario_name}")
        print(f"📝 Description: {scenario_test.scenario_description}")
        print("=" * 60)
        
        # Reset state machine for each scenario
        self.orchestrator.sm.reset()
        
        results = []
        current_scenario = scenario_test.scenario_description
        
        # Test initial interaction (no user answer) - this should ask question for M state
        print(f"\n🔍 Initial Query: {current_scenario}")
        result = self.test_single_interaction(current_scenario)
        results.append(result)
        
        if not result.success:
            print(f"❌ Initial query failed: {result.error}")
            scenario_test.results = results
            scenario_test.overall_success = False
            scenario_test.completion_rate = 0.0
            return scenario_test
        
        print(f"✅ State {result.state}: {result.question}")
        if result.response.get('checklist'):
            print(f"   📋 Checklist: {result.response.get('checklist')}")
        if result.response.get('citations'):
            print(f"   📚 Citations: {result.response.get('citations')}")
        print(f"Current state after: A")
        
        # Test each subsequent state with user answers
        # Each call should return checklist for current state + question for next state
        state_sequence = ['A', 'R', 'C', 'H', 'P', 'A2', 'W', 'S']
        for i, user_answer in enumerate(scenario_test.user_answers):
            current_state = state_sequence[i] if i < len(state_sequence) else 'END'
            next_state = state_sequence[i + 1] if i + 1 < len(state_sequence) else 'END'
            
            print(f"\n👤 User Answer: {user_answer}")
            print(f"Current state: {current_state}")
            
            result = self.test_single_interaction(current_scenario, user_answer)
            results.append(result)
            
            if not result.success:
                print(f"❌ State {result.state} failed: {result.error}")
                break
            
            if result.response.get('checklist'):
                print(f"   📋 Checklist: {result.response.get('checklist')}")
            if result.response.get('citations'):
                print(f"   📚 Citations: {result.response.get('citations')}")
            print(f"✅ State {result.state}: {result.question}")
            print(f"Current state after: {next_state}")
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.5)
        
        # Calculate completion metrics
        successful_states = sum(1 for r in results if r.success)
        total_expected_states = len(scenario_test.expected_states) + 1  # +1 for initial query
        completion_rate = (successful_states / total_expected_states) * 100
        
        scenario_test.results = results
        scenario_test.overall_success = completion_rate >= 80  # 80% success threshold
        scenario_test.completion_rate = completion_rate
        
        print(f"\n📊 Completion Rate: {completion_rate:.1f}% ({successful_states}/{total_expected_states})")
        print(f"🎯 Overall Success: {'✅' if scenario_test.overall_success else '❌'}")
        
        return scenario_test
    
    def calculate_metrics(self, all_results: List[ScenarioTest]) -> Dict[str, Any]:
        """Calculate comprehensive metrics for all test results"""
        
        total_scenarios = len(all_results)
        successful_scenarios = sum(1 for s in all_results if s.overall_success)
        
        all_interactions = []
        for scenario in all_results:
            if scenario.results:
                all_interactions.extend(scenario.results)
        
        total_interactions = len(all_interactions)
        successful_interactions = sum(1 for r in all_interactions if r.success)
        
        # State-specific metrics
        state_success = {}
        state_questions = {}
        
        for result in all_interactions:
            if result.success:
                state = result.state
                if state not in state_success:
                    state_success[state] = 0
                    state_questions[state] = []
                state_success[state] += 1
                state_questions[state].append(result.question)
        
        # Response time metrics
        response_times = [r.response_time for r in all_interactions if r.success]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Question quality analysis
        question_quality = self.analyze_question_quality(all_interactions)
        
        # Calculate quality scores
        question_scores = [r.question_quality_score for r in all_interactions if r.success]
        answer_scores = [r.answer_quality_score for r in all_interactions if r.success]
        avg_question_quality = sum(question_scores) / len(question_scores) if question_scores else 0
        avg_answer_quality = sum(answer_scores) / len(answer_scores) if answer_scores else 0
        
        # Calculate checklist and citation metrics
        total_with_checklist = sum(1 for r in all_interactions if r.has_checklist)
        total_with_citations = sum(1 for r in all_interactions if r.has_citations)
        
        return {
            "overall_metrics": {
                "total_scenarios": total_scenarios,
                "successful_scenarios": successful_scenarios,
                "scenario_success_rate": (successful_scenarios / total_scenarios) * 100,
                "total_interactions": total_interactions,
                "successful_interactions": successful_interactions,
                "interaction_success_rate": (successful_interactions / total_interactions) * 100,
                "average_response_time": avg_response_time,
                "average_question_quality": avg_question_quality,
                "average_answer_quality": avg_answer_quality,
                "checklist_coverage": (total_with_checklist / total_interactions) * 100,
                "citation_coverage": (total_with_citations / total_interactions) * 100
            },
            "state_metrics": {
                state: {
                    "success_count": state_success.get(state, 0),
                    "total_attempts": sum(1 for r in all_interactions if r.state == state),
                    "success_rate": (state_success.get(state, 0) / sum(1 for r in all_interactions if r.state == state)) * 100 if sum(1 for r in all_interactions if r.state == state) > 0 else 0,
                    "sample_questions": state_questions.get(state, [])[:3]  # First 3 questions
                }
                for state in ["M", "A", "R", "C", "H", "P", "A2", "W", "S"]
            },
            "question_quality": question_quality
        }
    
    def analyze_question_quality(self, results: List[TestResult]) -> Dict[str, Any]:
        """Analyze the quality of generated questions"""
        
        quality_issues = {
            "examiner_style": 0,  # "Is it necessary to..." questions
            "anatomical_mismatch": 0,  # Wrong anatomy for scenario
            "stage_inappropriate": 0,  # Wrong stage focus
            "too_complex": 0,  # Multi-part questions
            "good_questions": 0
        }
        
        for result in results:
            if not result.success or not result.question:
                continue
                
            question = result.question.lower()
            
            # Check for examiner-style questions
            if any(phrase in question for phrase in ["is it necessary", "should you", "what actions"]):
                quality_issues["examiner_style"] += 1
            # Check for complex multi-part questions
            elif question.count("?") > 1 or "if" in question:
                quality_issues["too_complex"] += 1
            # Check for stage appropriateness (basic check)
            elif self.is_stage_appropriate(result.state, result.question):
                quality_issues["good_questions"] += 1
            else:
                quality_issues["stage_inappropriate"] += 1
        
        total_questions = sum(quality_issues.values())
        quality_issues["total_questions"] = total_questions
        
        return quality_issues
    
    def is_stage_appropriate(self, state: str, question: str) -> bool:
        """Basic check if question is appropriate for the stage"""
        question_lower = question.lower()
        
        stage_keywords = {
            "M": ["bleeding", "hemorrhage", "blood", "tourniquet", "pressure"],
            "A": ["airway", "breathing", "obstruction", "clear"],
            "R": ["breathing", "respiration", "chest", "lung"],
            "C": ["pulse", "circulation", "shock", "perfusion"],
            "H": ["hypothermia", "head", "consciousness", "temperature"],
            "P": ["pain", "analgesia", "comfort"],
            "A2": ["antibiotic", "allergy", "penetrating"],
            "W": ["wound", "injury", "dressing", "inspect"],
            "S": ["fracture", "splint", "immobilize", "bone"]
        }
        
        keywords = stage_keywords.get(state, [])
        return any(keyword in question_lower for keyword in keywords)
    
    def evaluate_question_quality(self, question: str, state: str) -> float:
        """Evaluate the quality of a generated question (0-1 scale)"""
        if not question or question == "No question generated":
            return 0.0
        
        score = 1.0
        question_lower = question.lower()
        
        # Deduct points for examiner-style questions
        if any(phrase in question_lower for phrase in ["is it necessary", "should you", "what actions"]):
            score -= 0.3
        
        # Deduct points for complex multi-part questions
        if question.count("?") > 1 or "if" in question_lower:
            score -= 0.2
        
        # Deduct points for stage-inappropriate questions
        if not self.is_stage_appropriate(state, question):
            score -= 0.3
        
        # Deduct points for very short or very long questions
        if len(question) < 10:
            score -= 0.2
        elif len(question) > 200:
            score -= 0.1
        
        return max(0.0, score)
    
    def evaluate_answer_quality(self, checklist: List[str], citations: List[str]) -> float:
        """Evaluate the quality of generated answers (0-1 scale)"""
        if not checklist and not citations:
            return 0.0
        
        score = 0.0
        
        # Checklist quality (60% weight)
        if checklist:
            checklist_score = 0.0
            checklist_score += min(1.0, len(checklist) / 3.0)  # Prefer 2-4 items
            checklist_score += 0.2 if all(len(item) > 10 for item in checklist) else 0  # Substantial items
            checklist_score += 0.2 if all("ch" in item.lower() for item in checklist) else 0  # Medical references
            score += checklist_score * 0.6
        
        # Citations quality (40% weight)
        if citations:
            citation_score = 0.0
            citation_score += min(1.0, len(citations) / 2.0)  # Prefer 2-3 citations
            citation_score += 0.3 if all("ch" in citation.lower() for citation in citations) else 0  # Proper format
            citation_score += 0.2 if all("p." in citation.lower() for citation in citations) else 0  # Page references
            score += citation_score * 0.4
        
        return min(1.0, score)
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        print("🚀 Starting Comprehensive MARCH-PAWS RAG System Test")
        print("=" * 70)
        
        # Define test scenarios
        scenarios = self.define_test_scenarios()
        
        # Test each scenario
        all_results = []
        for scenario in scenarios:
            result = self.test_complete_scenario(scenario)
            all_results.append(result)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_results)
        
        # Generate report
        self.generate_report(all_results, metrics)
        
        return {
            "scenarios": all_results,
            "metrics": metrics
        }
    
    def generate_report(self, results: List[ScenarioTest], metrics: Dict[str, Any]):
        """Generate a comprehensive test report"""
        
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 70)
        
        # Overall metrics
        overall = metrics["overall_metrics"]
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   Scenarios Tested: {overall['total_scenarios']}")
        print(f"   Successful Scenarios: {overall['successful_scenarios']}")
        print(f"   Scenario Success Rate: {overall['scenario_success_rate']:.1f}%")
        print(f"   Total Interactions: {overall['total_interactions']}")
        print(f"   Successful Interactions: {overall['successful_interactions']}")
        print(f"   Interaction Success Rate: {overall['interaction_success_rate']:.1f}%")
        print(f"   Average Response Time: {overall['average_response_time']:.2f}s")
        print(f"   Average Question Quality: {overall['average_question_quality']:.2f}/1.0")
        print(f"   Average Answer Quality: {overall['average_answer_quality']:.2f}/1.0")
        print(f"   Checklist Coverage: {overall['checklist_coverage']:.1f}%")
        print(f"   Citation Coverage: {overall['citation_coverage']:.1f}%")
        
        # Scenario-specific results
        print(f"\n📋 SCENARIO RESULTS:")
        for scenario in results:
            status = "✅ PASS" if scenario.overall_success else "❌ FAIL"
            print(f"   {scenario.scenario_name}: {status} ({scenario.completion_rate:.1f}%)")
        
        # State-specific metrics
        print(f"\n🔬 STATE-SPECIFIC PERFORMANCE:")
        state_metrics = metrics["state_metrics"]
        for state in ["M", "A", "R", "C", "H", "P", "A2", "W", "S"]:
            if state in state_metrics:
                state_data = state_metrics[state]
                print(f"   {state}: {state_data['success_rate']:.1f}% ({state_data['success_count']}/{state_data['total_attempts']})")
        
        # Question quality analysis
        print(f"\n❓ QUESTION QUALITY ANALYSIS:")
        quality = metrics["question_quality"]
        print(f"   Good Questions: {quality['good_questions']} ({quality['good_questions']/quality['total_questions']*100:.1f}%)")
        print(f"   Examiner Style: {quality['examiner_style']} ({quality['examiner_style']/quality['total_questions']*100:.1f}%)")
        print(f"   Too Complex: {quality['too_complex']} ({quality['too_complex']/quality['total_questions']*100:.1f}%)")
        print(f"   Stage Inappropriate: {quality['stage_inappropriate']} ({quality['stage_inappropriate']/quality['total_questions']*100:.1f}%)")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if overall['scenario_success_rate'] < 80:
            print("   ⚠️  Overall success rate is below 80% - system needs improvement")
        if quality['examiner_style'] > quality['total_questions'] * 0.1:
            print("   ⚠️  Too many examiner-style questions - improve prompt engineering")
        if quality['stage_inappropriate'] > quality['total_questions'] * 0.2:
            print("   ⚠️  Many stage-inappropriate questions - review stage definitions")
        
        print(f"\n✅ Test completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main test execution"""
    try:
        tester = MARCHPAWSTester()
        results = tester.run_comprehensive_test()
        
        # Save results to file
        with open('test_results.json', 'w') as f:
            # Convert dataclasses to dict for JSON serialization
            json_results = []
            for scenario in results['scenarios']:
                scenario_dict = {
                    'scenario_name': scenario.scenario_name,
                    'scenario_description': scenario.scenario_description,
                    'overall_success': scenario.overall_success,
                    'completion_rate': scenario.completion_rate,
                    'results': [
                        {
                            'scenario': r.scenario,
                            'state': r.state,
                            'question': r.question,
                            'user_answer': r.user_answer,
                            'success': r.success,
                            'error': r.error,
                            'response_time': r.response_time,
                            'checklist': r.checklist,
                            'citations': r.citations,
                            'has_checklist': r.has_checklist,
                            'has_citations': r.has_citations,
                            'question_quality_score': r.question_quality_score,
                            'answer_quality_score': r.answer_quality_score
                        }
                        for r in scenario.results
                    ]
                }
                json_results.append(scenario_dict)
            
            json.dump({
                'scenarios': json_results,
                'metrics': results['metrics']
            }, f, indent=2)
        
        print(f"\n💾 Results saved to test_results.json")
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
