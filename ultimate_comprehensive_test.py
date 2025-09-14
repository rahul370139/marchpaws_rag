#!/usr/bin/env python3
"""
Ultimate Comprehensive Test Suite for MARCH-PAWS RAG System

This script merges all existing test files and provides:
1. Comprehensive medical scenarios (6 different injury types)
2. Non-medical refusal testing
3. Quality evaluation with advanced metrics
4. Async orchestrator testing
5. Performance benchmarking
6. Detailed reporting and analysis

Combines:
- comprehensive_test.py (6 medical scenarios)
- comprehensive_test_v2.py (2 different scenarios)  
- comprehensive_async_test_v2.py (async testing)
- comprehensive_quality_test.py (quality evaluation)
- test_validation_and_refusal.py (refusal testing)
"""

import sys
import os
import asyncio
import time
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback

sys.path.append('src')

from src.orchestrator_async import AsyncOrchestrator
from quality_evaluator import QualityEvaluator
from src.utils import map_citations_to_database

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
    citation_accuracy_score: float = 0.0
    checklist_quality_score: float = 0.0
    overall_quality_score: float = 0.0
    refusal_detected: bool = False
    medical_relevance: bool = True

@dataclass
class ScenarioTest:
    """Container for complete scenario testing"""
    scenario_name: str
    scenario_description: str
    scenario_type: str  # 'medical', 'non-medical', 'edge-case'
    user_answers: List[str]
    expected_states: List[str]
    expected_refusal: bool = False
    results: List[TestResult] = None
    overall_success: bool = False
    completion_rate: float = 0.0
    avg_quality_score: float = 0.0
    avg_response_time: float = 0.0
    citation_accuracy: float = 0.0
    refusal_accuracy: float = 0.0

class UltimateComprehensiveTester:
    """Ultimate comprehensive tester for MARCH-PAWS RAG system"""
    
    def __init__(self):
        self.orchestrator = None
        self.evaluator = QualityEvaluator()
        self.results = []
        print("🚀 Ultimate Comprehensive Test Suite Initialized")
        print("✅ Quality Evaluator loaded with advanced metrics")
    
    async def initialize(self):
        """Initialize the async orchestrator"""
        print("\n🔧 Initializing Async Orchestrator...")
        
        self.orchestrator = AsyncOrchestrator(
            bm25_path="data/window_bm25_index.pkl",
            embeddings_path="data/window_embeddings.npy",
            metadata_path="data/window_metadata.json"
        )
        
        await self.orchestrator.__aenter__()
        print("✅ Async orchestrator initialized successfully")
    
    async def cleanup(self):
        """Cleanup the orchestrator"""
        if self.orchestrator:
            await self.orchestrator.__aexit__(None, None, None)
    
    def define_test_scenarios(self) -> List[ScenarioTest]:
        """Define comprehensive test scenarios covering all cases"""
        
        scenarios = [
            # MEDICAL SCENARIOS (6 scenarios from comprehensive_test.py)
            
            # Scenario 1: Chest Injury (Gunshot)
            ScenarioTest(
                scenario_name="Chest Gunshot",
                scenario_description="Gunshot wound to the chest with active bleeding",
                scenario_type="medical",
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
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"],
                expected_refusal=False
            ),
            
            # Scenario 2: Arm Burn
            ScenarioTest(
                scenario_name="Arm Burn",
                scenario_description="Severe burns to the right arm",
                scenario_type="medical",
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
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"],
                expected_refusal=False
            ),
            
            # Scenario 3: Head Injury
            ScenarioTest(
                scenario_name="Head Injury",
                scenario_description="Blunt force trauma to the head with altered consciousness",
                scenario_type="medical",
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
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"],
                expected_refusal=False
            ),
            
            # Scenario 4: Leg Fracture
            ScenarioTest(
                scenario_name="Leg Fracture",
                scenario_description="Open fracture of the left femur with bleeding",
                scenario_type="medical",
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
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"],
                expected_refusal=False
            ),
            
            # Scenario 5: Multi-Trauma (from comprehensive_test_v2.py)
            ScenarioTest(
                scenario_name="Multi-Trauma",
                scenario_description="Motor vehicle accident with multiple injuries: head trauma, chest injury, and leg fracture. Patient is unconscious with signs of shock.",
                scenario_type="medical",
                user_answers=[
                    "Yes, there is active bleeding from the head wound and leg fracture",
                    "Airway is compromised - patient is unconscious and snoring",
                    "Breathing is irregular and shallow with gurgling sounds",
                    "Weak, rapid pulse at 120 bpm, skin is pale and clammy",
                    "Patient is unconscious, possible head injury, body temperature normal",
                    "Patient is unconscious, cannot assess pain level",
                    "Yes, open fractures and head wound require antibiotics, no known allergies",
                    "Multiple injuries: head laceration, chest bruising, open leg fracture",
                    "Obvious fracture of right femur, possible head injury, chest appears intact"
                ],
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"],
                expected_refusal=False
            ),
            
            # Scenario 6: Environmental Emergency (from comprehensive_test_v2.py)
            ScenarioTest(
                scenario_name="Environmental Emergency",
                scenario_description="Hiker found unconscious in cold weather with severe hypothermia, no obvious trauma but altered mental status.",
                scenario_type="medical",
                user_answers=[
                    "No active bleeding visible, but patient is very pale",
                    "Airway is clear but patient is barely responsive",
                    "Breathing is very slow and shallow, 6 breaths per minute",
                    "Very weak pulse, 40 bpm, skin is cold to touch",
                    "Severe hypothermia - body temperature feels very cold, no head injury",
                    "Patient is barely responsive, cannot assess pain",
                    "No penetrating wounds, no antibiotics needed, no known allergies",
                    "No visible injuries, but severe cold exposure signs",
                    "No fractures detected, but patient is very stiff from cold"
                ],
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"],
                expected_refusal=False
            ),
            
            # NON-MEDICAL SCENARIOS (refusal testing)
            
            # Scenario 7: Weather Query
            ScenarioTest(
                scenario_name="Weather Query",
                scenario_description="What's the weather like today?",
                scenario_type="non-medical",
                user_answers=[],
                expected_states=[],
                expected_refusal=True
            ),
            
            # Scenario 8: Cooking Query
            ScenarioTest(
                scenario_name="Cooking Query",
                scenario_description="How do I cook pasta?",
                scenario_type="non-medical",
                user_answers=[],
                expected_states=[],
                expected_refusal=True
            ),
            
            # Scenario 9: General Knowledge Query
            ScenarioTest(
                scenario_name="General Knowledge Query",
                scenario_description="What's the capital of France?",
                scenario_type="non-medical",
                user_answers=[],
                expected_states=[],
                expected_refusal=True
            ),
            
            # MEDICAL EDGE CASES (boundary testing)
            
            # Scenario 10: Minor Medical Condition
            ScenarioTest(
                scenario_name="Minor Medical Condition",
                scenario_description="I have a cold and runny nose",
                scenario_type="medical-edge-case",
                user_answers=[
                    "No active bleeding visible",
                    "Airway is clear, patient is breathing normally",
                    "Breathing is normal, slight congestion",
                    "Strong pulse, good circulation",
                    "No hypothermia, no head injury",
                    "Mild discomfort (2/10)",
                    "No antibiotics needed for cold",
                    "No visible injuries, just congestion",
                    "No fractures detected"
                ],
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"],
                expected_refusal=False
            ),
            
            # Scenario 11: Dental Issue
            ScenarioTest(
                scenario_name="Dental Issue",
                scenario_description="My tooth hurts",
                scenario_type="medical-edge-case",
                user_answers=[
                    "No active bleeding visible",
                    "Airway is clear, patient is breathing normally",
                    "Breathing is normal",
                    "Strong pulse, good circulation",
                    "No hypothermia, no head injury",
                    "Severe tooth pain (7/10)",
                    "Dental antibiotics may be needed, no known allergies",
                    "No visible injuries, just dental pain",
                    "No fractures detected"
                ],
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"],
                expected_refusal=False
            )
        ]
        
        return scenarios
    
    async def test_single_interaction(self, scenario: str, user_answer: str = None, expected_refusal: bool = False) -> TestResult:
        """Test a single interaction with comprehensive quality evaluation"""
        start_time = time.time()
        
        try:
            response = await self.orchestrator.run_step(scenario, user_answer)
            response_time = time.time() - start_time
            
            # Check for refusal
            refusal_detected = response.get('refusal', False) if response else False
            success = (
                response is not None and 
                not (refusal_detected and not expected_refusal) and
                'error' not in response
            )
            
            # Extract components
            checklist = response.get('checklist', []) if response else []
            citations = response.get('citations', []) if response else []
            question = response.get('question', '') if response else ''
            state = response.get('state', 'Unknown') if response else 'Error'
            
            # Map citations to database format for quality evaluation
            citation_db = {}
            try:
                with open('data/tc4-02.1_sections.jsonl', 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        citation_db[data['id']] = data
            except:
                citation_db = self.evaluator.citation_data
            
            mapped_citations = map_citations_to_database(citations, citation_db) if citations else []
            
            # Comprehensive quality evaluation
            question_eval = self.evaluator.evaluate_question_quality(question, state, scenario) if question else {"quality_score": 0.0}
            citation_eval = self.evaluator.evaluate_citation_accuracy(mapped_citations) if mapped_citations else {"accuracy_score": 0.0}
            checklist_eval = self.evaluator.evaluate_checklist_quality(checklist, user_answer or "", state) if checklist else {"quality_score": 0.0}
            
            # Overall quality score
            overall_eval = self.evaluator.evaluate_complete_response(
                question, checklist, mapped_citations, user_answer or "", state, scenario
            )
            
            return TestResult(
                scenario=scenario,
                state=state,
                question=question,
                user_answer=user_answer or "Initial query",
                response=response or {},
                success=success,
                response_time=response_time,
                checklist=checklist,
                citations=citations,
                has_checklist=len(checklist) > 0,
                has_citations=len(citations) > 0,
                question_quality_score=question_eval.get("quality_score", 0.0),
                answer_quality_score=overall_eval.get("overall_score", 0.0),
                citation_accuracy_score=citation_eval.get("accuracy_score", 0.0),
                checklist_quality_score=checklist_eval.get("quality_score", 0.0),
                overall_quality_score=overall_eval.get("overall_score", 0.0),
                refusal_detected=refusal_detected,
                medical_relevance=not refusal_detected
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
                response_time=response_time,
                refusal_detected=False,
                medical_relevance=False
            )
    
    async def test_complete_scenario(self, scenario_test: ScenarioTest) -> ScenarioTest:
        """Test a complete scenario with comprehensive evaluation"""
        print(f"\n🧪 Testing Scenario: {scenario_test.scenario_name}")
        print(f"📝 Description: {scenario_test.scenario_description}")
        print(f"🏷️  Type: {scenario_test.scenario_type}")
        print(f"🎯 Expected Refusal: {scenario_test.expected_refusal}")
        print("=" * 60)
        
        # Reset state machine for each scenario
        self.orchestrator.sm.reset()
        
        results = []
        current_scenario = scenario_test.scenario_description
        
        # Test initial interaction
        print(f"\n🔍 Initial Query: {current_scenario}")
        result = await self.test_single_interaction(current_scenario, expected_refusal=scenario_test.expected_refusal)
        results.append(result)
        
        if not result.success and not scenario_test.expected_refusal:
            print(f"❌ Initial query failed: {result.error}")
            scenario_test.results = results
            scenario_test.overall_success = False
            scenario_test.completion_rate = 0.0
            return scenario_test
        
        if result.refusal_detected:
            print(f"✅ REFUSAL: Non-medical query detected - no valid response generated")
            scenario_test.results = results
            scenario_test.overall_success = result.refusal_detected == scenario_test.expected_refusal
            scenario_test.completion_rate = 100.0 if scenario_test.overall_success else 0.0
            scenario_test.refusal_accuracy = 100.0 if scenario_test.overall_success else 0.0
            return scenario_test
        
        print(f"✅ State {result.state}: {result.question}")
        print(f"📊 Quality Score: {result.overall_quality_score:.3f}")
        if result.checklist:
            print(f"   📋 Checklist: {len(result.checklist)} items")
        if result.citations:
            print(f"   📚 Citations: {len(result.citations)} items (Accuracy: {result.citation_accuracy_score:.3f})")
        
        # Test each subsequent state with user answers (for medical scenarios and edge cases)
        if scenario_test.scenario_type in ["medical", "medical-edge-case"] and not scenario_test.expected_refusal:
            state_sequence = ['A', 'R', 'C', 'H', 'P', 'A2', 'W', 'S']
            for i, user_answer in enumerate(scenario_test.user_answers):
                current_state = state_sequence[i] if i < len(state_sequence) else 'END'
                next_state = state_sequence[i + 1] if i + 1 < len(state_sequence) else 'END'
                
                print(f"\n👤 User Answer: {user_answer}")
                print(f"Current state: {current_state}")
                
                result = await self.test_single_interaction(current_scenario, user_answer)
                results.append(result)
                
                if not result.success:
                    print(f"❌ State {result.state} failed: {result.error}")
                    break
                
                print(f"✅ State {result.state}: {result.question}")
                print(f"📊 Quality Score: {result.overall_quality_score:.3f}")
                if result.checklist:
                    print(f"   📋 Checklist: {len(result.checklist)} items")
                if result.citations:
                    print(f"   📚 Citations: {len(result.citations)} items (Accuracy: {result.citation_accuracy_score:.3f})")
                print(f"🔄 State: {current_state} → {next_state}")
                
                # Small delay to avoid overwhelming the system
                await asyncio.sleep(0.5)
        
        # Calculate comprehensive metrics
        successful_states = sum(1 for r in results if r.success)
        total_expected_states = len(scenario_test.expected_states) + 1 if not scenario_test.expected_refusal else 1
        completion_rate = (successful_states / total_expected_states) * 100 if total_expected_states > 0 else 0.0
        
        # Quality metrics
        quality_scores = [r.overall_quality_score for r in results if r.success]
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        response_times = [r.response_time for r in results if r.success]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        citation_scores = [r.citation_accuracy_score for r in results if r.success and r.citations]
        citation_accuracy = sum(citation_scores) / len(citation_scores) if citation_scores else 0.0
        
        # Refusal accuracy
        refusal_accuracy = 100.0 if result.refusal_detected == scenario_test.expected_refusal else 0.0
        
        scenario_test.results = results
        scenario_test.overall_success = completion_rate >= 80.0 if not scenario_test.expected_refusal else refusal_accuracy == 100.0
        scenario_test.completion_rate = completion_rate
        scenario_test.avg_quality_score = avg_quality_score
        scenario_test.avg_response_time = avg_response_time
        scenario_test.citation_accuracy = citation_accuracy
        scenario_test.refusal_accuracy = refusal_accuracy
        
        print(f"\n📊 SCENARIO RESULTS:")
        print(f"   States Completed: {successful_states}/{total_expected_states}")
        print(f"   Completion Rate: {completion_rate:.1f}%")
        print(f"   Avg Quality Score: {avg_quality_score:.3f}")
        print(f"   Avg Response Time: {avg_response_time:.2f}s")
        print(f"   Citation Accuracy: {citation_accuracy:.3f}")
        print(f"   Refusal Accuracy: {refusal_accuracy:.1f}%")
        print(f"🎯 Overall Success: {'✅' if scenario_test.overall_success else '❌'}")
        
        return scenario_test
    
    def calculate_comprehensive_metrics(self, all_results: List[ScenarioTest]) -> Dict[str, Any]:
        """Calculate comprehensive metrics for all test results"""
        
        total_scenarios = len(all_results)
        successful_scenarios = sum(1 for s in all_results if s.overall_success)
        
        # Separate by scenario type
        medical_scenarios = [s for s in all_results if s.scenario_type == "medical"]
        non_medical_scenarios = [s for s in all_results if s.scenario_type == "non-medical"]
        edge_case_scenarios = [s for s in all_results if s.scenario_type == "medical-edge-case"]
        
        # Overall metrics
        all_interactions = []
        for scenario in all_results:
            if scenario.results:
                all_interactions.extend(scenario.results)
        
        total_interactions = len(all_interactions)
        successful_interactions = sum(1 for r in all_interactions if r.success)
        
        # Quality metrics
        quality_scores = [r.overall_quality_score for r in all_interactions if r.success]
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        citation_scores = [r.citation_accuracy_score for r in all_interactions if r.citations and r.success]
        avg_citation_accuracy = sum(citation_scores) / len(citation_scores) if citation_scores else 0.0
        
        checklist_scores = [r.checklist_quality_score for r in all_interactions if r.checklist and r.success]
        avg_checklist_quality = sum(checklist_scores) / len(checklist_scores) if checklist_scores else 0.0
        
        question_scores = [r.question_quality_score for r in all_interactions if r.success]
        avg_question_quality = sum(question_scores) / len(question_scores) if question_scores else 0.0
        
        # Performance metrics
        response_times = [r.response_time for r in all_interactions if r.success]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        # Refusal metrics
        refusal_tests = [s for s in all_results if s.expected_refusal]
        correct_refusals = sum(1 for s in refusal_tests if s.refusal_accuracy == 100.0)
        refusal_accuracy = (correct_refusals / len(refusal_tests)) * 100 if refusal_tests else 100.0
        
        # State-specific metrics
        state_success = {}
        for result in all_interactions:
            if result.success and not result.refusal_detected:
                state = result.state
                if state not in state_success:
                    state_success[state] = 0
                state_success[state] += 1
        
        return {
            "overall_metrics": {
                "total_scenarios": total_scenarios,
                "successful_scenarios": successful_scenarios,
                "scenario_success_rate": (successful_scenarios / total_scenarios) * 100,
                "total_interactions": total_interactions,
                "successful_interactions": successful_interactions,
                "interaction_success_rate": (successful_interactions / total_interactions) * 100,
                "average_response_time": avg_response_time,
                "average_quality_score": avg_quality_score,
                "average_citation_accuracy": avg_citation_accuracy,
                "average_checklist_quality": avg_checklist_quality,
                "average_question_quality": avg_question_quality,
                "refusal_accuracy": refusal_accuracy
            },
            "scenario_type_metrics": {
                "medical_scenarios": {
                    "total": len(medical_scenarios),
                    "successful": sum(1 for s in medical_scenarios if s.overall_success),
                    "avg_quality_score": sum(s.avg_quality_score for s in medical_scenarios) / len(medical_scenarios) if medical_scenarios else 0.0,
                    "avg_completion_rate": sum(s.completion_rate for s in medical_scenarios) / len(medical_scenarios) if medical_scenarios else 0.0
                },
                "non_medical_scenarios": {
                    "total": len(non_medical_scenarios),
                    "successful": sum(1 for s in non_medical_scenarios if s.overall_success),
                    "refusal_accuracy": sum(s.refusal_accuracy for s in non_medical_scenarios) / len(non_medical_scenarios) if non_medical_scenarios else 0.0
                },
                "edge_case_scenarios": {
                    "total": len(edge_case_scenarios),
                    "successful": sum(1 for s in edge_case_scenarios if s.overall_success),
                    "avg_quality_score": sum(s.avg_quality_score for s in edge_case_scenarios) / len(edge_case_scenarios) if edge_case_scenarios else 0.0
                }
            },
            "state_metrics": {
                state: {
                    "success_count": state_success.get(state, 0),
                    "total_attempts": sum(1 for r in all_interactions if r.state == state),
                    "success_rate": (state_success.get(state, 0) / sum(1 for r in all_interactions if r.state == state)) * 100 if sum(1 for r in all_interactions if r.state == state) > 0 else 0
                }
                for state in ["M", "A", "R", "C", "H", "P", "A2", "W", "S"]
            }
        }
    
    def generate_comprehensive_report(self, results: List[ScenarioTest], metrics: Dict[str, Any]):
        """Generate a comprehensive test report"""
        
        print("\n" + "=" * 80)
        print("🏆 ULTIMATE COMPREHENSIVE TEST REPORT")
        print("=" * 80)
        
        # Overall metrics
        overall = metrics["overall_metrics"]
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   Total Scenarios: {overall['total_scenarios']}")
        print(f"   Successful Scenarios: {overall['successful_scenarios']}")
        print(f"   Scenario Success Rate: {overall['scenario_success_rate']:.1f}%")
        print(f"   Total Interactions: {overall['total_interactions']}")
        print(f"   Successful Interactions: {overall['successful_interactions']}")
        print(f"   Interaction Success Rate: {overall['interaction_success_rate']:.1f}%")
        print(f"   Average Response Time: {overall['average_response_time']:.2f}s")
        print(f"   Refusal Accuracy: {overall['refusal_accuracy']:.1f}%")
        
        print(f"\n📊 QUALITY METRICS:")
        print(f"   Overall Quality Score: {overall['average_quality_score']:.3f}")
        print(f"   Citation Accuracy: {overall['average_citation_accuracy']:.3f}")
        print(f"   Checklist Quality: {overall['average_checklist_quality']:.3f}")
        print(f"   Question Quality: {overall['average_question_quality']:.3f}")
        
        # Scenario type breakdown
        type_metrics = metrics["scenario_type_metrics"]
        print(f"\n🏥 MEDICAL SCENARIOS:")
        medical = type_metrics["medical_scenarios"]
        print(f"   Total: {medical['total']}")
        print(f"   Successful: {medical['successful']}")
        print(f"   Success Rate: {(medical['successful']/medical['total'])*100:.1f}%" if medical['total'] > 0 else "   Success Rate: N/A")
        print(f"   Avg Quality Score: {medical['avg_quality_score']:.3f}")
        print(f"   Avg Completion Rate: {medical['avg_completion_rate']:.1f}%")
        
        print(f"\n🚫 NON-MEDICAL SCENARIOS (Refusal Testing):")
        non_medical = type_metrics["non_medical_scenarios"]
        print(f"   Total: {non_medical['total']}")
        print(f"   Successful: {non_medical['successful']}")
        print(f"   Refusal Accuracy: {non_medical['refusal_accuracy']:.1f}%")
        
        print(f"\n🔬 EDGE CASE SCENARIOS:")
        edge_case = type_metrics["edge_case_scenarios"]
        print(f"   Total: {edge_case['total']}")
        print(f"   Successful: {edge_case['successful']}")
        print(f"   Success Rate: {(edge_case['successful']/edge_case['total'])*100:.1f}%" if edge_case['total'] > 0 else "   Success Rate: N/A")
        print(f"   Avg Quality Score: {edge_case['avg_quality_score']:.3f}")
        
        # Scenario-specific results
        print(f"\n📋 DETAILED SCENARIO RESULTS:")
        for scenario in results:
            status = "✅ PASS" if scenario.overall_success else "❌ FAIL"
            print(f"   {scenario.scenario_name} ({scenario.scenario_type}): {status}")
            print(f"     Quality: {scenario.avg_quality_score:.3f}, Completion: {scenario.completion_rate:.1f}%, Time: {scenario.avg_response_time:.2f}s")
            if scenario.scenario_type == "non-medical":
                print(f"     Refusal Accuracy: {scenario.refusal_accuracy:.1f}%")
        
        # State-specific metrics
        print(f"\n🔬 STATE-SPECIFIC PERFORMANCE:")
        state_metrics = metrics["state_metrics"]
        for state in ["M", "A", "R", "C", "H", "P", "A2", "W", "S"]:
            if state in state_metrics:
                state_data = state_metrics[state]
                print(f"   {state}: {state_data['success_rate']:.1f}% ({state_data['success_count']}/{state_data['total_attempts']})")
        
        # Performance analysis
        print(f"\n⚡ PERFORMANCE ANALYSIS:")
        if overall['average_response_time'] < 10:
            print("   🚀 Excellent response times (< 10s)")
        elif overall['average_response_time'] < 30:
            print("   ✅ Good response times (< 30s)")
        else:
            print("   ⚠️  Response times could be improved (> 30s)")
        
        if overall['average_quality_score'] >= 0.8:
            print("   🎯 Excellent quality scores (≥ 0.8)")
        elif overall['average_quality_score'] >= 0.6:
            print("   ✅ Good quality scores (≥ 0.6)")
        else:
            print("   ⚠️  Quality scores need improvement (< 0.6)")
        
        if overall['refusal_accuracy'] >= 90:
            print("   🛡️  Excellent refusal accuracy (≥ 90%)")
        elif overall['refusal_accuracy'] >= 80:
            print("   ✅ Good refusal accuracy (≥ 80%)")
        else:
            print("   ⚠️  Refusal accuracy needs improvement (< 80%)")
        
        print(f"\n✅ Ultimate comprehensive test completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    async def run_ultimate_test(self) -> Dict[str, Any]:
        """Run the ultimate comprehensive test"""
        print("🚀 STARTING ULTIMATE COMPREHENSIVE TEST")
        print("=" * 80)
        print("This test combines all previous test suites:")
        print("• 6 Medical scenarios (from comprehensive_test.py)")
        print("• 2 Additional medical scenarios (from comprehensive_test_v2.py)")
        print("• 3 Non-medical refusal tests")
        print("• 2 Medical edge case tests")
        print("• Advanced quality evaluation with 94.2% target")
        print("• Async orchestrator performance testing")
        print("• Comprehensive metrics and reporting")
        print("=" * 80)
        
        scenarios = self.define_test_scenarios()
        all_results = []
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n🔄 Running scenario {i}/{len(scenarios)}: {scenario.scenario_name}")
            result = await self.test_complete_scenario(scenario)
            all_results.append(result)
        
        # Calculate comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(all_results)
        
        # Generate comprehensive report
        self.generate_comprehensive_report(all_results, metrics)
        
        return {
            "scenarios": all_results,
            "metrics": metrics,
            "test_timestamp": datetime.now().isoformat(),
            "test_version": "ultimate_comprehensive_v1.0"
        }

async def main():
    """Main test execution"""
    try:
        tester = UltimateComprehensiveTester()
        
        # Initialize orchestrator
        await tester.initialize()
        
        # Run ultimate test
        results = await tester.run_ultimate_test()
        
        # Save comprehensive results
        results_file = 'ultimate_comprehensive_test_results.json'
        
        # Convert dataclasses to dict for JSON serialization
        json_results = []
        for scenario in results['scenarios']:
            scenario_dict = asdict(scenario)
            # Convert TestResult objects to dicts
            if scenario_dict['results']:
                scenario_dict['results'] = [asdict(r) if hasattr(r, '__dataclass_fields__') else r for r in scenario_dict['results']]
            json_results.append(scenario_dict)
        
        results['scenarios'] = json_results
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Comprehensive results saved to {results_file}")
        
        # Final summary
        overall = results['metrics']['overall_metrics']
        print(f"\n🏆 FINAL SUMMARY:")
        print(f"   Overall Quality Score: {overall['average_quality_score']:.3f}")
        print(f"   Scenario Success Rate: {overall['scenario_success_rate']:.1f}%")
        print(f"   Refusal Accuracy: {overall['refusal_accuracy']:.1f}%")
        print(f"   Citation Accuracy: {overall['average_citation_accuracy']:.3f}")
        
        if overall['average_quality_score'] >= 0.8:
            print("   🎉 EXCELLENT: Quality score meets 80%+ target!")
        elif overall['average_quality_score'] >= 0.6:
            print("   ✅ GOOD: Quality score meets 60%+ target")
        else:
            print("   ⚠️  NEEDS IMPROVEMENT: Quality score below 60%")
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        if 'tester' in locals():
            await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
