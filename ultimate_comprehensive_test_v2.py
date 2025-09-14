#!/usr/bin/env python3
"""
Ultimate Comprehensive Test Suite v2 - Extended Coverage
Tests the MARCH-PAWS RAG system with 10 additional diverse scenarios
covering various medical conditions, edge cases, and challenging situations.
"""

import asyncio
import json
import sys
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.append('src')

from src.orchestrator_async import AsyncOrchestrator
from quality_evaluator import QualityEvaluator
from src.utils import map_citations_to_database

@dataclass
class TestResult:
    """Individual test interaction result"""
    state: str
    question: str
    user_answer: str
    response: Dict[str, Any]
    success: bool
    refusal_detected: bool
    error: Optional[str]
    response_time: float
    overall_quality_score: float
    question_quality_score: float
    checklist_quality_score: float
    citation_accuracy_score: float
    checklist: List[str]
    citations: List[str]

@dataclass
class ScenarioTest:
    """Complete scenario test definition"""
    scenario_name: str
    scenario_description: str
    scenario_type: str
    user_answers: List[str]
    expected_states: List[str]
    expected_refusal: bool
    difficulty_level: str
    medical_complexity: str
    
    # Results attributes (set during testing)
    results: List = None
    overall_success: bool = False
    completion_rate: float = 0.0
    avg_quality_score: float = 0.0
    avg_response_time: float = 0.0
    citation_accuracy: float = 0.0
    refusal_accuracy: float = 0.0

class UltimateComprehensiveTestV2:
    """Extended comprehensive test suite with 10 additional scenarios"""
    
    def __init__(self):
        self.orchestrator = None
        self.evaluator = None
        self.results = []
        
    async def initialize(self):
        """Initialize the test environment"""
        print("🔧 Initializing Extended Test Suite...")
        
        # Initialize quality evaluator
        self.evaluator = QualityEvaluator()
        
        # Initialize async orchestrator
        self.orchestrator = AsyncOrchestrator(
            bm25_path="data/window_bm25_index.pkl",
            embeddings_path="data/window_embeddings.npy",
            metadata_path="data/window_metadata.json"
        )
        
        # Enter async context
        await self.orchestrator.__aenter__()
        
        print("✅ Extended test suite initialized successfully")
    
    async def cleanup(self):
        """Clean up resources"""
        if self.orchestrator:
            await self.orchestrator.__aexit__(None, None, None)
        
    async def test_single_interaction(self, scenario: str, user_answer: str = None, expected_refusal: bool = False) -> TestResult:
        """Test a single interaction with quality evaluation"""
        import time
        start_time = time.time()
        
        try:
            # Run the interaction
            result = await self.orchestrator.run_step(scenario, user_answer or "")
            response_time = time.time() - start_time
            
            # Check for refusal detection (matching v1 logic)
            refusal_detected = result.get('refusal', False) if result else False
            success = (
                result is not None and 
                not (refusal_detected and not expected_refusal) and
                'error' not in result
            )
            
            # Evaluate quality if not a refusal
            if refusal_detected:
                return TestResult(
                    state=result.get('state', ''),
                    question=result.get('question', ''),
                    user_answer=user_answer or '',
                    response=result,
                    success=success,
                    refusal_detected=refusal_detected,
                    error=None,
                    response_time=response_time,
                    overall_quality_score=0.0,
                    question_quality_score=0.0,
                    checklist_quality_score=0.0,
                    citation_accuracy_score=0.0,
                    checklist=[],
                    citations=[]
                )
            
            # Evaluate quality for non-refusal responses
            question = result.get('question', '')
            checklist = result.get('checklist', [])
            citations = result.get('citations', [])
            state = result.get('state', '')
            
            # Map citations to database format
            try:
                citation_db = self.evaluator.citation_data
            except:
                citation_db = self.evaluator.citation_data
            
            mapped_citations = map_citations_to_database(citations, citation_db) if citations else []
            
            # Evaluate individual components
            question_eval = self.evaluator.evaluate_question_quality(question, state, scenario) if question else {"quality_score": 0.0}
            citation_eval = self.evaluator.evaluate_citation_accuracy(mapped_citations) if mapped_citations else {"accuracy_score": 0.0}
            checklist_eval = self.evaluator.evaluate_checklist_quality(checklist, user_answer or "", state) if checklist else {"quality_score": 0.0}
            
            # Overall quality score
            overall_eval = self.evaluator.evaluate_complete_response(
                question, checklist, mapped_citations, user_answer or "", state, scenario
            )
            
            return TestResult(
                state=result.get('state', ''),
                question=result.get('question', ''),
                user_answer=user_answer or '',
                response=result,
                success=success,
                refusal_detected=refusal_detected,
                error=None,
                response_time=response_time,
                overall_quality_score=overall_eval['overall_score'],
                question_quality_score=question_eval['quality_score'],
                checklist_quality_score=checklist_eval['quality_score'],
                citation_accuracy_score=citation_eval['accuracy_score'],
                checklist=result.get('checklist', []),
                citations=result.get('citations', [])
            )
            
        except Exception as e:
            return TestResult(
                state='',
                question='',
                user_answer=user_answer or '',
                response={},
                success=False,
                refusal_detected=False,
                error=str(e),
                response_time=time.time() - start_time,
                overall_quality_score=0.0,
                question_quality_score=0.0,
                checklist_quality_score=0.0,
                citation_accuracy_score=0.0,
                checklist=[],
                citations=[]
            )
    
    async def test_complete_scenario(self, scenario_test: ScenarioTest) -> ScenarioTest:
        """Test a complete scenario with comprehensive evaluation"""
        print(f"\n🧪 Testing Scenario: {scenario_test.scenario_name}")
        print(f"📝 Description: {scenario_test.scenario_description}")
        print(f"🏷️  Type: {scenario_test.scenario_type}")
        print(f"🎯 Expected Refusal: {scenario_test.expected_refusal}")
        print(f"📊 Difficulty: {scenario_test.difficulty_level} | Complexity: {scenario_test.medical_complexity}")
        print("=" * 80)
        
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
            scenario_test.avg_quality_score = 0.0
            scenario_test.avg_response_time = result.response_time
            scenario_test.citation_accuracy = 0.0
            return scenario_test
        
        print(f"✅ State {result.state}: {result.question}")
        print(f"📊 Quality Score: {result.overall_quality_score:.3f}")
        if result.checklist:
            print(f"   📋 Checklist: {len(result.checklist)} items")
        if result.citations:
            print(f"   📚 Citations: {len(result.citations)} items (Accuracy: {result.citation_accuracy_score:.3f})")
        
        # Test each subsequent state with user answers (for medical scenarios and edge cases)
        if scenario_test.scenario_type in ["medical", "medical-edge-case", "complex-medical"] and not scenario_test.expected_refusal:
            state_sequence = ['A', 'R', 'C', 'H', 'P', 'A2', 'W', 'S']
            for i, user_answer in enumerate(scenario_test.user_answers):
                current_state = state_sequence[i] if i < len(state_sequence) else 'END'
                next_state = state_sequence[i + 1] if i + 1 < len(state_sequence) else 'END'
                
                print(f"\n👤 User Answer: {user_answer}")
                print(f"Current state: {current_state}")
                
                result = await self.test_single_interaction(current_scenario, user_answer)
                results.append(result)
                
                if not result.success:
                    print(f"❌ State {current_state} failed: {result.error}")
                    break
                
                print(f"✅ State {result.state}: {result.question}")
                print(f"📊 Quality Score: {result.overall_quality_score:.3f}")
                if result.checklist:
                    print(f"   📋 Checklist: {len(result.checklist)} items")
                if result.citations:
                    print(f"   📚 Citations: {len(result.citations)} items (Accuracy: {result.citation_accuracy_score:.3f})")
                print(f"🔄 State: {current_state} → {result.state}")
        
        # Calculate scenario metrics (matching v1 logic)
        successful_interactions = sum(1 for r in results if r.success and not r.refusal_detected)
        total_interactions = len([r for r in results if not r.refusal_detected])
        
        # Calculate completion rate
        completion_rate = (successful_interactions / len(scenario_test.expected_states)) * 100 if scenario_test.expected_states else 0.0
        
        # Calculate quality metrics
        quality_scores = [r.overall_quality_score for r in results if not r.refusal_detected]
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Calculate citation accuracy
        citation_scores = [r.citation_accuracy_score for r in results if not r.refusal_detected]
        citation_accuracy = sum(citation_scores) / len(citation_scores) if citation_scores else 0.0
        
        # Refusal accuracy (matching v1 logic)
        first_result = results[0] if results else None
        refusal_accuracy = 100.0 if first_result and first_result.refusal_detected == scenario_test.expected_refusal else 0.0
        
        scenario_test.results = results
        scenario_test.overall_success = completion_rate >= 80.0 if not scenario_test.expected_refusal else refusal_accuracy == 100.0
        scenario_test.completion_rate = completion_rate
        scenario_test.avg_quality_score = avg_quality_score
        scenario_test.avg_response_time = sum(r.response_time for r in results) / len(results)
        scenario_test.citation_accuracy = citation_accuracy
        scenario_test.refusal_accuracy = refusal_accuracy
        
        # Print scenario summary
        print(f"\n📊 SCENARIO RESULTS:")
        print(f"   States Completed: {successful_interactions}/{len(scenario_test.expected_states)}")
        print(f"   Completion Rate: {scenario_test.completion_rate:.1f}%")
        print(f"   Avg Quality Score: {scenario_test.avg_quality_score:.3f}")
        print(f"   Avg Response Time: {scenario_test.avg_response_time:.2f}s")
        print(f"   Citation Accuracy: {scenario_test.citation_accuracy:.3f}")
        print(f"   Refusal Accuracy: {scenario_test.refusal_accuracy:.1f}%")
        print(f"🎯 Overall Success: {'✅' if scenario_test.overall_success else '❌'}")
        
        return scenario_test
    
    def get_extended_test_scenarios(self) -> List[ScenarioTest]:
        """Get the 10 additional diverse test scenarios"""
        return [
            # Scenario 1: Complex Multi-Trauma with Shock
            ScenarioTest(
                scenario_name="Complex Multi-Trauma with Shock",
                scenario_description="Motorcycle accident with multiple injuries: open femur fracture with bleeding, chest trauma, head injury, and signs of shock",
                scenario_type="complex-medical",
                user_answers=[
                    "Yes, massive bleeding from the thigh, patient is pale and unresponsive",
                    "Airway is compromised - patient is unconscious with snoring sounds",
                    "Breathing is irregular and shallow, chest wound is bubbling",
                    "No radial pulse detected, patient is in severe shock",
                    "Patient is unconscious and hypothermic from blood loss",
                    "Patient is unresponsive to pain stimuli",
                    "Open fractures require antibiotics, no known allergies",
                    "Multiple injuries: thigh laceration, chest wound, head injury",
                    "Obvious femur fracture, needs immediate splinting and evacuation"
                ],
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"],
                expected_refusal=False,
                difficulty_level="High",
                medical_complexity="Critical"
            ),
            
            # Scenario 2: Burn Victim with Inhalation Injury
            ScenarioTest(
                scenario_name="Burn Victim with Inhalation Injury",
                scenario_description="House fire victim with extensive burns, facial burns, hoarse voice, and possible inhalation injury",
                scenario_type="medical",
                user_answers=[
                    "No external bleeding, but severe burns present",
                    "Airway compromised - hoarse voice, singed nasal hairs",
                    "Breathing is labored with stridor, possible inhalation injury",
                    "Weak pulse due to fluid loss from burns",
                    "Patient is hypothermic from heat loss through burns",
                    "Severe pain from burns, patient is conscious but distressed",
                    "Burns require antibiotics, no known allergies",
                    "Extensive burns on face, arms, and torso",
                    "No fractures, but burns may affect mobility"
                ],
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"],
                expected_refusal=False,
                difficulty_level="High",
                medical_complexity="Critical"
            ),
            
            # Scenario 3: Pediatric Emergency
            ScenarioTest(
                scenario_name="Pediatric Emergency",
                scenario_description="5-year-old child with severe abdominal pain, vomiting, fever, and signs of dehydration",
                scenario_type="medical-edge-case",
                user_answers=[
                    "No bleeding visible",
                    "Child is crying and talking, airway is clear",
                    "Breathing is normal but child is distressed",
                    "Weak pulse, signs of dehydration",
                    "No hypothermia, child feels warm to touch",
                    "Child reports severe stomach pain, crying continuously",
                    "Food poisoning may require antibiotics, check for allergies",
                    "No visible wounds, just abdominal distension",
                    "No fractures detected"
                ],
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"],
                expected_refusal=False,
                difficulty_level="Medium",
                medical_complexity="Moderate"
            ),
            
            # Scenario 4: Elderly Fall with Hip Fracture
            ScenarioTest(
                scenario_name="Elderly Fall with Hip Fracture",
                scenario_description="85-year-old patient who fell down stairs with severe hip pain, unable to bear weight, and signs of confusion",
                scenario_type="medical",
                user_answers=[
                    "No external bleeding visible",
                    "Patient is conscious but confused, airway is clear",
                    "Breathing is normal but patient is in pain",
                    "Weak pulse, patient appears pale",
                    "Patient is hypothermic and confused from shock",
                    "Severe hip pain, patient rates it 9/10",
                    "No antibiotics needed, no known allergies",
                    "No visible wounds, just hip deformity",
                    "Obvious hip fracture, leg is shortened and externally rotated"
                ],
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"],
                expected_refusal=False,
                difficulty_level="Medium",
                medical_complexity="Moderate"
            ),
            
            # Scenario 5: Allergic Reaction with Anaphylaxis
            ScenarioTest(
                scenario_name="Allergic Reaction with Anaphylaxis",
                scenario_description="Patient with bee sting allergy showing signs of anaphylaxis: difficulty breathing, facial swelling, hives, and rapid pulse",
                scenario_type="medical",
                user_answers=[
                    "No bleeding from the sting site",
                    "Airway is compromised - facial swelling and difficulty breathing",
                    "Breathing is labored with wheezing",
                    "Rapid pulse at 140 bpm, signs of shock",
                    "No hypothermia, patient feels warm",
                    "Patient reports severe itching and burning sensation",
                    "Anaphylaxis requires epinephrine, patient has known allergies",
                    "Visible hives and swelling at sting site",
                    "No fractures, but patient may collapse"
                ],
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"],
                expected_refusal=False,
                difficulty_level="High",
                medical_complexity="Critical"
            ),
            
            # Scenario 6: Stroke with Neurological Deficit
            ScenarioTest(
                scenario_name="Stroke with Neurological Deficit",
                scenario_description="65-year-old patient with sudden onset of facial droop, slurred speech, and weakness on one side of the body",
                scenario_type="medical",
                user_answers=[
                    "No bleeding visible",
                    "Airway is clear but speech is slurred",
                    "Breathing is normal",
                    "Irregular pulse, possible atrial fibrillation",
                    "No hypothermia, but patient is confused",
                    "Patient cannot rate pain due to confusion",
                    "No antibiotics needed, no known allergies",
                    "No visible wounds, but facial asymmetry present",
                    "No fractures, but weakness in left arm and leg"
                ],
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"],
                expected_refusal=False,
                difficulty_level="High",
                medical_complexity="Critical"
            ),
            
            # Scenario 7: Drug Overdose with Respiratory Depression
            ScenarioTest(
                scenario_name="Drug Overdose with Respiratory Depression",
                scenario_description="Unconscious patient found with empty pill bottles, slow breathing, pinpoint pupils, and signs of overdose",
                scenario_type="medical",
                user_answers=[
                    "No bleeding visible",
                    "Airway is compromised - patient is unconscious",
                    "Breathing is very slow and shallow, 6 breaths per minute",
                    "Weak pulse, patient is pale",
                    "Patient is hypothermic and unconscious",
                    "Patient is unresponsive to pain",
                    "Overdose may require naloxone, check for allergies",
                    "No visible wounds, but empty pill bottles nearby",
                    "No fractures detected"
                ],
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"],
                expected_refusal=False,
                difficulty_level="High",
                medical_complexity="Critical"
            ),
            
            # Scenario 8: Heat Stroke in Athlete
            ScenarioTest(
                scenario_name="Heat Stroke in Athlete",
                scenario_description="Marathon runner found unconscious with high body temperature, dry skin, and altered mental status",
                scenario_type="medical",
                user_answers=[
                    "No bleeding visible",
                    "Airway is clear but patient is unconscious",
                    "Breathing is rapid and shallow",
                    "Rapid weak pulse, signs of dehydration",
                    "Patient is hyperthermic, body temperature feels very hot",
                    "Patient is unconscious, cannot assess pain",
                    "Heat stroke requires cooling, no antibiotics needed",
                    "No visible wounds, just signs of heat exposure",
                    "No fractures, but patient may have muscle cramps"
                ],
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"],
                expected_refusal=False,
                difficulty_level="Medium",
                medical_complexity="Moderate"
            ),
            
            # Scenario 9: Non-Medical Query (Refusal Test)
            ScenarioTest(
                scenario_name="Cooking Recipe Query",
                scenario_description="How do I make chocolate chip cookies?",
                scenario_type="non-medical",
                user_answers=[],
                expected_states=[],
                expected_refusal=True,
                difficulty_level="Low",
                medical_complexity="None"
            ),
            
            # Scenario 10: Minor Medical Issue (Edge Case)
            ScenarioTest(
                scenario_name="Minor Cut with Infection",
                scenario_description="Small cut on finger that has become red, swollen, and painful with pus",
                scenario_type="medical-edge-case",
                user_answers=[
                    "No life-threatening bleeding, just small cut",
                    "Airway is clear, patient is breathing normally",
                    "Breathing is normal",
                    "Normal pulse, no signs of systemic infection",
                    "No hypothermia, but finger feels warm",
                    "Moderate pain at cut site, rates 4/10",
                    "Infected cut may need antibiotics, no known allergies",
                    "Small infected cut on index finger",
                    "No fractures, just soft tissue infection"
                ],
                expected_states=["M", "A", "R", "C", "H", "P", "A2", "W", "S"],
                expected_refusal=False,
                difficulty_level="Low",
                medical_complexity="Minor"
            )
        ]
    
    async def run_extended_test(self) -> Dict[str, Any]:
        """Run the extended comprehensive test suite"""
        print("🚀 STARTING ULTIMATE COMPREHENSIVE TEST SUITE V2")
        print("=" * 80)
        print("This extended test suite adds 10 additional diverse scenarios:")
        print("• 7 Complex medical scenarios (multi-trauma, burns, pediatric, elderly)")
        print("• 1 Non-medical refusal test")
        print("• 2 Medical edge cases (minor conditions)")
        print("• Advanced quality evaluation with 94.2% target")
        print("• Async orchestrator performance testing")
        print("• Comprehensive metrics and reporting")
        print("=" * 80)
        
        scenarios = self.get_extended_test_scenarios()
        results = []
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n🔄 Running scenario {i}/{len(scenarios)}: {scenario.scenario_name}")
            result = await self.test_complete_scenario(scenario)
            results.append(result)
        
        # Calculate comprehensive metrics
        all_results = results
        successful_scenarios = sum(1 for r in all_results if r.overall_success)
        total_interactions = sum(len(r.results) for r in all_results)
        successful_interactions = sum(sum(1 for tr in r.results if tr.success) for r in all_results)
        
        # Calculate quality metrics
        quality_scores = [r.avg_quality_score for r in all_results if not r.expected_refusal]
        citation_scores = [r.citation_accuracy for r in all_results if not r.expected_refusal]
        response_times = [r.avg_response_time for r in all_results]
        
        # Scenario type breakdown
        medical_scenarios = [r for r in all_results if r.scenario_type == "medical"]
        complex_medical_scenarios = [r for r in all_results if r.scenario_type == "complex-medical"]
        edge_case_scenarios = [r for r in all_results if r.scenario_type == "medical-edge-case"]
        non_medical_scenarios = [r for r in all_results if r.scenario_type == "non-medical"]
        
        # Difficulty breakdown
        high_difficulty = [r for r in all_results if r.difficulty_level == "High"]
        medium_difficulty = [r for r in all_results if r.difficulty_level == "Medium"]
        low_difficulty = [r for r in all_results if r.difficulty_level == "Low"]
        
        metrics = {
            "overall_metrics": {
                "total_scenarios": len(all_results),
                "successful_scenarios": successful_scenarios,
                "scenario_success_rate": (successful_scenarios / len(all_results)) * 100,
                "total_interactions": total_interactions,
                "successful_interactions": successful_interactions,
                "interaction_success_rate": (successful_interactions / total_interactions) * 100 if total_interactions > 0 else 0,
                "average_response_time": sum(response_times) / len(response_times) if response_times else 0,
                "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                "average_citation_accuracy": sum(citation_scores) / len(citation_scores) if citation_scores else 0,
                "refusal_accuracy": sum(1 for r in non_medical_scenarios if r.refusal_accuracy == 100.0) / len(non_medical_scenarios) * 100 if non_medical_scenarios else 100.0
            },
            "scenario_type_metrics": {
                "medical_scenarios": {
                    "total": len(medical_scenarios),
                    "successful": sum(1 for r in medical_scenarios if r.overall_success),
                    "avg_quality_score": sum(r.avg_quality_score for r in medical_scenarios) / len(medical_scenarios) if medical_scenarios else 0,
                    "avg_completion_rate": sum(r.completion_rate for r in medical_scenarios) / len(medical_scenarios) if medical_scenarios else 0
                },
                "complex_medical_scenarios": {
                    "total": len(complex_medical_scenarios),
                    "successful": sum(1 for r in complex_medical_scenarios if r.overall_success),
                    "avg_quality_score": sum(r.avg_quality_score for r in complex_medical_scenarios) / len(complex_medical_scenarios) if complex_medical_scenarios else 0,
                    "avg_completion_rate": sum(r.completion_rate for r in complex_medical_scenarios) / len(complex_medical_scenarios) if complex_medical_scenarios else 0
                },
                "edge_case_scenarios": {
                    "total": len(edge_case_scenarios),
                    "successful": sum(1 for r in edge_case_scenarios if r.overall_success),
                    "avg_quality_score": sum(r.avg_quality_score for r in edge_case_scenarios) / len(edge_case_scenarios) if edge_case_scenarios else 0,
                    "avg_completion_rate": sum(r.completion_rate for r in edge_case_scenarios) / len(edge_case_scenarios) if edge_case_scenarios else 0
                },
                "non_medical_scenarios": {
                    "total": len(non_medical_scenarios),
                    "successful": sum(1 for r in non_medical_scenarios if r.overall_success),
                    "refusal_accuracy": sum(r.refusal_accuracy for r in non_medical_scenarios) / len(non_medical_scenarios) if non_medical_scenarios else 100.0
                }
            },
            "difficulty_metrics": {
                "high_difficulty": {
                    "total": len(high_difficulty),
                    "successful": sum(1 for r in high_difficulty if r.overall_success),
                    "success_rate": sum(1 for r in high_difficulty if r.overall_success) / len(high_difficulty) * 100 if high_difficulty else 0,
                    "avg_quality_score": sum(r.avg_quality_score for r in high_difficulty) / len(high_difficulty) if high_difficulty else 0
                },
                "medium_difficulty": {
                    "total": len(medium_difficulty),
                    "successful": sum(1 for r in medium_difficulty if r.overall_success),
                    "success_rate": sum(1 for r in medium_difficulty if r.overall_success) / len(medium_difficulty) * 100 if medium_difficulty else 0,
                    "avg_quality_score": sum(r.avg_quality_score for r in medium_difficulty) / len(medium_difficulty) if medium_difficulty else 0
                },
                "low_difficulty": {
                    "total": len(low_difficulty),
                    "successful": sum(1 for r in low_difficulty if r.overall_success),
                    "success_rate": sum(1 for r in low_difficulty if r.overall_success) / len(low_difficulty) * 100 if low_difficulty else 0,
                    "avg_quality_score": sum(r.avg_quality_score for r in low_difficulty) / len(low_difficulty) if low_difficulty else 0
                }
            }
        }
        
        # Print comprehensive report
        self.print_extended_report(metrics, all_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ultimate_comprehensive_test_v2_results_{timestamp}.json"
        
        output_data = {
            "test_metadata": {
                "test_name": "Ultimate Comprehensive Test Suite V2",
                "test_version": "v2.0",
                "timestamp": datetime.now().isoformat(),
                "total_scenarios": len(all_results),
                "scenarios_added": 10
            },
            "scenarios": [asdict(result) for result in all_results],
            "metrics": metrics
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\n💾 Extended test results saved to {filename}")
        print(f"✅ Ultimate comprehensive test v2 completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return metrics
    
    def print_extended_report(self, metrics: Dict[str, Any], results: List[ScenarioTest]):
        """Print comprehensive extended test report"""
        print("\n" + "=" * 80)
        print("🏆 ULTIMATE COMPREHENSIVE TEST REPORT V2")
        print("=" * 80)
        
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
        
        # Scenario type breakdown
        type_metrics = metrics["scenario_type_metrics"]
        print(f"\n🏥 MEDICAL SCENARIOS:")
        medical = type_metrics["medical_scenarios"]
        print(f"   Total: {medical['total']}")
        print(f"   Successful: {medical['successful']}")
        print(f"   Success Rate: {(medical['successful']/medical['total'])*100:.1f}%" if medical['total'] > 0 else "   Success Rate: N/A")
        print(f"   Avg Quality Score: {medical['avg_quality_score']:.3f}")
        print(f"   Avg Completion Rate: {medical['avg_completion_rate']:.1f}%")
        
        print(f"\n🔬 COMPLEX MEDICAL SCENARIOS:")
        complex_med = type_metrics["complex_medical_scenarios"]
        print(f"   Total: {complex_med['total']}")
        print(f"   Successful: {complex_med['successful']}")
        print(f"   Success Rate: {(complex_med['successful']/complex_med['total'])*100:.1f}%" if complex_med['total'] > 0 else "   Success Rate: N/A")
        print(f"   Avg Quality Score: {complex_med['avg_quality_score']:.3f}")
        print(f"   Avg Completion Rate: {complex_med['avg_completion_rate']:.1f}%")
        
        print(f"\n🔬 EDGE CASE SCENARIOS:")
        edge_case = type_metrics["edge_case_scenarios"]
        print(f"   Total: {edge_case['total']}")
        print(f"   Successful: {edge_case['successful']}")
        print(f"   Success Rate: {(edge_case['successful']/edge_case['total'])*100:.1f}%" if edge_case['total'] > 0 else "   Success Rate: N/A")
        print(f"   Avg Quality Score: {edge_case['avg_quality_score']:.3f}")
        print(f"   Avg Completion Rate: {edge_case['avg_completion_rate']:.1f}%")
        
        print(f"\n🚫 NON-MEDICAL SCENARIOS (Refusal Testing):")
        non_medical = type_metrics["non_medical_scenarios"]
        print(f"   Total: {non_medical['total']}")
        print(f"   Successful: {non_medical['successful']}")
        print(f"   Refusal Accuracy: {non_medical['refusal_accuracy']:.1f}%")
        
        # Difficulty breakdown
        difficulty_metrics = metrics["difficulty_metrics"]
        print(f"\n📈 DIFFICULTY BREAKDOWN:")
        for difficulty, stats in difficulty_metrics.items():
            print(f"   {difficulty.replace('_', ' ').title()}:")
            print(f"     Total: {stats['total']}")
            print(f"     Successful: {stats['successful']}")
            print(f"     Success Rate: {stats['success_rate']:.1f}%")
            print(f"     Avg Quality Score: {stats['avg_quality_score']:.3f}")
        
        # Scenario-specific results
        print(f"\n📋 DETAILED SCENARIO RESULTS:")
        for scenario in results:
            status = "✅ PASS" if scenario.overall_success else "❌ FAIL"
            print(f"   {scenario.scenario_name} ({scenario.scenario_type}): {status}")
            print(f"     Quality: {scenario.avg_quality_score:.3f}, Completion: {scenario.completion_rate:.1f}%, Time: {scenario.avg_response_time:.2f}s")
            print(f"     Difficulty: {scenario.difficulty_level}, Complexity: {scenario.medical_complexity}")
            if scenario.scenario_type == "non-medical":
                print(f"     Refusal Accuracy: {scenario.refusal_accuracy:.1f}%")
        
        # Performance analysis
        print(f"\n⚡ PERFORMANCE ANALYSIS:")
        if overall['average_response_time'] < 10:
            print("   ✅ Excellent response times (< 10s)")
        elif overall['average_response_time'] < 20:
            print("   ✅ Good response times (< 20s)")
        else:
            print("   ⚠️ Response times could be improved (> 20s)")
        
        if overall['average_quality_score'] >= 0.8:
            print("   🎯 Excellent quality scores (≥ 0.8)")
        elif overall['average_quality_score'] >= 0.7:
            print("   ✅ Good quality scores (≥ 0.7)")
        else:
            print("   ⚠️ Quality scores need improvement (< 0.7)")
        
        if overall['refusal_accuracy'] >= 90:
            print("   🛡️ Excellent refusal accuracy (≥ 90%)")
        else:
            print("   ⚠️ Refusal accuracy needs improvement (< 90%)")
        
        print(f"\n🏆 FINAL SUMMARY:")
        print(f"   Overall Quality Score: {overall['average_quality_score']:.3f}")
        print(f"   Scenario Success Rate: {overall['scenario_success_rate']:.1f}%")
        print(f"   Refusal Accuracy: {overall['refusal_accuracy']:.1f}%")
        print(f"   Citation Accuracy: {overall['average_citation_accuracy']:.3f}")
        
        if overall['scenario_success_rate'] >= 90 and overall['average_quality_score'] >= 0.8:
            print("   🎉 EXCELLENT: System meets all performance targets!")
        elif overall['scenario_success_rate'] >= 80 and overall['average_quality_score'] >= 0.7:
            print("   ✅ GOOD: System meets most performance targets!")
        else:
            print("   ⚠️ NEEDS IMPROVEMENT: System requires optimization!")

async def main():
    """Main test execution"""
    test_suite = UltimateComprehensiveTestV2()
    
    await test_suite.initialize()
    try:
        await test_suite.run_extended_test()
    finally:
        await test_suite.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
