# MARCH‑PAWS RAG Assistant (Offline)

This project contains a robust, production-ready implementation of an offline Retrieval Augmented Generation (RAG) system designed to read medical PDF manuals (such as the **Tactical Combat Casualty Care Handbook**) and produce step‑by‑step procedural checklists following the **MARCH‑PAWS** protocol. The system enforces a finite‑state ordering and always grounds its answers in the source document with intelligent stage-specific question generation.

The code is designed to run on a laptop‑class machine completely offline. Internet connectivity is **not** required at inference time once the model weights and indexes have been prepared. The only external dependency at runtime is a local Large Language Model (LLM) such as Mistral‑7B served via [Ollama](https://github.com/ollama/ollama).

## 🚀 Recent Achievements & Improvements

### ✅ **Consistency & Quality Improvements**
- **Fixed Examiner-Style Questions**: Eliminated "Is it necessary to..." and "What actions..." questions
- **Anatomical Consistency**: Questions now match scenario anatomy (no chest questions for arm injuries)
- **Stage-Specific Focus**: Each MARCH-PAWS stage generates appropriate questions only
- **Robust Prompt Engineering**: Enhanced prompts prevent cross-stage contamination

### ✅ **Advanced Retrieval System**
- **Hybrid Retrieval**: BM25 + FAISS + Cross-encoder with Reciprocal Rank Fusion (RRF)
- **Adaptive α Parameter**: Dynamic weighting based on query characteristics
- **Memory-Optimized FAISS**: Uses `IndexFlatIP` with memory mapping for efficiency
- **Lemmatized BM25**: Better lexical matching with NLTK lemmatization
- **Cross-Encoder Re-ranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` for precision

### ✅ **Optimized Performance**
- **Temperature Differentiation**: Higher temp (0.2) for questions, zero temp for answers
- **Lowered Thresholds**: Better coverage with `retrieval_threshold=0.005`
- **Focused Retrieval**: Reduced `max_chunks=6` for more relevant content
- **HTTP Keep-Alive**: Persistent connections for faster LLM communication

### ✅ **Production-Ready Features**
- **Streamlit Web Interface**: User-friendly medical guidance application
- **Comprehensive Error Handling**: Multiple guardrails prevent hallucinations
- **Memory Efficiency**: Optimized for laptops and small devices (4.5-7.5GB RAM)
- **Offline Operation**: Complete independence from internet connectivity

## Key Features

- **🎯 Intelligent Stage-Specific Question Generation**: Contextually appropriate questions for each MARCH-PAWS stage, ignoring anatomical location bias
- **🔍 Advanced Hybrid Retrieval**: Combines BM25 lexical search with FAISS dense vector search using Reciprocal Rank Fusion (RRF)
- **🎯 Cross-Encoder Re-ranking**: Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` for precise relevance scoring
- **📄 Window-Based Chunking**: Overlapping text windows with semantic boundary detection for better context preservation
- **💾 Memory-Optimized FAISS**: Uses `IndexFlatIP` with memory mapping for efficient retrieval
- **📝 Robust Prompt Engineering**: Stage-specific definitions prevent inappropriate question generation
- **🌐 Streamlit Web Interface**: User-friendly web application for interactive medical guidance
- **🛡️ Comprehensive Error Handling**: Multiple guardrails prevent hallucinations and ensure medical accuracy
- **⚡ Performance Optimized**: Designed for small devices with SLMs (4.5-7.5GB RAM usage)

## Repository Structure

```
marchpaws_rag/
├── data/                              # Data directory
│   ├── tc4-02.1wc1x2.pdf             # Source PDF manual (TCCC Handbook)
│   ├── tc4-02.1_sections.jsonl       # Extracted sections from PDF
│   ├── windows.jsonl                  # Windowed text chunks
│   ├── window_metadata.json           # Window metadata
│   ├── window_embeddings.npy          # Dense embeddings
│   ├── window_embeddings_optimized.index  # Optimized FAISS index
│   ├── window_bm25_index.pkl          # BM25 index
│   └── embedding_info.json            # Embedding metadata
├── src/
│   ├── parse_tc4021.py               # PDF extraction and section parsing
│   ├── make_windows.py                # Window-based chunking
│   ├── build_window_bm25.py           # BM25 index construction
│   ├── embed_windows.py               # Dense embedding generation
│   ├── build_optimized_faiss.py       # Optimized FAISS index creation
│   ├── retriever.py                   # Hybrid retrieval system
│   ├── orchestrator.py                # Main RAG pipeline orchestrator
│   ├── prompts.py                     # Stage-specific prompt templates
│   ├── fsm.py                         # MARCH-PAWS finite state machine
│   └── utils.py                       # Utility functions
├── app.py                             # Streamlit web interface
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Your PDF Manual

Place your medical manual PDF in `data/tc4-02.1wc1x2.pdf` (TCCC Handbook included).

### 3. Extract and Process the PDF

```bash
# Extract sections from PDF
python src/parse_tc4021.py --pdf data/tc4-02.1wc1x2.pdf --out data/tc4-02.1_sections.jsonl

# Create overlapping text windows
python src/make_windows.py --sections data/tc4-02.1_sections.jsonl --out data/windows.jsonl

# Build BM25 index with lemmatization
python src/build_window_bm25.py --windows data/windows.jsonl --out data/window_bm25_index.pkl

# Generate dense embeddings
python src/embed_windows.py --windows data/windows.jsonl --out data/window_embeddings.npy

# Build optimized FAISS index
python src/build_optimized_faiss.py --embeddings data/window_embeddings.npy
```

### 4. Start the LLM Server

Using Ollama (recommended):

```bash
# Install Ollama and download Mistral 7B
ollama pull mistral:latest
ollama serve &
```

### 5. Launch the Web Interface

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## MARCH‑PAWS Protocol

The system enforces the following medical assessment sequence:

1. **M** – **Massive Hemorrhage Control**: Locate life-threatening bleeding and apply tourniquet, packing, or direct pressure
2. **A** – **Airway Management**: Ensure airway patency and breathing
3. **R** – **Respiration Assessment**: Check breathing effectiveness and chest wounds
4. **C** – **Circulation Assessment**: Evaluate pulse, perfusion, and shock
5. **H** – **Hypothermia Prevention**: Prevent heat loss and assess head injury
6. **P** – **Pain Management**: Assess pain level and provide analgesia
7. **A2** – **Antibiotics**: Determine antibiotic needs for penetrating wounds
8. **W** – **Wound Reassessment**: Inspect for missed injuries and verify dressings
9. **S** – **Splinting**: Identify fractures and provide immobilization

## Technical Architecture

### Retrieval System

- **Hybrid Retrieval**: Combines BM25 lexical search with FAISS dense vector search
- **Reciprocal Rank Fusion (RRF)**: Intelligently combines results from both retrieval methods with adaptive α
- **Cross-Encoder Re-ranking**: Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` for precise relevance scoring
- **Lemmatization**: BM25 queries are lemmatized for better lexical matching
- **Memory Optimization**: Uses `IndexFlatIP` with memory mapping for efficient retrieval

### Chunking Strategy

- **Window-Based Chunking**: Creates overlapping text windows with configurable stride
- **Semantic Boundary Detection**: Identifies paragraph boundaries for better context preservation
- **Metadata Preservation**: Maintains section citations, page numbers, and heading information
- **Token-Aware**: Respects token limits while maximizing context overlap

### Prompt Engineering

- **Stage-Specific Definitions**: Clear, explicit definitions for each MARCH-PAWS stage
- **Bias Prevention**: Instructions to ignore anatomical location bias in question generation
- **Consistency Guidelines**: Ensures questions focus on stage requirements, not injury location
- **Medical Accuracy**: Uses appropriate medical terminology and procedures
- **Question vs Answer Differentiation**: Different temperature settings for natural questions vs deterministic answers

### Performance Optimizations

- **Memory-Mapped FAISS**: Efficient loading of large embedding indexes
- **Sentence-Transformer Warmup**: JIT compilation optimization for faster inference
- **Batched Cross-Encoder**: Efficient re-ranking of multiple query-result pairs
- **HTTP Keep-Alive**: Persistent connections for LLM communication
- **Temperature Control**: Different temperatures for questions (0.2) and answers (0.0)

## Configuration

### Environment Variables

```bash
export LLM_ENDPOINT="http://localhost:11434/api/generate"  # Ollama endpoint
export LLM_MODEL="mistral:latest"                          # LLM model name
```

### Retrieval Parameters

- `k=20`: Number of initial candidates retrieved
- `max_chunks=6`: Maximum chunks used in final response (optimized)
- `ce_threshold=0.0001`: Cross-encoder score threshold
- `retrieval_threshold=0.005`: Minimum relevance score for responses (lowered for better coverage)

## Usage Examples

### Web Interface

1. Open `http://localhost:8501` in your browser
2. Enter a medical scenario (e.g., "gunshot wound to the chest")
3. Answer the system's questions following the MARCH-PAWS sequence
4. Receive step-by-step medical guidance with citations

### Programmatic Usage

```python
from src.orchestrator import Orchestrator

# Initialize the system
orc = Orchestrator(
    bm25_path='data/window_bm25_index.pkl',
    embeddings_path='data/window_embeddings.npy',
    metadata_path='data/window_metadata.json'
)

# Process a medical scenario
result = orc.run_step("gunshot wound to the chest")
print(f"Question: {result['question']}")
print(f"State: {result['state']}")
```

## Quality Assurance

### Guardrails

- **Retrieval Score Threshold**: Refuses responses when no relevant content is found
- **Citation Requirements**: All checklist items must have proper citations
- **JSON Schema Validation**: Enforces structured output format
- **Stage Focus Validation**: Prevents inappropriate question generation
- **Medical Accuracy**: Grounds all responses in source documentation
- **Anatomical Consistency**: Questions match scenario anatomy

### Error Handling

- **Graceful Degradation**: Falls back to BM25-only when FAISS fails
- **Timeout Protection**: Prevents hanging on LLM requests
- **Memory Management**: Efficient handling of large embedding files
- **Robust Parsing**: Handles various LLM response formats
- **Validation Loops**: Regenerates responses on validation failure

## Memory Usage & Device Compatibility

### Memory Requirements

| Component | Memory Usage |
|-----------|-------------|
| Mistral 7B (4-bit) | ~4GB |
| Sentence-Transformer | ~150MB |
| Cross-Encoder | ~150MB |
| FAISS Index | ~50MB |
| BM25 Index | ~20MB |
| Python Runtime | ~100MB |
| **Total** | **~4.5GB** |

### Device Compatibility

| Device Type | RAM | Feasibility | Recommended |
|-------------|-----|-------------|-------------|
| High-end Laptop | 16GB+ | ✅ Excellent | Full features |
| Mid-range Laptop | 8GB | ✅ Good | Optimized settings |
| Low-end Laptop | 4GB | ⚠️ Tight | Use Phi-3 Mini |
| Raspberry Pi 5 | 8GB | ✅ Good | Sequential processing |

## Dependencies

### Core Requirements

- `streamlit>=1.28.0` - Web interface
- `sentence-transformers>=2.2.2` - Dense embeddings
- `faiss-cpu>=1.7.4` - Vector similarity search
- `rank-bm25>=0.2.2` - Lexical search
- `transformers>=4.30.0` - Cross-encoder models
- `nltk>=3.8.1` - Text preprocessing
- `numpy>=1.24.0` - Numerical operations
- `requests>=2.31.0` - HTTP communication

### Optional Optimizations

- `optimum>=1.16.0` - Model optimization
- `intel-extension-for-pytorch>=2.0.0` - CPU acceleration
- `psutil>=5.9.0` - Memory monitoring

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   - Ensure Ollama is running: `ollama serve`
   - Check endpoint: `curl http://localhost:11434/api/generate`

2. **FAISS Index Not Found**
   - Rebuild the index: `python src/build_optimized_faiss.py --embeddings data/window_embeddings.npy`

3. **Memory Issues**
   - Use smaller batch sizes in `embed_windows.py`
   - Consider using quantized models (Phi-3 Mini for 4GB devices)

4. **Inconsistent Questions**
   - Check stage definitions in `src/prompts.py`
   - Verify LLM temperature settings (0.2 for questions, 0.0 for answers)

5. **Anatomical Inconsistency**
   - Ensure proper stage definitions are being used
   - Check that anatomical filtering is working correctly

### Performance Tuning

- **Increase Retrieval Candidates**: Modify `k` parameter in `orchestrator.py`
- **Adjust Cross-Encoder Threshold**: Tune `ce_threshold` for your use case
- **Optimize Window Size**: Modify `window_size` in `make_windows.py`
- **Memory Optimization**: Use quantized models for smaller devices

## Future Enhancements

### Planned Improvements

- **Agentic Architecture**: Parallel processing with specialized agents
- **Async Processing**: Non-blocking operations for better performance
- **Advanced Validation**: ML-based quality assessment
- **Multi-Modal Support**: Image and text processing capabilities
- **Real-time Collaboration**: Multi-user support for medical teams

### Research Directions

- **Federated Learning**: Privacy-preserving model updates
- **Edge Deployment**: Optimized for mobile and embedded devices
- **Medical Validation**: Integration with medical knowledge bases
- **Performance Benchmarking**: Comprehensive evaluation metrics

## Acknowledgments

- **Tactical Combat Casualty Care (TCCC)** guidelines for medical protocol structure
- **Mistral AI** for the open-source language model
- **Hugging Face** for the sentence-transformers and cross-encoder models
- **Streamlit** for the web interface framework
- **GitHub** for hosting and version control

## Contributing

We welcome contributions! Please see our [GitHub repository](https://github.com/rahul370139/marchpaws_rag) for:
- Issue reporting
- Feature requests
- Pull requests
- Documentation improvements

---

**Repository**: [https://github.com/rahul370139/marchpaws_rag](https://github.com/rahul370139/marchpaws_rag)
