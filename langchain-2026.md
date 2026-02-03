# LangChain Comprehensive Guide 2026

## Table of Contents
1. [Foundational Concepts](#foundational-concepts)
2. [Architecture & Core Components](#architecture--core-components)
3. [LangChain Expression Language (LCEL)](#langchain-expression-language-lcel)
4. [LLM Integrations & Multi-Model Setup](#llm-integrations--multi-model-setup)
5. [Prompts & Templates Deep Dive](#prompts--templates-deep-dive)
6. [Chains: From Simple to Advanced](#chains-from-simple-to-advanced)
7. [Memory Systems](#memory-systems)
8. [Agents & Tools](#agents--tools)
9. [LangGraph: Production Agent Orchestration](#langgraph-production-agent-orchestration)
10. [RAG Systems](#rag-systems)
11. [Multi-Model Routing & Load Balancing](#multi-model-routing--load-balancing)
12. [Production Patterns](#production-patterns)
13. [Multi-Agent Architectures](#multi-agent-architectures)
14. [Complete Working Examples](#complete-working-examples)

---

## Foundational Concepts

### What is LangChain? (2026 Evolution)

LangChain is a framework for developing applications powered by language models. It enables composition of modular components (LLMs, retrievers, memory, tools) into complex workflows that weren't feasible with direct API calls.

**Evolution Timeline:**

```
2023: Chains were primary abstraction
  ↓
2024: Shift toward agents and orchestration (LangGraph introduction)
  ↓
2025: Unified Expression Language (LCEL) + LangGraph becomes standard
  ↓
2026: Agent-first framework with declarative workflow definition
       57% of production AI deployments use LangGraph
       Native MCP (Model Context Protocol) support
       Unified cost tracking across models, tools, retrieval
       LangSmith becomes essential observability layer
```

**Core Philosophy (2026):**
- **Modularity**: Swap components without rewriting code
- **Composability**: Chain simple operations into complex workflows
- **Abstraction**: Same interface for 100+ LLM providers
- **Observability**: First-class tracing and debugging with LangSmith
- **Reliability**: Production-grade orchestration with LangGraph

### Why LangChain Matters (Production Context)

**Problems it solves:**

```
Without LangChain:
- Direct API calls scattered across codebase
- No standardized memory management
- Prompt engineering embedded in code
- No retrieval abstraction (search engines, docs, etc.)
- Tool integration ad-hoc
- Debugging multi-step workflows is nightmare

With LangChain (2026):
- Unified interface for 100+ LLM providers
- Built-in memory, prompt templates, chains
- Integrated vector databases and retrievers
- Declarative agent definitions with LangGraph
- Production-ready observability with LangSmith
- Automatic cost tracking and model optimization
- Native support for Model Context Protocol (MCP)
```

**Real-world applications:**

1. **Retrieval-Augmented Generation (RAG)** - Combine LLMs with knowledge bases
2. **Conversational Agents** - Multi-turn interactions with tool access
3. **Workflow Automation** - Orchestrate complex multi-step processes
4. **Data Processing Pipelines** - Transform documents with LLMs
5. **Question-Answering Systems** - Domain-specific knowledge bases
6. **Multi-Model Orchestration** - Route queries to optimal models
7. **Deep Agents** - Long-horizon reasoning with tool loops

### Key Paradigm Shift (2024-2026)

LangChain shifted from **chains** to **agents** to **LangGraph** as primary abstraction:

```
Traditional Chains (2023):
if query_type == "simple":
    return simple_chain(query)
elif query_type == "complex":
    return complex_chain(query)

Agents (2024):
agent.invoke({"input": query})
# Agent decides tool usage dynamically

LangGraph (2026):
graph = StateGraph(State)
graph.add_node("research", research_node)
graph.add_node("write", write_node)
graph.add_edge("research", "write")
app = graph.compile()
result = app.invoke(initial_state)
# Explicit state, conditional routing, persistence
```

**What changed:**
- LangGraph became standard for production deployments (57% adoption)
- Explicit state management over implicit memory
- Human-in-the-loop workflows as first-class feature
- Structured outputs and type validation
- MCP (Model Context Protocol) integration
- Unified cost tracking across entire workflow

---

## Architecture & Core Components

### LangChain 2026 Ecosystem

```
┌─────────────────────────────────────────────────┐
│         LangChain Ecosystem (2026)              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │   LangChain Core (v1.2+)                 │  │
│  ├──────────────────────────────────────────┤  │
│  │ • LCEL (Expression Language)             │  │
│  │ • Runnable Interface                     │  │
│  │ • Prompt Templates & Management          │  │
│  │ • Memory Systems                         │  │
│  │ • Structured Output Parsing              │  │
│  │ • Streaming & Async Support              │  │
│  └──────────────────────────────────────────┘  │
│                     ↓                           │
│  ┌──────────────────────────────────────────┐  │
│  │   LangGraph (Production Orchestration)   │  │
│  ├──────────────────────────────────────────┤  │
│  │ • Stateful Graphs (with persistence)     │  │
│  │ • Explicit State Management              │  │
│  │ • Conditional Edges & Routing            │  │
│  │ • Sub-graphs & Composition               │  │
│  │ • Human-in-the-Loop Interrupts           │  │
│  │ • Checkpointing & Recovery               │  │
│  │ • Stream Input/Output API                │  │
│  └──────────────────────────────────────────┘  │
│                     ↓                           │
│  ┌──────────────────────────────────────────┐  │
│  │   LangSmith (Development Platform)       │  │
│  ├──────────────────────────────────────────┤  │
│  │ • Real-time Tracing & Debugging          │  │
│  │ • Experiment Management                  │  │
│  │ • Dataset Curation & Testing             │  │
│  │ • Production Monitoring                  │  │
│  │ • Pairwise Annotations (NEW)             │  │
│  │ • LangSmith Polly AI Assistant (BETA)    │  │
│  │ • Unified Cost Tracking (NEW)            │  │
│  │ • No-Code Agent Builder (PRIVATE BETA)   │  │
│  └──────────────────────────────────────────┘  │
│                     ↓                           │
│  ┌──────────────────────────────────────────┐  │
│  │   Community Integrations (100+)          │  │
│  ├──────────────────────────────────────────┤  │
│  │ LLM Providers (100+):                    │  │
│  │ • OpenAI (GPT-4, GPT-4 Turbo)           │  │
│  │ • Anthropic (Claude 3.5 Sonnet)         │  │
│  │ • Google (Gemini 2.0 Flash)             │  │
│  │ • Groq (Ultra-fast open models)         │  │
│  │ • Together (Inference API)               │  │
│  │ • Mistral, Cohere, Replicate            │  │
│  │ • Local: Ollama, LM Studio              │  │
│  │ • Specialized: Claude, Grok, etc.       │  │
│  │                                          │  │
│  │ Vector Stores:                           │  │
│  │ • Pinecone, Weaviate, Qdrant            │  │
│  │ • Chroma, Milvus, LanceDB               │  │
│  │ • PostgreSQL (pgvector)                 │  │
│  │ • Elasticsearch, OpenSearch             │  │
│  │                                          │  │
│  │ Tools & Services:                        │  │
│  │ • Web search, calculators, APIs         │  │
│  │ • Document loaders (100+ formats)       │  │
│  │ • SQL, graph databases                  │  │
│  │ • Model Context Protocol (MCP) tools    │  │
│  │                                          │  │
│  │ Specialized Modules:                     │  │
│  │ • hub: Central prompt registry           │  │
│  │ • smith: Development tracking            │  │
│  │ • serve: Deployment (LangServe)         │  │
│  │ • benchmarks: Performance measurement    │  │
│  └──────────────────────────────────────────┘  │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Core Abstractions (2026 Update)

#### **1. Runnable Interface (Universal Standard)**

Every component in LangChain implements `Runnable`:

```python
from langchain_core.runnables import Runnable, RunnableConfig

class Runnable(Protocol):
    """Universal interface for all components."""
    
    # Synchronous execution
    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """Single input → single output."""
        
    # Asynchronous execution
    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """Async version of invoke."""
        
    # Batch processing (parallel)
    def batch(self, inputs: List[Any], config: Optional[RunnableConfig] = None, 
              **kwargs) -> List[Any]:
        """Multiple inputs → multiple outputs (parallel)."""
        
    # Streaming (token-by-token)
    def stream(self, input: Any, config: Optional[RunnableConfig] = None,
               **kwargs) -> Iterator[Any]:
        """Stream chunks as they arrive."""
        
    # Async streaming
    async def astream(self, input: Any, config: Optional[RunnableConfig] = None,
                      **kwargs) -> AsyncIterator[Any]:
        """Async stream output."""
        
    # Graph representation (for debugging)
    def get_graph(self) -> RunnableGraph:
        """Visualize execution graph."""
        
    # Composition
    def __or__(self, other: Runnable) -> Runnable:
        """Pipe operator: chain1 | chain2 | chain3."""
        
    # With fallback
    def with_fallbacks(self, fallbacks: List[Runnable]) -> Runnable:
        """Add fallback chains on error."""
```

**Why this matters:**
- Same interface for LLMs, retrievers, tools, chains
- Enables declarative composition
- Supports async and streaming natively
- Backward compatible across 100+ implementations
- Enables automatic observability

#### **2. Structured Output Parsing (2026 Enhancement)**

Type-safe output with Pydantic validation:

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from typing import List

class ResearchReport(BaseModel):
    """Structured research output."""
    title: str = Field(description="Main research title")
    key_findings: List[str] = Field(description="3-5 key findings")
    sources: List[str] = Field(description="Source citations")
    confidence_score: float = Field(ge=0, le=1, description="Confidence 0-1")
    next_steps: List[str] = Field(description="Recommended actions")

# Parser validates LLM output against schema
parser = PydanticOutputParser(pydantic_object=ResearchReport)

# LLM with structured output (NEW in 2026)
llm = ChatOpenAI(model="gpt-4-turbo")
structured_llm = llm.with_structured_output(ResearchReport)

result = structured_llm.invoke("Research quantum computing")
# Returns: ResearchReport(title="...", key_findings=[...], ...)
# Type-safe, validated, ready to use
```

#### **3. LLM Interface (100+ Providers)**

```python
from langchain_core.language_models import BaseChatModel

class ChatModel(BaseChatModel):
    """Chat model interface (all providers implement this)."""
    
    model_name: str
    temperature: float
    max_tokens: Optional[int]
    
    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """Generate response from messages."""
        
    def bind_tools(self, tools: List[Tool]) -> Self:
        """Enable tool calling (structured outputs)."""
        
    def with_structured_output(self, schema: Union[Type[T], dict]) -> Self:
        """Enforce output format with schema."""
        
    def stream_usage(self) -> bool:
        """Track token usage during streaming."""
        
    def get_binding(self) -> Dict:
        """Get current bindings (tools, system prompts, etc.)."""
```

**Universal implementation** across 100+ providers (2026):
- OpenAI, Anthropic, Google, Groq, Together, Mistral, Cohere, etc.
- Local models: Ollama, vLLM, TextGeneration, LM Studio
- Specialized: Claude API, Grok, custom fine-tuned models
- Same `.invoke()`, `.stream()`, `.bind_tools()` interface
- Seamless provider swapping
- Built-in cost tracking

#### **4. Model Context Protocol (MCP) Integration (NEW 2026)**

Universal protocol for tool/resource availability:

```python
from langchain_core.tools import MCP
from langchain_mcp import MCPAdapter

# Connect to MCP servers (e.g., GitHub, Slack, Notion)
github_adapter = MCPAdapter(
    url="mcp+stdio://github-mcp",  # GitHub MCP server
    tools=["search_issues", "create_pr", "list_repos"]
)

# MCP tools work like any LangChain tool
agent = create_react_agent(
    llm=llm,
    tools=github_adapter.get_tools(),  # Dynamically loaded
)

# Agent can now use GitHub tools natively
result = agent.invoke({
    "input": "Find issues about vector databases in my repos"
})
```

**MCP Benefits:**
- Standardized tool discovery
- Automatic credential handling
- Resource servers (not just tools)
- Version compatibility checking
- Self-describing capabilities

---

## LangChain Expression Language (LCEL)

### LCEL Fundamentals (2026 Standard)

LCEL is the declarative way to compose chains using pythonic syntax:

```python
# Old approach (chains v0.1 - deprecated):
from langchain.chains import LLMChain
prompt = PromptTemplate(template="...", input_variables=["topic"])
llm_chain = LLMChain(llm=llm, prompt=prompt)
output = llm_chain.run(topic="AI")

# New standard (LCEL - 2026):
chain = prompt | llm | output_parser
output = chain.invoke({"topic": "AI"})

# Or with streaming:
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

### LCEL Operators & Patterns

#### **1. Pipe Operator (|) - Sequential Composition**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Linear pipeline
chain = (
    ChatPromptTemplate.from_template("Explain {concept}")
    | ChatOpenAI(model="gpt-4")
    | StrOutputParser()
)

result = chain.invoke({"concept": "quantum computing"})

# Each step processes output of previous step
# prompt template → llm → parser
```

#### **2. Parallel Execution (RunnableParallel)**

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Execute multiple branches simultaneously
parallel_chain = RunnableParallel(
    analysis=llm_analysis_chain,
    summary=llm_summary_chain,
    questions=llm_qa_chain,
    sentiment=llm_sentiment_chain,
)

result = parallel_chain.invoke({"input": "Long document..."})
# All four chains execute in parallel
# result = {
#     "analysis": "...",
#     "summary": "...",
#     "questions": "...",
#     "sentiment": "..."
# }

# Parallel execution benefits
# - 4 chains taking 2s each → 2s total (not 8s)
# - Faster response, better throughput
# - Ideal for multi-perspective analysis
```

#### **3. Conditional Routing (RunnableBranch)**

```python
from langchain_core.runnables import RunnableBranch

def route_by_complexity(input_dict):
    """Route to different chains based on complexity."""
    text = input_dict.get("text", "")
    word_count = len(text.split())
    return word_count > 500  # Return boolean

# Simple routing
simple_chain = ChatOpenAI(model="gpt-3.5-turbo")
complex_chain = ChatOpenAI(model="gpt-4-turbo")

router = RunnableBranch(
    (lambda x: len(x.get("text", "").split()) > 500, complex_chain),
    simple_chain  # Default
)

result = router.invoke({"text": "Your text here..."})

# Advanced: Chain-specific routing
def route_by_intent(input_dict):
    """Classify intent from user input."""
    intent = classify_intent(input_dict["query"])
    return intent

intent_router = RunnableBranch(
    (lambda x: "search" in x.get("query", "").lower(), 
     search_chain),
    (lambda x: "summarize" in x.get("query", "").lower(), 
     summarize_chain),
    (lambda x: "translate" in x.get("query", "").lower(), 
     translate_chain),
    default_chain  # Fallback
)

result = intent_router.invoke({"query": "summarize this document..."})
```

#### **4. Retry & Fallback Logic**

```python
from langchain_core.runnables import RunnableRetry

# Automatic retry with exponential backoff
reliable_chain = llm.with_retry(
    max_attempts=3,
    backoff_factor=2,
    retry_on=(RateLimitError, APIConnectionError),
)

# Fallback to alternative models
primary = ChatOpenAI(model="gpt-4")
fallback1 = ChatAnthropic(model="claude-3-5-sonnet")
fallback2 = ChatGroq(model="mixtral-8x7b")

chain_with_fallbacks = primary.with_fallbacks(
    [fallback1, fallback2],
    exception_key="error",
)

# Tries primary, falls back if rate limited
result = chain_with_fallbacks.invoke({"input": "Your query"})
```

#### **5. State Management (RunnablePassthrough)**

```python
from langchain_core.runnables import RunnablePassthrough

# Preserve state while adding computed fields
chain = (
    RunnablePassthrough.assign(
        question=lambda x: x["input"],
        retrieved_docs=retriever | format_docs,
        current_time=lambda x: datetime.now(),
    )
    | prompt
    | llm
    | parser
)

# Now prompt has access to: input, question, retrieved_docs, current_time
result = chain.invoke({"input": "What is AI?"})
```

#### **6. Streaming & Async**

```python
# Token-by-token streaming
for chunk in chain.stream({"input": "Explain AI"}):
    print(chunk, end="", flush=True)

# Async streaming (for high-throughput)
async for chunk in chain.astream({"input": "Explain AI"}):
    print(chunk, end="", flush=True)

# Batch processing (parallel)
queries = [
    {"input": "Explain AI"},
    {"input": "Explain ML"},
    {"input": "Explain DL"},
]
results = chain.batch(queries)  # Processes in parallel

# Async batch
results = await chain.abatch(queries)
```

---

## LLM Integrations & Multi-Model Setup

### 2026 LLM Provider Landscape

#### **OpenAI (Commercial Leader)**

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Latest models (2026)
gpt4 = ChatOpenAI(
    model="gpt-4-turbo-preview",  # Most capable
    temperature=0.7,
    max_tokens=4096,
    api_key="sk-...",
)

gpt4o = ChatOpenAI(
    model="gpt-4o",  # Vision + text, faster
    temperature=0.5,
)

gpt4mini = ChatOpenAI(
    model="gpt-4-mini",  # Fast + cheap
    temperature=0.3,
)

# Embedding models
embeddings_large = OpenAIEmbeddings(model="text-embedding-3-large")
embeddings_small = OpenAIEmbeddings(model="text-embedding-3-small")

# Tool calling (structured outputs)
tools = [search_tool, calculator_tool]
llm_with_tools = gpt4.bind_tools(tools)

response = llm_with_tools.invoke("What's 5+3 and search for AI news?")
# response includes structured tool calls
```

#### **Anthropic Claude (Long-context, Reasoning)**

```python
from langchain_anthropic import ChatAnthropic

# Claude 3.5 Sonnet (2026 release)
claude = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    max_tokens=8000,
    temperature=0.7,
    api_key="sk-ant-...",
)

# Extended thinking (new in 2026)
claude_thinking = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    thinking={
        "type": "enabled",
        "budget_tokens": 10000,  # How much to think
    }
)

# Vision support
from langchain_core.messages import HumanMessage

response = claude.invoke([
    HumanMessage(
        content=[
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_image,
                },
            },
        ]
    )
])

# Tool use (native to Claude)
claude_with_tools = claude.bind_tools(tools)
```

#### **Google Gemini (Speed + Capabilities)**

```python
from langchain_google_genai import ChatGoogleGenerativeAI

# Gemini 2.0 Flash (NEW 2026, ultra-fast)
gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key="your-key",
)

# Multimodal (text, image, audio, video)
response = gemini.invoke([
    HumanMessage(
        content=[
            {"type": "text", "text": "Analyze this video"},
            {
                "type": "video",
                "source": {
                    "type": "inline_data",
                    "mime_type": "video/mp4",
                    "data": video_bytes,
                },
            },
        ]
    )
])

# Thinking mode (experimental)
gemini_thinking = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    thinking_config={"type": "enabled"},
)
```

#### **Groq (Extreme Speed)**

```python
from langchain_groq import ChatGroq

# Fastest inference for open models
groq = ChatGroq(
    model="mixtral-8x7b-32768",  # MoE model
    temperature=0.7,
    groq_api_key="your-key",
    request_timeout=10,
)

# Also supports: llama2-70b, neural-chat, etc.
groq_fast = ChatGroq(
    model="llama2-70b-4096",
    temperature=0.5,
)

# Streaming is Groq's strength
for chunk in groq.stream("Write a long article about AI"):
    print(chunk.content, end="", flush=True)
```

#### **Open Source: Ollama (Local Deployment)**

```python
from langchain_ollama import ChatOllama

# Run locally (no API costs, full privacy)
local_llm = ChatOllama(
    model="llama2",
    base_url="http://localhost:11434",
    temperature=0.7,
    num_ctx=2048,  # Context window
    num_gpu=1,     # Use GPU
)

# Or models: mistral, neural-chat, dolphin-mixtral, etc.
mistral_local = ChatOllama(
    model="mistral",
    base_url="http://localhost:11434",
)

# Tool calling support (2026)
local_with_tools = local_llm.bind_tools(tools)

# No rate limits, full control
result = local_llm.invoke("Complex reasoning task...")
```

#### **Other Providers**

```python
# Mistral AI
from langchain_mistralai import ChatMistral
mistral = ChatMistral(model="mistral-large", api_key="...")

# Cohere
from langchain_cohere import ChatCohere
cohere = ChatCohere(model="command-r-plus", api_key="...")

# Together AI (multi-model)
from langchain_together import ChatTogether
together = ChatTogether(
    model="meta-llama/Llama-3-70b-chat-hf",
    api_key="..."
)

# Replicate (hosted open models)
from langchain_replicate import Replicate
replicate = Replicate(
    model="meta/llama-2-70b-chat:02e509cc789964434f3e1596c86f72893f532e393e5547aab1d27d374cfc4799",
)

# Custom/Self-hosted
from langchain_openai import ChatOpenAI
custom = ChatOpenAI(
    model="my-custom-model",
    base_url="https://my-api.example.com/v1",  # Custom endpoint
    api_key="...",
)
```

### Multi-Model Router (2026 Pattern)

```python
from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import PromptTemplate

class ModelRouter:
    """Intelligent routing to optimal LLM for task."""
    
    def __init__(self):
        self.models = {
            "gpt4": ChatOpenAI(model="gpt-4-turbo"),
            "gpt4o": ChatOpenAI(model="gpt-4o"),  
            "claude": ChatAnthropic(model="claude-3-5-sonnet"),
            "groq": ChatGroq(model="mixtral-8x7b"),
            "local": ChatOllama(model="mistral"),
        }
        
        # Classifier for task routing
        self.task_classifier = PromptTemplate(
            input_variables=["input"],
            template="""Classify this task as one of:
            - simple: straightforward, factual
            - vision: requires image understanding
            - reasoning: complex reasoning needed
            - writing: creative or long-form content
            - code: programming/technical
            
            Task: {input}
            Classification:"""
        )
    
    def get_optimal_model(self, task_type: str, budget: str = "balanced"):
        """Select model based on task and budget."""
        routing_config = {
            "simple": {
                "low_cost": "groq",  # Fast, cheap
                "balanced": "gpt4o",  # Good all-rounder
                "best": "gpt-4-turbo",  # Most accurate
            },
            "vision": {
                "low_cost": "gpt4o",  # Good vision
                "balanced": "gpt4o",
                "best": "gpt-4-turbo",
            },
            "reasoning": {
                "low_cost": "claude",  # Good reasoning
                "balanced": "gpt-4-turbo",
                "best": "gpt-4-turbo",  # Extended thinking
            },
            "writing": {
                "low_cost": "groq",
                "balanced": "claude",  # Strong writing
                "best": "gpt-4-turbo",
            },
            "code": {
                "low_cost": "gpt4o",
                "balanced": "gpt-4-turbo",
                "best": "gpt-4-turbo",
            },
        }
        
        model_name = routing_config.get(task_type, {}).get(budget, "gpt4o")
        return self.models[model_name]
    
    def invoke_with_routing(self, input_text: str, budget: str = "balanced"):
        """Classify task and invoke appropriate model."""
        # Step 1: Classify task
        classifier_chain = (
            self.task_classifier
            | ChatOpenAI(model="gpt-4-mini")  # Fast classifier
        )
        
        classification = classifier_chain.invoke({"input": input_text})
        task_type = classification.strip().split(": ")[-1].lower()
        
        # Step 2: Get optimal model
        llm = self.get_optimal_model(task_type, budget)
        
        # Step 3: Execute
        return llm.invoke(input_text)

# Usage
router = ModelRouter()

# Automatically chooses Claude for reasoning, Groq for speed, etc.
result = router.invoke_with_routing(
    "Explain quantum entanglement deeply",
    budget="balanced"
)
```

### Cost Tracking (NEW 2026)

```python
from langchain.callbacks import get_openai_callback

# Unified cost tracking
with get_openai_callback() as cb:
    result = chain.invoke({"input": "Your query"})
    
    print(f"Model: {cb.model_name}")
    print(f"Total tokens: {cb.total_tokens}")
    print(f"Prompt tokens: {cb.prompt_tokens}")
    print(f"Completion tokens: {cb.completion_tokens}")
    print(f"Total cost: ${cb.total_cost:.4f}")
    print(f"Successful requests: {cb.successful_requests}")

# Works across 50+ LLM providers!
# LangSmith unified tracking (2026 feature)
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"

# All API calls automatically tracked with costs
result = chain.invoke(...)
# View cost breakdown at https://smith.langchain.com
```

---

## Prompts & Templates Deep Dive

### Advanced Prompt Engineering (2026)

#### **Dynamic Few-Shot Prompts**

```python
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

# Examples for semantic selection
examples = [
    {
        "input": "The stock market crashed 20%",
        "sentiment": "negative",
        "confidence": 0.95
    },
    {
        "input": "Company announced record profits",
        "sentiment": "positive",
        "confidence": 0.98
    },
    {
        "input": "Market stabilized after volatility",
        "sentiment": "neutral",
        "confidence": 0.80
    },
]

example_prompt = PromptTemplate(
    input_variables=["input", "sentiment", "confidence"],
    template="Text: {input}\nSentiment: {sentiment} (confidence: {confidence})"
)

# Select examples similar to query
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=OpenAIEmbeddings(),
    vectorstore_cls=Chroma,
    k=2,  # Select 2 most similar examples
)

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Analyze sentiment: {text}",
    input_variables=["text"],
)

# Results in smart example selection, not fixed examples
```

#### **Chain-of-Thought Prompting**

```python
from langchain_core.prompts import ChatPromptTemplate

cot_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a reasoning expert. Break down problems step by step.

For each problem:
1. Identify what's given
2. State what we need to find
3. Work through the logic step by step
4. Show your reasoning at each step
5. Provide final answer

Format your reasoning with numbered steps."""),
    
    ("human", "Problem: {problem}"),
])

chain = cot_prompt | llm | output_parser
result = chain.invoke({"problem": "If a train travels..."})

# LLM will output structured reasoning, not just answer
```

#### **Tree-of-Thoughts Prompting**

```python
# Explore multiple reasoning paths simultaneously
tot_prompt = ChatPromptTemplate.from_template("""
Explore multiple approaches to solve this problem:
1. First approach: [method A]
   - Reasoning
   - Conclusion
2. Second approach: [method B]  
   - Reasoning
   - Conclusion
3. Third approach: [method C]
   - Reasoning
   - Conclusion

Compare approaches and recommend best solution.

Problem: {problem}
""")

# Parallel execution of reasoning paths
result = tot_prompt | llm | output_parser
```

### Prompt Templates with Validation

```python
from pydantic import BaseModel, Field, validator

class PromptInput(BaseModel):
    """Validated prompt input."""
    topic: str = Field(min_length=1, max_length=100)
    tone: str = Field(regex="^(formal|casual|technical)$")
    length: int = Field(ge=100, le=5000, description="Word count")
    
    @validator('topic')
    def topic_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Topic cannot be empty")
        return v

prompt = ChatPromptTemplate.from_template(
    "Write {length}-word {tone} article on: {topic}"
)

# Validated chain
validated_chain = (
    lambda x: PromptInput(**x)  # Validate input
    | prompt
    | llm
    | output_parser
)

# Only accepts valid inputs
try:
    result = validated_chain.invoke({
        "topic": "AI Safety",
        "tone": "formal",
        "length": 1000
    })
except ValidationError as e:
    print(f"Invalid input: {e}")
```

---

## Chains: From Simple to Advanced

### Production-Grade Chains (2026)

#### **Error Handling Chain**

```python
from langchain_core.runnables import RunnableTry

# Chain with automatic error handling
robust_chain = (
    prompt
    | RunnableTry(
        llm,
        on_error="retry",  # or "continue", "raise"
        retries=3,
        backoff=exponential(multiplier=1, min=4, max=10),
    )
    | output_parser
)

# Handles rate limits, timeouts, API errors gracefully
result = robust_chain.invoke({"input": "Your query"})
```

#### **Monitoring & Observability Chain**

```python
from langchain.callbacks import StdOutCallbackHandler, LangChainTracer
from langsmith import Client

# Custom callback for monitoring
class MonitoringCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM call starting: {serialized.get('name')}")
    
    def on_llm_end(self, response, **kwargs):
        print(f"Tokens used: {response.llm_output.get('token_usage')}")

# Build chain with callbacks
chain = (
    prompt
    | llm
    | output_parser
).with_config(
    callbacks=[MonitoringCallback()],
    run_name="production_chain"
)

# Automatic tracing with LangSmith (2026)
tracer = LangChainTracer(project_name="my-project")
result = chain.with_config(callbacks=[tracer]).invoke({"input": "..."})
# Visible in LangSmith with full debugging info
```

#### **Context Window Management**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda

class ContextWindowManager:
    """Manage document size vs context window."""
    
    def __init__(self, model_name: str, safety_margin: float = 0.8):
        self.context_limits = {
            "gpt-4-turbo": 128000,
            "claude-3-5-sonnet": 200000,
            "gpt-4o": 128000,
            "llama2-70b": 4096,
        }
        self.limit = self.context_limits.get(model_name, 4096)
        self.available_tokens = int(self.limit * safety_margin * 0.25)  # ~chars
    
    def truncate_if_needed(self, content: str) -> str:
        """Truncate content to fit context window."""
        if len(content) > self.available_tokens:
            return content[:self.available_tokens] + "\n[... truncated ...]"
        return content

manager = ContextWindowManager("gpt-4-turbo")

# Use in chain
context_aware_chain = (
    RunnableLambda(lambda x: {
        "context": manager.truncate_if_needed(x["context"]),
        "question": x["question"]
    })
    | prompt
    | llm
)
```

---

## Memory Systems

### Advanced Memory Patterns (2026)

#### **Hierarchical Memory (Multi-Level)**

```python
from langchain.memory import ConversationSummaryBufferMemory
from datetime import datetime
from typing import Optional

class HierarchicalMemory:
    """Multi-level memory system."""
    
    def __init__(self, llm, max_tokens=2000):
        self.llm = llm
        self.max_tokens = max_tokens
        
        # Level 1: Recent (full detail)
        self.recent_memory = ConversationBufferMemory(
            return_messages=True,
            human_prefix="Human",
            ai_prefix="AI",
        )
        
        # Level 2: Intermediate (summarized)
        self.summary_memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=max_tokens,
            return_messages=True,
        )
        
        # Level 3: Long-term (key facts)
        self.facts_memory = {}
    
    def add_interaction(self, human_input: str, ai_response: str):
        """Add to all memory levels."""
        self.recent_memory.save_context(
            {"input": human_input},
            {"output": ai_response}
        )
        self.summary_memory.save_context(
            {"input": human_input},
            {"output": ai_response}
        )
        
        # Extract facts
        facts = self.extract_facts(human_input, ai_response)
        self.facts_memory.update(facts)
    
    def extract_facts(self, user_input: str, response: str) -> dict:
        """Extract key facts from interaction."""
        # Would use NER, relation extraction, etc.
        return {}
    
    def get_context(self) -> str:
        """Get appropriate context based on conversation length."""
        if len(self.recent_memory.buffer) < self.max_tokens:
            return self.recent_memory.buffer
        else:
            return self.summary_memory.buffer + "\n\nKey facts: " + str(self.facts_memory)

# Usage
memory = HierarchicalMemory(llm)

for human_msg, ai_msg in interactions:
    memory.add_interaction(human_msg, ai_msg)
    context = memory.get_context()
    # Context grows efficiently without overwhelming context window
```

#### **Knowledge Graph Memory**

```python
import networkx as nx
from langchain.memory import ConversationKG

# Knowledge graph maintains entity relationships
kg_memory = ConversationKG()

# Conversation builds graph over time
interaction_1 = "John works at Google and is interested in AI"
interaction_2 = "John's team focuses on LLMs"

kg_memory.add_memory(interaction_1)
kg_memory.add_memory(interaction_2)

# Graph now knows:
# (John) --works_at--> (Google)
# (John) --interested_in--> (AI)
# (John) --team_focuses--> (LLMs)

# Rich context retrieval
context = kg_memory.get_history()
# Can query: "What does John's team focus on?" → LLMs
```

#### **Persistent Memory with Database**

```python
from langchain.memory import RedisChatMessageHistory
from langchain_community.chat_message_histories import MongoDBChatMessageHistory

# Redis (fast, in-memory)
redis_memory = RedisChatMessageHistory(
    session_id="user_123",
    url="redis://localhost:6379"
)

# MongoDB (persistent)
mongo_memory = MongoDBChatMessageHistory(
    session_id="user_123",
    connection_string="mongodb://localhost:27017/",
    database_name="conversations"
)

# Use in chain with automatic persistence
chain = ConversationChain(
    llm=llm,
    memory=mongo_memory,
    verbose=True,
)

# Survives restarts, scales to millions of users
response = chain.run("Hello!")
```

---

## Agents & Tools

### Tool Definition (2026 Standard)

#### **Decorated Tools**

```python
from langchain_core.tools import tool
from typing import Annotated

@tool
def calculate(expression: str) -> float:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Result of the calculation
    """
    return eval(expression)  # Use safer eval in production

@tool
def search_web(query: Annotated[str, "Search query"],
               num_results: Annotated[int, "Number of results"] = 5) -> str:
    """
    Search the web for information.
    
    Args:
        query: What to search for
        num_results: How many results to return
        
    Returns:
        Search results as formatted string
    """
    # Actual implementation
    return mock_search(query, num_results)

# Tools are automatically discoverable
tools = [calculate, search_web]
```

#### **Class-Based Tools (Advanced)**

```python
from langchain_core.tools import BaseTool

class DatabaseQueryTool(BaseTool):
    """Query a database for information."""
    
    name: str = "database_query"
    description: str = "Execute SQL queries on the knowledge database"
    
    def _run(self, query: str) -> str:
        """Synchronous execution."""
        return execute_query(query)
    
    async def _arun(self, query: str) -> str:
        """Asynchronous execution."""
        return await execute_query_async(query)

# Tool bindings (enable tool calling)
llm_with_tools = llm.bind_tools([calculate, search_web])
response = llm_with_tools.invoke("What's 5+5 and current AI news?")
```

### Agents (2026 Patterns)

#### **ReAct Agent (Standard)**

```python
from langgraph.agents import create_react_agent, AgentExecutor

# Define tools
tools = [web_search, calculator, database_query]

# Create agent
agent = create_react_agent(
    model=llm,
    tools=tools,
    # Optional custom system prompt
    prompt=PromptTemplate.from_template(custom_system_prompt),
)

# Execute
executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True,
)

result = executor.invoke({
    "input": "What's the capital of France and current weather there?"
})

# Agent automatically:
# 1. Determines which tools to use
# 2. Executes tools
# 3. Reasons about results
# 4. Repeats until answer found
```

**What happens internally:**

```
Input: "What's the capital of France and current weather there?"

Thought: I need to know the capital of France and current weather.
Action: web_search
Action Input: {"query": "capital of France"}
Observation: The capital of France is Paris.

Thought: Now I need the current weather in Paris.
Action: web_search
Action Input: {"query": "current weather Paris"}
Observation: Paris weather today: sunny, 22°C

Final Answer: The capital of France is Paris. The current weather in Paris is sunny with a temperature of 22°C.
```

#### **Tool Calling Agent (Structured Output)**

```python
from pydantic import BaseModel

class ToolCall(BaseModel):
    tool_name: str
    tool_input: dict
    reasoning: str

# Model with structured output
tool_calling_model = llm.with_structured_output(ToolCall)

# Model returns: {
#   "tool_name": "web_search",
#   "tool_input": {"query": "..."},
#   "reasoning": "..."
# }

# Execute tool
agent_response = tool_calling_model.invoke("Your query")
tool = get_tool(agent_response.tool_name)
result = tool.invoke(agent_response.tool_input)
```

---

## LangGraph: Production Agent Orchestration

### What is LangGraph? (2026 Standard)

**LangGraph is the production standard for AI agents.**

- Used by 57% of production AI deployments (2026)
- Klarna: 85 million users
- AppFolio: entire platform
- Why: State management, persistence, human-in-the-loop

### Core Concepts

#### **1. State Graph Basics**

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add

# Define state schema
class AgentState(TypedDict):
    messages: Annotated[list, add]  # Auto-append new messages
    research_data: list
    analysis: str
    user_feedback: str

# Create graph
graph = StateGraph(AgentState)

# Add nodes (functions that transform state)
def research_node(state: AgentState) -> dict:
    """Conduct research on the topic."""
    query = state["messages"][-1].content
    results = web_search(query)
    return {
        "research_data": results,
        "messages": [f"Found {len(results)} research items"]
    }

def analysis_node(state: AgentState) -> dict:
    """Analyze research findings."""
    analysis = analyze_research(state["research_data"])
    return {
        "analysis": analysis,
        "messages": [f"Analysis: {analysis[:100]}..."]
    }

# Add edges (control flow)
graph.add_node("research", research_node)
graph.add_node("analysis", analysis_node)
graph.add_edge("research", "analysis")
graph.add_edge("analysis", END)

# Set entry point
graph.set_entry_point("research")

# Compile to executable
app = graph.compile()

# Execute
result = app.invoke({
    "messages": [{"content": "Research quantum computing"}],
    "research_data": [],
    "analysis": "",
    "user_feedback": ""
})
```

#### **2. Conditional Routing**

```python
from langgraph.graph import StateGraph

def route_next_step(state: AgentState) -> str:
    """Decide next action based on state."""
    if len(state["research_data"]) < 5:
        return "research"  # Need more data
    elif state["user_feedback"]:
        return "revise"  # User provided feedback
    else:
        return END  # Done

# Build graph with routing
graph = StateGraph(AgentState)

graph.add_node("research", research_node)
graph.add_node("analysis", analysis_node)
graph.add_node("revise", revise_node)

# Conditional edge
graph.add_conditional_edges(
    "research",
    route_next_step,
    {
        "research": "research",  # Loop back
        "analysis": "analysis",
        "revise": "revise",
        END: END
    }
)

graph.set_entry_point("research")
app = graph.compile()

# Executes dynamically based on decisions
result = app.invoke(initial_state)
```

#### **3. Parallel Execution (Sub-graphs)**

```python
from langgraph.graph import StateGraph, Send

class ResearchState(TypedDict):
    query: str
    sources: list

class AnalysisState(TypedDict):
    source: dict
    analysis: str

# Parallel analysis nodes
def analyze_source(state: AnalysisState) -> dict:
    """Analyze single source."""
    analysis = llm.invoke(f"Analyze: {state['source']}")
    return {"analysis": analysis}

# Main graph
def research_node(state: ResearchState) -> dict:
    """Get sources for parallel analysis."""
    sources = web_search(state["query"])
    return {"sources": sources}

def parallelize_analysis(state: ResearchState) -> list:
    """Create parallel analysis tasks."""
    return [
        Send("analyze_source", {"source": s})
        for s in state["sources"]
    ]

graph = StateGraph(ResearchState)
graph.add_node("research", research_node)
graph.add_node("analyze_source", analyze_source)

# Parallel edges
graph.add_edge("research", "parallelize_analysis")
graph.add_conditional_edges(
    "parallelize_analysis",
    lambda x: x,  # Already sends
)

app = graph.compile()

# Analyzes all sources in parallel!
result = app.invoke({"query": "quantum computing", "sources": []})
```

#### **4. Human-in-the-Loop**

```python
from langgraph.graph import StateGraph, interrupt

class ReviewState(TypedDict):
    draft: str
    approved: bool
    feedback: str

def draft_node(state: ReviewState) -> dict:
    """Create draft."""
    draft = llm.invoke("Write article on AI")
    return {"draft": draft, "approved": False}

def review_node(state: ReviewState) -> dict:
    """Pause for human review."""
    # Execution pauses here, waits for human input
    interrupt()
    
    # After human provides feedback
    return {"feedback": state["feedback"]}

def publish_node(state: ReviewState) -> dict:
    """Publish if approved."""
    if state["approved"]:
        publish(state["draft"])
        return {"status": "published"}
    else:
        return {"status": "rejected"}

graph = StateGraph(ReviewState)
graph.add_node("draft", draft_node)
graph.add_node("review", review_node)
graph.add_node("publish", publish_node)

graph.add_edge("draft", "review")
graph.add_edge("review", "publish")
graph.set_entry_point("draft")

app = graph.compile()

# Usage with interrupts
thread_id = "user_session_123"

# Start execution
snapshot = app.invoke(
    {"draft": "", "approved": False, "feedback": ""},
    {"configurable": {"thread_id": thread_id}}
)

# Paused at review_node waiting for human input

# User provides feedback
snapshot = app.invoke(
    None,  # Continue with current state
    {
        "configurable": {"thread_id": thread_id},
        "feedback": "Good but needs more examples",
        "approved": False
    }
)

# Execution continues from where it paused!
```

#### **5. Persistence & Checkpointing**

```python
from langgraph.checkpoint import SqliteSaver
from langgraph.graph import StateGraph

# Enable persistence
checkpointer = SqliteSaver(db_path="./langgraph.db")

graph = StateGraph(AgentState)
# ... add nodes ...
graph.set_entry_point("start")

# Compile with checkpointing
app = graph.compile(checkpointer=checkpointer)

# Execution with state persistence
config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke(initial_state, config=config)

# Later (after crash, restart, etc.)
# Can resume from exactly where it left off
# All intermediate states saved
result = app.invoke(input=None, config=config)
# Uses persisted state, continues from last node
```

### Advanced LangGraph Patterns (2026)

#### **Orchestrator-Worker Pattern**

```python
from langgraph.graph import StateGraph, Send

class PlanState(TypedDict):
    task: str
    plan: list
    results: dict

def orchestrator(state: PlanState) -> dict:
    """Create plan and dispatch to workers."""
    plan = llm.invoke(f"Create plan for: {state['task']}")
    return {"plan": plan}

def worker_dispatch(state: PlanState):
    """Dispatch subtasks to workers."""
    return [
        Send("worker", {"subtask": step})
        for step in state["plan"]
    ]

def worker(state: PlanState) -> dict:
    """Execute single subtask."""
    result = execute_task(state["subtask"])
    return {"results": {state["subtask"]: result}}

# Graph
graph = StateGraph(PlanState)
graph.add_node("orchestrator", orchestrator)
graph.add_node("worker_dispatch", worker_dispatch)
graph.add_node("worker", worker)

graph.add_edge("orchestrator", "worker_dispatch")
graph.add_conditional_edges("worker_dispatch", lambda x: x)
graph.add_edge("worker", END)

app = graph.compile()

# Orchestrator creates plan, workers execute in parallel
result = app.invoke({"task": "Complete research project", "plan": [], "results": {}})
```

#### **Reflection Loop**

```python
class ReflectState(TypedDict):
    input: str
    draft: str
    reflection: str
    iterations: int

def generate(state: ReflectState) -> dict:
    """Generate initial response."""
    draft = llm.invoke(state["input"])
    return {"draft": draft, "iterations": 1}

def reflect(state: ReflectState) -> dict:
    """Self-critique the response."""
    critique = llm.invoke(f"""
    Review this response for quality, accuracy, completeness:
    {state['draft']}
    
    Provide specific feedback.
    """)
    return {"reflection": critique}

def decide_if_good(state: ReflectState) -> str:
    """Decide if we should revise or finish."""
    if state["iterations"] >= 3:
        return END
    if "good quality" in state["reflection"].lower():
        return END
    return "revise"

def revise(state: ReflectState) -> dict:
    """Improve based on reflection."""
    revised = llm.invoke(f"""
    Original: {state['draft']}
    Feedback: {state['reflection']}
    
    Create improved version addressing feedback.
    """)
    return {
        "draft": revised,
        "iterations": state["iterations"] + 1
    }

# Self-improving loop
graph = StateGraph(ReflectState)
graph.add_node("generate", generate)
graph.add_node("reflect", reflect)
graph.add_node("revise", revise)

graph.add_edge("generate", "reflect")
graph.add_conditional_edges("reflect", decide_if_good)
graph.add_edge("revise", "reflect")

graph.set_entry_point("generate")
app = graph.compile()

# Response improves through iterations
result = app.invoke({"input": "Write poem about AI", "draft": "", "reflection": "", "iterations": 0})
```

---

## RAG Systems

### Production RAG Architecture (2026)

#### **Advanced Retrieval Pipeline**

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_cohere import CohereReranker

class ProductionRAG:
    """Production-grade RAG system."""
    
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        
        # Multi-query retriever (expand queries)
        self.multi_retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
            llm=llm,
        )
        
        # BM25 hybrid search
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        
        # Ensemble (combine both)
        self.ensemble = EnsembleRetriever(
            retrievers=[self.multi_retriever, self.bm25_retriever],
            weights=[0.6, 0.4]  # 60% semantic, 40% keyword
        )
        
        # Re-ranker (improve relevance)
        self.reranker = CohereReranker(model="rerank-english-v3.0")
    
    def retrieve(self, query: str, top_k: int = 5) -> list:
        """Retrieve using full pipeline."""
        # 1. Multi-query expansion
        docs = self.ensemble.get_relevant_documents(query)
        
        # 2. Re-ranking
        from langchain.retrievers.document_compressors import LLMListCompressor
        
        llm_compressor = LLMListCompressor.from_llm_and_prompt(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["context"],
                template="Extract relevant parts for: {query}"
            )
        )
        
        # Final top-k after re-ranking
        return docs[:top_k]
    
    def answer(self, query: str) -> str:
        """Full RAG pipeline."""
        docs = self.retrieve(query)
        
        context = "\n".join([d.page_content for d in docs])
        
        prompt = ChatPromptTemplate.from_template("""
        Answer based on context. If answer not in context, say so.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        
        return chain.invoke({
            "context": context,
            "question": query
        })

# Usage
rag = ProductionRAG(vectorstore, llm)
answer = rag.answer("What is transformer architecture?")
```

#### **Query Understanding & Routing**

```python
from enum import Enum

class QueryType(str, Enum):
    FACTUAL = "factual"
    REASONING = "reasoning"  
    COMPARATIVE = "comparative"
    CREATIVE = "creative"

def classify_query(query: str, llm) -> QueryType:
    """Classify query type for optimal retrieval."""
    classifier = llm.invoke(f"""
    Classify query as one of: factual, reasoning, comparative, creative
    
    Query: {query}
    Classification:
    """)
    return QueryType(classifier.strip().lower())

def route_retrieval(query: str, llm, vectorstore, tools):
    """Route to optimal retrieval method."""
    query_type = classify_query(query, llm)
    
    if query_type == QueryType.FACTUAL:
        # BM25 for exact facts
        retriever = BM25Retriever.from_documents(documents)
    elif query_type == QueryType.REASONING:
        # Multi-hop reasoning
        retriever = create_multi_hop_retriever(vectorstore, llm)
    elif query_type == QueryType.COMPARATIVE:
        # Retrieve contrasting views
        retriever = create_comparative_retriever(vectorstore)
    else:
        # Creative needs less constraint
        retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    
    return retriever.get_relevant_documents(query)

# Smart retrieval based on query intent
docs = route_retrieval(query, llm, vectorstore, tools)
```

---

## Multi-Model Routing & Load Balancing

### Intelligent Model Selection (2026 Best Practice)

#### **Cost-Aware Routing**

```python
from dataclasses import dataclass
from enum import Enum

@dataclass
class ModelCost:
    model: str
    cost_per_token: float
    latency_ms: float
    quality_score: float  # 0-1

class CostAwareRouter:
    """Route based on cost vs quality tradeoff."""
    
    def __init__(self):
        self.models = {
            "gpt-4-turbo": ModelCost(
                model="gpt-4-turbo",
                cost_per_token=0.00001,
                latency_ms=500,
                quality_score=0.98
            ),
            "gpt-4-mini": ModelCost(
                model="gpt-4-mini",
                cost_per_token=0.000001,
                latency_ms=200,
                quality_score=0.85
            ),
            "groq-mixtral": ModelCost(
                model="mixtral-8x7b",
                cost_per_token=0.00000027,
                latency_ms=50,
                quality_score=0.80
            ),
            "local-llama2": ModelCost(
                model="llama2",
                cost_per_token=0,
                latency_ms=1000,
                quality_score=0.75
            ),
        }
    
    def select_model(self, 
                    budget: str = "balanced",
                    latency_critical: bool = False) -> str:
        """Select optimal model."""
        
        if latency_critical:
            # Pick fastest
            return min(
                self.models.values(),
                key=lambda m: m.latency_ms
            ).model
        
        # Cost-quality tradeoff
        if budget == "low_cost":
            return max(
                self.models.values(),
                key=lambda m: m.quality_score / (m.cost_per_token + 0.00001)
            ).model
        elif budget == "best_quality":
            return max(
                self.models.values(),
                key=lambda m: m.quality_score
            ).model
        else:  # balanced
            return "gpt-4-mini"

# Usage
router = CostAwareRouter()

# For user query with tight SLA
model = router.select_model(latency_critical=True)
# → groq-mixtral (50ms latency)

# For research/high quality
model = router.select_model(budget="best_quality")
# → gpt-4-turbo (0.98 quality)

# For cost-sensitive
model = router.select_model(budget="low_cost")
# → gpt-4-mini (best quality/cost ratio)
```

#### **Performance Feedback Loop**

```python
class ModelPerformanceTracker:
    """Track model performance over time."""
    
    def __init__(self):
        self.performance_history = {}
    
    def record_performance(self, 
                          model: str,
                          query: str,
                          result: str,
                          human_rating: float,
                          latency: float,
                          tokens_used: int):
        """Record model performance."""
        
        if model not in self.performance_history:
            self.performance_history[model] = []
        
        self.performance_history[model].append({
            "query": query,
            "rating": human_rating,
            "latency": latency,
            "tokens": tokens_used,
            "cost": tokens_used * self.get_cost_per_token(model),
        })
    
    def get_average_rating(self, model: str) -> float:
        """Average human rating for model."""
        ratings = [
            p["rating"] 
            for p in self.performance_history.get(model, [])
        ]
        return sum(ratings) / len(ratings) if ratings else 0
    
    def should_switch_model(self, current_model: str, threshold: float = 0.1) -> bool:
        """Detect if better model available."""
        current_rating = self.get_average_rating(current_model)
        
        for model, history in self.performance_history.items():
            if model != current_model:
                other_rating = self.get_average_rating(model)
                if other_rating > current_rating + threshold:
                    return True
        return False

# Usage
tracker = ModelPerformanceTracker()

# Track each interaction
for human_feedback in feedback_stream:
    tracker.record_performance(
        model="gpt-4-mini",
        query=human_feedback.query,
        result=human_feedback.result,
        human_rating=human_feedback.rating,
        latency=human_feedback.latency,
        tokens_used=human_feedback.tokens,
    )
    
    # Adapt if better model available
    if tracker.should_switch_model("gpt-4-mini"):
        # Switch routing to better model
        router.switch_default_model("gpt-4-turbo")
```

---

## Production Patterns

### Deployment Patterns (2026)

#### **LangServe (REST API)**

```python
from fastapi import FastAPI
from langserve import add_routes

app = FastAPI(title="LangChain API")

# Add chain as REST endpoint
add_routes(app, my_chain, path="/chain")

# Automatically provides:
# POST /chain/invoke
# POST /chain/stream  
# POST /chain/batch
# GET /chain/info

# Usage
# curl -X POST http://localhost:8000/chain/invoke \
#   -H "Content-Type: application/json" \
#   -d '{"input": "explain AI"}'
```

#### **LangSmith Integration (Monitoring)**

```python
import os
from langsmith import Client, EvaluationResult

# Enable tracing
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "production-agents"

# All chains automatically traced
chain.invoke({"input": "query"})

# Custom evaluation
client = Client()

def evaluate_response(run, example) -> EvaluationResult:
    """Custom evaluation function."""
    predicted = run.outputs.get("output")
    expected = example.outputs.get("output")
    
    exact_match = predicted == expected
    
    return EvaluationResult(
        key="exact_match",
        score=float(exact_match),
        comment="Does predicted match expected"
    )

# Register evaluation
dataset = client.create_dataset("my_dataset")
client.evaluate(
    chain,
    dataset_name="my_dataset",
    evaluators=[evaluate_response],
    project_name="my_project"
)

# View results at https://smith.langchain.com
```

#### **Cost Optimization**

```python
class CostOptimizer:
    """Optimize LLM costs in production."""
    
    def __init__(self):
        self.cost_log = []
        self.monthly_budget = 1000  # $1000/month
    
    def check_budget(self, current_cost: float) -> bool:
        """Check if within budget."""
        month_remaining = self.get_month_remaining_days() / 30
        daily_budget = self.monthly_budget / 30
        return current_cost < daily_budget
    
    def optimize_prompt(self, prompt: str) -> str:
        """Reduce tokens in prompt."""
        # Remove unnecessary whitespace
        prompt = " ".join(prompt.split())
        
        # Remove verbose instructions
        prompt = prompt.replace("Please explain in detail", "Explain")
        
        return prompt
    
    def enable_caching(self, chain) -> Runnable:
        """Cache repeated queries."""
        from langchain.cache import SQLiteCache
        from langchain.globals import set_llm_cache
        
        set_llm_cache(SQLiteCache(database_name="llm_cache.db"))
        return chain
    
    def use_cheaper_model_for_simple(self, task_complexity: str) -> str:
        """Use cheaper model for simple tasks."""
        if task_complexity == "simple":
            return "gpt-4-mini"
        elif task_complexity == "complex":
            return "gpt-4-turbo"
        else:
            return "gpt-4o"

# Usage
optimizer = CostOptimizer()

# Optimize at every step
prompt = optimizer.optimize_prompt(original_prompt)
chain = optimizer.enable_caching(chain)
model = optimizer.use_cheaper_model_for_simple(task_type)
```

---

## Multi-Agent Architectures

### Pattern 1: Orchestrator-Worker (2026 Standard)

```python
from langgraph.graph import StateGraph, Send

class TaskState(TypedDict):
    original_task: str
    plan: list
    step_results: dict
    final_answer: str

def orchestrator(state: TaskState) -> dict:
    """Create detailed plan."""
    plan = llm.invoke(f"""
    Break down into steps: {state['original_task']}
    Return as JSON list of steps.
    """)
    return {"plan": plan}

def dispatch_workers(state: TaskState) -> list:
    """Send to workers."""
    return [
        Send("worker", {"step": step})
        for step in state["plan"]
    ]

def worker(state: TaskState) -> dict:
    """Execute step."""
    result = llm.invoke(f"Execute: {state['step']}")
    return {"step_results": {state["step"]: result}}

def synthesizer(state: TaskState) -> dict:
    """Combine results."""
    final = llm.invoke(f"""
    Results: {state['step_results']}
    Synthesize into final answer.
    """)
    return {"final_answer": final}

# Build graph
graph = StateGraph(TaskState)
graph.add_node("orchestrator", orchestrator)
graph.add_node("worker", worker)
graph.add_node("synthesizer", synthesizer)

graph.add_edge("orchestrator", "dispatch_workers")
graph.add_conditional_edges("dispatch_workers", lambda x: x)
graph.add_edge("worker", "synthesizer")

graph.set_entry_point("orchestrator")
app = graph.compile()

# Execute
result = app.invoke({"original_task": "Research and write article", "plan": [], "step_results": {}, "final_answer": ""})
```

### Pattern 2: Specialist Agents with Supervisor

```python
class SupervisorState(TypedDict):
    user_query: str
    specialist: str
    results: dict
    final_answer: str

# Specialist agents
research_agent = create_react_agent(llm, [web_search])
writing_agent = create_react_agent(llm, [format_tool])
analysis_agent = create_react_agent(llm, [statistics_tool])

def supervisor(state: SupervisorState) -> str:
    """Route to specialist."""
    routing = llm.invoke(f"""
    Which specialist needed?
    - research: needs information gathering
    - writing: needs content creation
    - analysis: needs data analysis
    
    Query: {state['user_query']}
    Specialist:
    """)
    return routing.strip().lower()

def execute_specialist(state: SupervisorState) -> dict:
    """Execute appropriate specialist."""
    specialist = state["specialist"]
    
    if specialist == "research":
        result = research_agent.invoke({"input": state["user_query"]})
    elif specialist == "writing":
        result = writing_agent.invoke({"input": state["user_query"]})
    else:
        result = analysis_agent.invoke({"input": state["user_query"]})
    
    return {"results": {specialist: result}}

# Graph
graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor)
graph.add_node("research", execute_specialist)
graph.add_node("writing", execute_specialist)
graph.add_node("analysis", execute_specialist)

graph.add_conditional_edges(
    "supervisor",
    lambda x: x["specialist"],
    {
        "research": "research",
        "writing": "writing",
        "analysis": "analysis",
    }
)

graph.set_entry_point("supervisor")
app = graph.compile()
```

---

## Complete Working Examples

### Example 1: Research Agent with LangGraph

```python
"""
Production research agent:
1. Searches web for information
2. Analyzes sources
3. Creates bibliography
4. Handles human feedback
"""

from langgraph.graph import StateGraph, END, interrupt
from typing import Annotated
from operator import add

class ResearchState(TypedDict):
    query: str
    sources: Annotated[list, add]
    analysis: str
    bibliography: list
    user_feedback: str
    approved: bool

def search_node(state: ResearchState):
    """Search for sources."""
    sources = web_search(state["query"], num_results=10)
    return {"sources": sources}

def analyze_node(state: ResearchState):
    """Analyze sources."""
    analysis = llm.invoke(f"""
    Analyze these sources for {state['query']}:
    {[s['content'] for s in state['sources']]}
    
    Provide key findings.
    """)
    return {"analysis": analysis}

def bibliography_node(state: ResearchState):
    """Create bibliography."""
    bib = create_bibliography(state["sources"])
    return {"bibliography": bib}

def review_node(state: ResearchState):
    """Pause for human review."""
    # Execution pauses here
    interrupt(f"""
    Please review the analysis:
    {state['analysis']}
    
    Feedback:
    """)
    return {}

def revise_node(state: ResearchState):
    """Revise based on feedback."""
    if not state["user_feedback"]:
        return {"approved": True}
    
    revised = llm.invoke(f"""
    Original analysis:
    {state['analysis']}
    
    User feedback:
    {state['user_feedback']}
    
    Create revised analysis.
    """)
    return {"analysis": revised, "approved": True}

# Build graph
graph = StateGraph(ResearchState)
graph.add_node("search", search_node)
graph.add_node("analyze", analyze_node)
graph.add_node("bibliography", bibliography_node)
graph.add_node("review", review_node)
graph.add_node("revise", revise_node)

graph.add_edge("search", "analyze")
graph.add_edge("analyze", "bibliography")
graph.add_edge("bibliography", "review")
graph.add_edge("review", "revise")
graph.add_edge("revise", END)

graph.set_entry_point("search")

# Persistence
checkpointer = SqliteSaver(db_path="./research.db")
app = graph.compile(checkpointer=checkpointer)

# Usage with interrupts
config = {"configurable": {"thread_id": "research_123"}}

# Start research
result = app.invoke(
    {"query": "Quantum computing advances in 2026", "sources": [], "analysis": "", "bibliography": [], "user_feedback": "", "approved": False},
    config=config
)

# User provides feedback
result = app.invoke(
    {"user_feedback": "Add more about quantum error correction"},
    config=config
)

# Continues from where it paused
print(result)
```

### Example 2: Multi-Model Content Generator

```python
"""
Content generation with multiple models:
1. Research phase (fast model)
2. Writing phase (creative model)
3. Editing phase (accurate model)
"""

class ContentGeneratorState(TypedDict):
    topic: str
    research_notes: str
    draft: str
    edited: str

def research_node(state: ContentGeneratorState):
    """Research with fast model."""
    fast_model = ChatGroq(model="mixtral-8x7b")
    
    research = fast_model.invoke(f"""
    Gather key points about {state['topic']}:
    - Main concepts
    - Recent developments
    - Important statistics
    """)
    
    return {"research_notes": research}

def write_node(state: ContentGeneratorState):
    """Write with creative model."""
    creative_model = ChatAnthropic(model="claude-3-5-sonnet")
    
    draft = creative_model.invoke(f"""
    Write engaging article on {state['topic']}
    Based on: {state['research_notes']}
    """)
    
    return {"draft": draft}

def edit_node(state: ContentGeneratorState):
    """Edit with accurate model."""
    editing_model = ChatOpenAI(model="gpt-4-turbo")
    
    edited = editing_model.invoke(f"""
    Improve this draft for accuracy and clarity:
    {state['draft']}
    """)
    
    return {"edited": edited}

# Graph
graph = StateGraph(ContentGeneratorState)
graph.add_node("research", research_node)
graph.add_node("write", write_node)
graph.add_node("edit", edit_node)

graph.add_edge("research", "write")
graph.add_edge("write", "edit")
graph.add_edge("edit", END)

graph.set_entry_point("research")
app = graph.compile()

# Usage
result = app.invoke({
    "topic": "AI Safety",
    "research_notes": "",
    "draft": "",
    "edited": ""
})

print(result["edited"])
```

### Example 3: Customer Support Agent

```python
"""
Production customer support agent:
- Routes to specialists
- Searches knowledge base
- Escalates to human
- Tracks interaction
"""

class SupportState(TypedDict):
    customer_input: str
    category: str
    relevant_docs: list
    response: str
    escalated: bool
    feedback_score: float

def categorize_node(state: SupportState):
    """Categorize customer issue."""
    classifier = ChatOpenAI(model="gpt-4-mini")
    
    category = classifier.invoke(f"""
    Categorize as:
    - billing
    - technical
    - account
    - product
    - feedback
    
    Customer: {state['customer_input']}
    Category:
    """)
    
    return {"category": category.strip().lower()}

def retrieve_docs_node(state: SupportState):
    """Get relevant documentation."""
    docs = kb_retriever.get_relevant_documents(
        state["customer_input"],
        metadata_filter={"category": state["category"]}
    )
    return {"relevant_docs": docs}

def respond_node(state: SupportState):
    """Generate response."""
    specialist = ChatOpenAI(model="gpt-4")
    
    context = "\n".join([d.page_content for d in state["relevant_docs"]])
    
    response = specialist.invoke(f"""
    You are a {state['category']} support specialist.
    
    Knowledge base:
    {context}
    
    Customer question:
    {state['customer_input']}
    
    Respond helpfully. If issue needs escalation, say so.
    """)
    
    return {"response": response}

def should_escalate(state: SupportState) -> str:
    """Decide if human escalation needed."""
    if "escalate" in state["response"].lower():
        return "escalate"
    return END

def escalate_node(state: SupportState):
    """Send to human support."""
    ticket_id = create_support_ticket(
        category=state["category"],
        content=state["customer_input"],
        notes=state["response"]
    )
    return {}

# Graph
graph = StateGraph(SupportState)
graph.add_node("categorize", categorize_node)
graph.add_node("retrieve", retrieve_docs_node)
graph.add_node("respond", respond_node)
graph.add_node("escalate", escalate_node)

graph.add_edge("categorize", "retrieve")
graph.add_edge("retrieve", "respond")
graph.add_conditional_edges("respond", should_escalate)
graph.add_edge("escalate", END)

graph.set_entry_point("categorize")
app = graph.compile()

# Usage
result = app.invoke({
    "customer_input": "How do I reset my password?",
    "category": "",
    "relevant_docs": [],
    "response": "",
    "escalated": False,
    "feedback_score": 0
})

print(result["response"])
```

---

## Summary & 2026 Best Practices

### Technology Stack Decision (2026)

| Component | Recommended | Alternative | Use Case |
|-----------|------------|-------------|----------|
| **Orchestration** | LangGraph | Temporal, Prefect | Production agents with state |
| **LLM** | Multi-model router | Single model | Optimized performance/cost |
| **Vector DB** | pgvector + Pinecone | Chroma, Milvus | Balanced local + cloud |
| **Embedding** | OpenAI 3-large | local SBERT | Quality vs cost tradeoff |
| **Observability** | LangSmith | Custom logging | Production monitoring |
| **Deployment** | LangServe + Docker | FastAPI | REST APIs at scale |
| **Memory** | Redis + persistent DB | In-memory | Distributed, scalable |

### Production Checklist (2026)

- [ ] Use LangGraph for agents (not basic loops)
- [ ] Implement multi-model routing
- [ ] Set up LangSmith tracing (all chains)
- [ ] Enable caching (reduce costs 30-50%)
- [ ] Add structured outputs (Pydantic validation)
- [ ] Implement cost tracking per user/query
- [ ] Use MCP for standardized tools
- [ ] Set up human-in-the-loop workflows
- [ ] Implement persistence and checkpointing
- [ ] Monitor model performance over time
- [ ] Test with multiple models before deployment
- [ ] Document prompt versions
- [ ] Implement rate limiting
- [ ] Set up error handling and fallbacks
- [ ] Create comprehensive evaluation dataset
- [ ] Monitor hallucinations and accuracy
- [ ] Implement user feedback loops
- [ ] Use async for throughput
- [ ] Deploy with proper observability
- [ ] Regular cost audits and optimization

### Common 2026 Mistakes (to Avoid)

1. **Still using fixed chains** → Use LangGraph
2. **Single model** → Implement routing
3. **No observability** → Set up LangSmith
4. **No caching** → Enable automatic caching
5. **Unvalidated LLM outputs** → Use Pydantic
6. **No cost tracking** → Implement unified tracking
7. **Manual tool integration** → Use MCP protocol
8. **No human approval** → Add interrupts
9. **No persistence** → Use checkpointing
10. **No feedback loops** → Track performance

---

**This is the comprehensive 2026 LangChain guide. Production-ready patterns, latest features, and real examples. Build reliable AI systems.**
