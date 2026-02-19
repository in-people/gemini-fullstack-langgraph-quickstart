import os
import logging
import time
from datetime import datetime

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

# Configure logging - compatible with Docker environment
def setup_logging():
    """Setup logging configuration, compatible with Docker environment"""
    # In Docker environment, prioritize stdout to avoid file write permission issues
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Configure logging format
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Standard output, Docker-friendly way
        ]
    )
    
    # If file logging is needed and has write permissions, also add file handler
    try:
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"agent_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        logging.info(f"Log file enabled: {log_file}")
    except Exception as e:
        logging.warning(f"Unable to create log file: {e}, will use stdout only")
    
    return logging.getLogger(__name__)

# 初始化日志
logger = setup_logging()

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)

# Load environment variables
load_dotenv()

# Record environment variable loading status
api_key_status = "Set" if os.getenv("ALI_QWEN_API_KEY") else "Not Set"
logger.info(f"Environment variables loaded - ALI_QWEN_API_KEY: {api_key_status}")

def load_key(key_name: str):
    """Load API key from environment"""
    return os.getenv(key_name)

# Initialize OpenAI client with Qwen model
client = ChatOpenAI(
    temperature=0.0, 
    model="qwen-plus", 
    api_key=load_key("ALI_QWEN_API_KEY"), 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses Qwen Plus to create an optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    logger.info("Starting query generation node")
    logger.info(f"Research topic: {get_research_topic(state['messages'])}")
    
    configurable = Configuration.from_runnable_config(config)
    logger.debug(f"Configuration: model={configurable.query_generator_model}, query count={configurable.number_of_initial_queries}")

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries
        logger.debug(f"Setting initial query count: {configurable.number_of_initial_queries}")

    # init Qwen Plus
    logger.info("Initializing Qwen Plus model for query generation")
    llm = ChatOpenAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=load_key("ALI_QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    logger.debug(f"Prompt formatting complete, query count: {state['initial_search_query_count']}")
    
    # Generate the search queries
    logger.info("Calling LLM to generate search queries...")
    result = structured_llm.invoke(formatted_prompt)
    
    logger.info(f"Query generation complete, generated {len(result.query)} queries")
    for i, query in enumerate(result.query, 1):
        logger.info(f"   {i}. {query}")
    
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that simulates web research using the Qwen API.

    Since we're removing actual web search functionality, this node will now generate
    simulated research results using the LLM itself.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    logger.info("Starting web research node")
    logger.info(f"Research query: {state['search_query']}")
    
    # Configure
    configurable = Configuration.from_runnable_config(config)
    logger.debug(f"Configuration: model={configurable.query_generator_model}")
    
    # Simulate research by having the model generate content based on the query
    logger.info("Generating research prompt...")
    formatted_prompt = f"""You are a research assistant. Based on the following research topic, generate a comprehensive response with relevant information:

Research Topic: {state["search_query"]}

Provide detailed information that would typically come from web research. Structure your response with headings, key findings, and relevant data."""

    # Use the client to generate research results
    logger.info("Calling LLM for research content generation...")
    response = client.invoke(formatted_prompt)
    
    logger.info(f"Research complete, generated content length: {len(response.content)} characters")
    logger.debug(f"Research content preview: {response.content[:200]}...")
    
    # For now, we'll use placeholder values for sources since we're not actually searching the web
    resolved_urls = {}  # Placeholder for resolved URLs
    sources_gathered = []  # No actual sources since we're not doing real search
    logger.debug("Currently not using actual web search, returning simulated results")
    
    # Return the generated content as research result
    logger.info("Returning research results")
    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [response.content],
    }

    
def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    logger.info("Starting reflection node")
    logger.info(f"Current research loop count: {state.get('research_loop_count', 0) + 1}")
    
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)
    logger.debug(f"Configuration: reflection model={reasoning_model}")

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    logger.debug(f"Reflection prompt formatting complete, research result count: {len(state['web_research_result'])}")
    
    # init Reasoning Model
    logger.info("Initializing reflection model")
    llm = ChatOpenAI(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=load_key("ALI_QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    logger.info("Calling LLM for reflection analysis...")
    result = client.with_structured_output(Reflection).invoke(formatted_prompt)
    
    logger.info(f"Reflection analysis complete")
    logger.info(f"   Knowledge sufficient: {'Yes' if result.is_sufficient else 'No'}")
    logger.info(f"   Knowledge gap: {result.knowledge_gap}")
    logger.info(f"   Follow-up query count: {len(result.follow_up_queries)}")
    
    for i, query in enumerate(result.follow_up_queries, 1):
        logger.info(f"   Follow-up query {i}: {query}")

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    logger.info("Starting answer generation node")
    logger.info(f"Research result count: {len(state['web_research_result'])}")
    
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model
    logger.debug(f"Configuration: answer model={reasoning_model}")

    # Format the prompt
    logger.info("Formatting final answer prompt...")
    # Use \n --- \n\n to join list elements into a long string
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )
    logger.debug(f"Answer prompt formatting complete, research result length: {len(state['web_research_result'])}")

    # init Reasoning Model, using Qwen
    logger.info("Initializing answer generation model")
    llm = ChatOpenAI(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=load_key("ALI_QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    logger.info("Calling LLM to generate final answer...")
    # result = llm.invoke(formatted_prompt)
    result = client.invoke(formatted_prompt)
    
    logger.info(f"Answer generation complete, content length: {len(result.content)} characters")
    logger.debug(f"Answer content preview: {result.content[:200]}...")

    # Since we don't have actual web sources, return the result as is
    logger.info("Returning final answer")
    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": state["sources_gathered"],  # Pass through any existing sources
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
