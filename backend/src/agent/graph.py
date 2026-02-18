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

# 配置日志 - 兼容Docker环境
def setup_logging():
    """设置日志配置，兼容Docker环境"""
    # 在Docker环境中，优先使用标准输出，避免文件写入权限问题
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # 配置日志格式
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # 标准输出，Docker友好的方式
        ]
    )
    
    # 如果需要文件日志且有写入权限，也添加文件处理器
    try:
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"agent_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        logging.info(f"📝 日志文件已启用: {log_file}")
    except Exception as e:
        logging.warning(f"⚠️  无法创建日志文件: {e}，将仅使用标准输出")
    
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

# 记录环境变量加载状态
api_key_status = "✅ 已设置" if os.getenv("ALI_QWEN_API_KEY") else "❌ 未设置"
logger.info(f"环境变量加载完成 - ALI_QWEN_API_KEY: {api_key_status}")

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
    logger.info("🔍 开始执行查询生成节点")
    logger.info(f"📝 研究主题: {get_research_topic(state['messages'])}")
    
    configurable = Configuration.from_runnable_config(config)
    logger.debug(f"⚙️ 配置信息: 模型={configurable.query_generator_model}, 查询数量={configurable.number_of_initial_queries}")

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries
        logger.debug(f"📊 设置初始查询数量: {configurable.number_of_initial_queries}")

    # init Qwen Plus
    logger.info("🤖 初始化Qwen Plus模型用于查询生成")
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
    logger.debug(f"📄 提示词格式化完成，查询数量: {state['initial_search_query_count']}")
    
    # Generate the search queries
    logger.info("🚀 调用LLM生成搜索查询...")
    result = structured_llm.invoke(formatted_prompt)
    
    logger.info(f"✅ 查询生成完成，共生成 {len(result.query)} 个查询")
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
    logger.info("🌐 开始执行网络研究节点")
    logger.info(f"🔍 研究查询: {state['search_query']}")
    
    # Configure
    configurable = Configuration.from_runnable_config(config)
    logger.debug(f"⚙️ 配置信息: 模型={configurable.query_generator_model}")
    
    # Simulate research by having the model generate content based on the query
    logger.info("📚 生成研究提示词...")
    formatted_prompt = f"""You are a research assistant. Based on the following research topic, generate a comprehensive response with relevant information:

Research Topic: {state["search_query"]}

Provide detailed information that would typically come from web research. Structure your response with headings, key findings, and relevant data."""

    # Use the client to generate research results
    logger.info("🚀 调用LLM进行研究内容生成...")
    response = client.invoke(formatted_prompt)
    
    logger.info(f"✅ 研究完成，生成内容长度: {len(response.content)} 字符")
    logger.debug(f"📄 研究内容预览: {response.content[:200]}...")
    
    # For now, we'll use placeholder values for sources since we're not actually searching the web
    resolved_urls = {}  # Placeholder for resolved URLs
    sources_gathered = []  # No actual sources since we're not doing real search
    logger.debug("📋 当前未使用实际网络搜索，返回模拟结果")
    
    # Return the generated content as research result
    logger.info("📤 返回研究结果")
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
    time.sleep(2)
    logger.info("🧠 开始执行反思节点")
    logger.info(f"📊 当前研究循环次数: {state.get('research_loop_count', 0) + 1}")
    
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)
    logger.debug(f"⚙️ 配置信息: 反思模型={reasoning_model}")

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    logger.debug(f"📄 反思提示词格式化完成，研究结果数量: {len(state['web_research_result'])}")
    
    # init Reasoning Model
    logger.info("🤖 初始化反思模型")
    llm = ChatOpenAI(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=load_key("ALI_QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    logger.info("🚀 调用LLM进行反思分析...")
    result = client.with_structured_output(Reflection).invoke(formatted_prompt)
    
    logger.info(f"✅ 反思分析完成")
    logger.info(f"   知识是否充分: {'是' if result.is_sufficient else '否'}")
    logger.info(f"   知识缺口: {result.knowledge_gap}")
    logger.info(f"   后续查询数量: {len(result.follow_up_queries)}")
    
    for i, query in enumerate(result.follow_up_queries, 1):
        logger.info(f"   后续查询 {i}: {query}")

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
    time.sleep(2)
    logger.info("🎯 开始执行答案生成节点")
    logger.info(f"📊 研究结果数量: {len(state['web_research_result'])}")
    
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model
    logger.debug(f"⚙️ 配置信息: 答案模型={reasoning_model}")

    # Format the prompt
    logger.info("📚 格式化最终答案提示词...")
    # 使用 \n --- \n\n 把列表元素拼接为一个长字符串
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )
    logger.debug(f"📄 答案提示词格式化完成，研究结果长度: {len(state['web_research_result'])}")

    # init Reasoning Model, using Qwen
    logger.info("🤖 初始化答案生成模型")
    llm = ChatOpenAI(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=load_key("ALI_QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    logger.info("🚀 调用LLM生成最终答案...")
    # result = llm.invoke(formatted_prompt)
    result = client.invoke(formatted_prompt)
    
    logger.info(f"✅ 答案生成完成，内容长度: {len(result.content)} 字符")
    logger.debug(f"📄 答案内容预览: {result.content[:200]}...")

    # Since we don't have actual web sources, return the result as is
    logger.info("📤 返回最终答案")
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
