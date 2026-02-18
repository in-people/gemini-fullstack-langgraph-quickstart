#!/usr/bin/env python3
"""
临时运行脚本，避免与site-packages中的agent模块冲突
可以在本地运行时使用，避免与site-packages中的agent模块冲突
在本地运行测试！
"""
import sys
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# 配置日志
def setup_logging():
    """设置日志配置"""
    # 创建logs目录
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_temp_{timestamp}.log")
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    return logging.getLogger(__name__)

# 初始化日志
logger = setup_logging()

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# 将项目目录添加到Python路径的最前面
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)

# 加载环境变量
logger.info("📥 开始加载环境变量...")
load_dotenv(os.path.join(current_dir, ".env"))

# 记录环境变量状态
api_key_status = "✅ 已设置" if os.getenv("ALI_QWEN_API_KEY") else "❌ 未设置"
logger.info(f"🔐 环境变量加载完成 - ALI_QWEN_API_KEY: {api_key_status}")
logger.info(f"📍 当前工作目录: {os.getcwd()}")
logger.debug(f"📚 Python路径前3个: {sys.path[:3]}")

print("当前工作目录:", os.getcwd())
print("Python路径:", sys.path[:3])  # 显示前3个路径
print("ALI_QWEN_API_KEY:", os.getenv("ALI_QWEN_API_KEY")[:10] + "..." if os.getenv("ALI_QWEN_API_KEY") else "未设置")

# 验证环境变量
if not os.getenv("ALI_QWEN_API_KEY"):
    logger.error("❌ 环境变量ALI_QWEN_API_KEY未设置")
    print("❌ 环境变量未设置")
    sys.exit(1)
else:
    logger.info("✅ 环境变量已设置")
    print("✅ 环境变量已设置")

# 手动导入项目模块
logger.info("📦 开始导入项目模块...")
try:
    logger.debug("正在导入agent.tools_and_schemas...")
    from agent.tools_and_schemas import SearchQueryList, Reflection
    logger.debug("正在导入agent.state...")
    from agent.state import OverallState, QueryGenerationState, ReflectionState, WebSearchState
    logger.debug("正在导入agent.configuration...")
    from agent.configuration import Configuration
    logger.debug("正在导入agent.prompts...")
    from agent.prompts import get_current_date, query_writer_instructions, answer_instructions
    logger.debug("正在导入agent.utils...")
    from agent.utils import get_research_topic
    logger.debug("正在导入langchain和langgraph模块...")
    from langchain_core.messages import AIMessage
    from langgraph.types import Send
    from langgraph.graph import StateGraph, START, END
    from langchain_core.runnables import RunnableConfig
    from langchain_openai import ChatOpenAI
    
    logger.info("✅ 所有模块导入成功")
    print("✅ 所有模块导入成功")
    
    # 重新定义需要的函数
    def load_key(key_name: str):
        """Load API key from environment"""
        key_value = os.getenv(key_name)
        logger.debug(f"🔑 加载环境变量: {key_name} = {'*' * len(key_value) if key_value else 'None'}")
        return key_value
    
    # Initialize OpenAI client with Qwen model
    logger.info("🤖 初始化Qwen Plus客户端...")
    client = ChatOpenAI(
        temperature=0.0, 
        model="qwen-plus", 
        api_key=load_key("ALI_QWEN_API_KEY"), 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    logger.info("✅ Qwen Plus客户端初始化完成")
    
    # 重新定义节点函数
    def generate_query(state: OverallState, config: RunnableConfig):
        logger.info("🔍 开始执行查询生成节点")
        logger.info(f"📝 研究主题: {get_research_topic(state['messages'])}")
        
        configurable = Configuration.from_runnable_config(config)
        logger.debug(f"⚙️ 配置信息: 模型={configurable.query_generator_model}, 查询数量={configurable.number_of_initial_queries}")
        
        if state.get("initial_search_query_count") is None:
            state["initial_search_query_count"] = configurable.number_of_initial_queries
            logger.debug(f"📊 设置初始查询数量: {configurable.number_of_initial_queries}")
        
        logger.info("🤖 初始化查询生成模型")
        llm = ChatOpenAI(
            model=configurable.query_generator_model,
            temperature=1.0,
            max_retries=2,
            api_key=load_key("ALI_QWEN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        structured_llm = llm.with_structured_output(SearchQueryList)
        
        current_date = get_current_date()
        formatted_prompt = query_writer_instructions.format(
            current_date=current_date,
            research_topic=get_research_topic(state["messages"]),
            number_queries=state["initial_search_query_count"],
        )
        logger.debug(f"📄 提示词格式化完成，查询数量: {state['initial_search_query_count']}")
        
        logger.info("🚀 调用LLM生成搜索查询...")
        result = structured_llm.invoke(formatted_prompt)
        
        logger.info(f"✅ 查询生成完成，共生成 {len(result.query)} 个查询")
        for i, query in enumerate(result.query, 1):
            logger.info(f"   {i}. {query}")
        
        return {"search_query": result.query}
    
    def continue_to_web_research(state):
        return [
            Send("web_research", {"search_query": search_query, "id": int(idx)})
            for idx, search_query in enumerate(state["search_query"])
        ]
    
    def web_research(state: WebSearchState, config: RunnableConfig):
        logger.info("🌐 开始执行网络研究节点")
        logger.info(f"🔍 研究查询: {state['search_query']}")
        
        configurable = Configuration.from_runnable_config(config)
        logger.debug(f"⚙️ 配置信息: 模型={configurable.query_generator_model}")
        
        logger.info("📚 生成研究提示词...")
        formatted_prompt = f"""You are a research assistant. Based on the following research topic, generate a comprehensive response with relevant information:

Research Topic: {state["search_query"]}

Provide detailed information that would typically come from web research. Structure your response with headings, key findings, and relevant data."""
        
        logger.info("🚀 调用LLM进行研究内容生成...")
        response = client.invoke(formatted_prompt)
        
        logger.info(f"✅ 研究完成，生成内容长度: {len(response.content)} 字符")
        logger.debug(f"📄 研究内容预览: {response.content[:200]}...")
        
        resolved_urls = {}
        sources_gathered = []
        logger.debug("📋 当前未使用实际网络搜索，返回模拟结果")
        
        logger.info("📤 返回研究结果")
        return {
            "sources_gathered": sources_gathered,
            "search_query": [state["search_query"]],
            "web_research_result": [response.content],
        }
    
    def reflection(state: OverallState, config: RunnableConfig):
        logger.info("🧠 开始执行反思节点")
        logger.info(f"📊 当前研究循环次数: {state.get('research_loop_count', 0) + 1}")
        
        configurable = Configuration.from_runnable_config(config)
        state["research_loop_count"] = state.get("research_loop_count", 0) + 1
        reasoning_model = state.get("reasoning_model", configurable.reflection_model)
        logger.debug(f"⚙️ 配置信息: 反思模型={reasoning_model}")
        
        current_date = get_current_date()
        # 构建提示词
        research_results = '\n\n---\n\n'.join(state["web_research_result"])
        logger.debug(f"📄 反思提示词格式化完成，研究结果数量: {len(state['web_research_result'])}")
        
        formatted_prompt = f"""Based on the research results below, analyze if the information is sufficient to answer the original question. If there are knowledge gaps, suggest follow-up queries.

Current Date: {current_date}
Research Topic: {get_research_topic(state["messages"])}
Research Results: {research_results}

Respond in JSON format with:
- is_sufficient: boolean
- knowledge_gap: string describing what's missing
- follow_up_queries: list of 1-2 follow-up search queries"""
        
        logger.info("🤖 初始化反思模型")
        llm = ChatOpenAI(
            model=reasoning_model,
            temperature=1.0,
            max_retries=2,
            api_key=load_key("ALI_QWEN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        logger.info("🚀 调用LLM进行反思分析...")
        result = llm.invoke(formatted_prompt)
        
        # 简单解析结果（实际应该使用结构化输出）
        content = result.content.lower()
        is_sufficient = "true" in content or "sufficient" in content
        
        logger.info(f"✅ 反思分析完成")
        logger.info(f"   知识是否充分: {'是' if is_sufficient else '否'}")
        knowledge_gap = "需要更多详细信息" if not is_sufficient else "信息充分"
        logger.info(f"   知识缺口: {knowledge_gap}")
        follow_up_queries = ["补充信息查询"] if not is_sufficient else []
        logger.info(f"   后续查询数量: {len(follow_up_queries)}")
        
        for i, query in enumerate(follow_up_queries, 1):
            logger.info(f"   后续查询 {i}: {query}")
        
        return {
            "is_sufficient": is_sufficient,
            "knowledge_gap": knowledge_gap,
            "follow_up_queries": follow_up_queries,
            "research_loop_count": state["research_loop_count"],
            "number_of_ran_queries": len(state["search_query"]),
        }
    
    def evaluate_research(state, config: RunnableConfig):
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
        logger.info("🎯 开始执行答案生成节点")
        logger.info(f"📊 研究结果数量: {len(state['web_research_result'])}")
        
        configurable = Configuration.from_runnable_config(config)
        reasoning_model = state.get("reasoning_model") or configurable.answer_model
        logger.debug(f"⚙️ 配置信息: 答案模型={reasoning_model}")
        
        logger.info("📚 格式化最终答案提示词...")
        current_date = get_current_date()
        formatted_prompt = answer_instructions.format(
            current_date=current_date,
            research_topic=get_research_topic(state["messages"]),
            summaries="\n---\n\n".join(state["web_research_result"]),
        )
        logger.debug(f"📄 答案提示词格式化完成，研究结果长度: {len(state['web_research_result'])}")
        
        logger.info("🤖 初始化答案生成模型")
        llm = ChatOpenAI(
            model=reasoning_model,
            temperature=0,
            max_retries=2,
            api_key=load_key("ALI_QWEN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        logger.info("🚀 调用LLM生成最终答案...")
        result = llm.invoke(formatted_prompt)
        
        logger.info(f"✅ 答案生成完成，内容长度: {len(result.content)} 字符")
        logger.debug(f"📄 答案内容预览: {result.content[:200]}...")
        
        logger.info("📤 返回最终答案")
        return {
            "messages": [AIMessage(content=result.content)],
            "sources_gathered": state["sources_gathered"],
        }
    
    # 创建graph
    logger.info("🏗️ 开始构建LangGraph流程图...")
    builder = StateGraph(OverallState, config_schema=Configuration)
    
    logger.debug("添加节点: generate_query, web_research, reflection, finalize_answer")
    builder.add_node("generate_query", generate_query)
    builder.add_node("web_research", web_research)
    builder.add_node("reflection", reflection)
    builder.add_node("finalize_answer", finalize_answer)
    
    logger.debug("设置图的连接关系...")
    builder.add_edge(START, "generate_query")
    builder.add_conditional_edges("generate_query", continue_to_web_research, ["web_research"])
    builder.add_edge("web_research", "reflection")
    builder.add_conditional_edges("reflection", evaluate_research, ["web_research", "finalize_answer"])
    builder.add_edge("finalize_answer", END)
    
    logger.info("🔄 编译LangGraph...")
    graph = builder.compile(name="pro-search-agent")
    
    logger.info("✅ Graph创建成功")
    print("✅ Graph创建成功")
    
    # 运行测试
    logger.info("=" * 60)
    logger.info("🚀 LangGraph 研究助手启动 (临时运行脚本)")
    logger.info("=" * 60)
    
    query = "巴菲特投资理念！一定使用中文回答"
    logger.info(f"📥 接收到用户查询: {query}")
    
    initial_state = {
        "messages": [{"role": "user", "content": query}]
    }
    
    logger.info("🔄 开始执行研究流程...")
    start_time = datetime.now()
    
    try:
        result = graph.invoke(initial_state)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("✅ 研究流程执行完成")
        logger.info(f"⏱️  执行耗时: {duration:.2f} 秒")
        
        # 打印结果
        print("\n" + "=" * 60)
        print("🎯 最终研究结果")
        print("=" * 60)
        print(result["messages"][-1].content)
        print("=" * 60)
        
        logger.info("📝 结果已输出到控制台")
        logger.info("📁 详细日志已保存到 logs/ 目录")
        
    except Exception as e:
        logger.error(f"❌ 研究流程执行失败: {str(e)}")
        logger.exception("详细错误信息:")
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        logger.info("=" * 60)
        logger.info("🔚 LangGraph 研究助手执行结束")
        logger.info("=" * 60)
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()