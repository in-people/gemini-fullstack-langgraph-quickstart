import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the agent."""

    query_generator_model: str = Field(
        default="qwen-plus",
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    reflection_model: str = Field(
        default="qwen-plus",
        metadata={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )

    answer_model: str = Field(
        default="qwen-plus",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
    )


# 这个方法的主要功能是：

# 1. **获取配置数据源** - 从传入的`RunnableConfig`中提取"configurable"部分，如果没有配置则使用空字典
# 2. **合并环境变量和配置** - 遍历类的所有模型字段，优先从环境变量（大写形式）获取值，如果环境变量不存在则从配置中获取
# 3. **过滤有效值** - 移除所有None值，只保留实际存在的配置值
# 4. **创建实例** - 使用过滤后的配置值创建并返回新的Configuration实例

# 这种方法允许配置通过多种方式提供：环境变量、运行时配置或默认值，为LangGraph代理提供了灵活的配置机制。

# classmethod 类方法，表示这是一个类方法，而不是实例方法。
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)