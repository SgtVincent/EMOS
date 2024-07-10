from typing import List, Tuple, Dict, Union, Callable, Optional
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import random


# from habitat_mas.agents.llm_agent_base import LLMAgentBase


class DummyAgent:
    def __init__(self, **kwargs):
        # TODO check if agent_index is needed in llm agent
        self.agent_index = kwargs.get("agent_index", 0)

    def chat(self, message: str) -> Optional[dict]:
        """
        Mimic a chatbot always sending the oraclenav action.
        """

        robot = random.choice(["robot_0", "robot_1", "robot_2", "robot_3"])

        return random.choice([
            {
                "name": "nav_to_obj",
                "arguments": {
                    "target_obj": "any_targets|0",
                    robot: robot,
                }
            },
            # {
            #     "name": "nav_to_goal",
            #     "arguments": {
            #         "target_obj": "any_targets|0",
            #         robot: robot,
            #     }
            # },
            # {
            #     "name": "pick",
            #     "arguments": {
            #         "any_targets|0": "any_targets|0",
            #         robot: robot,
            #     }
            # },
            # {
            #     "name": "place",
            #     "arguments": {
            #         "any_targets|0": "any_targets|0",
            #         "TARGET_any_targets|0": "TARGET_any_targets|0",
            #         robot: robot,
            #     }
            # },
            ]
        )
