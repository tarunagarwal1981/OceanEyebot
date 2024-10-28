# agents/agent_selector.py

from .hull_performance_agent import HullPerformanceAgent
from .speed_consumption_agent import SpeedConsumptionAgent
from .agent_base import Agent
import openai

class AgentSelector:
    def __init__(self):
        self.agents = {
            "hull_performance": HullPerformanceAgent(),
            "speed_consumption": SpeedConsumptionAgent()
        }

    def select_agents(self, query: str):
        decision = self.get_decision(query)
        if decision == "vessel_performance":
            return [self.agents["hull_performance"], self.agents["speed_consumption"]]
        elif decision in self.agents:
            return [self.agents[decision]]
        else:
            return []

    def get_decision(self, query: str) -> str:
        prompt = (
            f"Decide whether the following query is about 'hull performance', 'speed consumption', "
            f"or 'vessel performance' (both), or 'general_info' if none apply.\n\n"
            f"Query: \"{query}\"\n\nDecision:"
        )
        try:
            response = openai.Completion.create(
                engine="gpt-4",
                prompt=prompt,
                max_tokens=10,
                n=1,
                stop=None,
                temperature=0.0,
            )
            decision_text = response.choices[0].text.strip().lower()
            if "vessel performance" in decision_text:
                return "vessel_performance"
            elif "hull performance" in decision_text:
                return "hull_performance"
            elif "speed consumption" in decision_text:
                return "speed_consumption"
            else:
                return "general_info"
        except Exception as e:
            st.error(f"Error in LLM decision: {str(e)}")
            return "general_info"
