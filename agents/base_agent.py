from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def process_query(self, query: str, engine):
        pass

    @abstractmethod
    def display_charts(self, st):
        pass

    @abstractmethod
    def generate_data_summary(self, data):
        pass

    @abstractmethod
    def generate_report_section(self):
        pass
