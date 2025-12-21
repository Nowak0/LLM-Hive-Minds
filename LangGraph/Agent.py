from langgraph.graph import StateGraph, START, END
from State import State
from MessageClassifier import MessageClassifier


class Agent:
    def __init__(self, llm):
        self.llm = llm
        self.graph_builder = StateGraph(State)
        self.graph = None


    def classify_message(self, state: State):
        last_message = state["messages"][-1]
        classifier_llm = self.llm.with_structured_output(MessageClassifier)

        result = classifier_llm.invoke([
            {
                "role": "system",
                "content": """Classify the user message as either:
                - 'emotional': if it asks for emotional support, therapy, deals with feelings, personal problems, etc.
                - 'logical': if it asks for facts, information, logical analysis, practical solutions, etc.
                """
            },
            {
                "role": "user",
                "content": last_message.content,
            }
        ])

        return {"message_type": result.message_type}


    def router(self, state: State):
        message_type = state.get("message_type", "logical") # logical is default
        if message_type == "emotional":
            return {"next": "emotional"}

        return {"next": "logical"}


    def emotional_agent(self, state: State):
        last_message = state["messages"][-1]
        messages = [
            {
                "role": "system",
                "content": """You are a compassionate converser. Focus on the emotional aspects of the user's message.
                Show empathy, validate their feelings and help them process their emotions.
                Ask thoughtful questions to help them explore their feelings more deeply.
                Avoid giving logical solutions unless explicitly asked.
                """
            },
            {
                "role": "user",
                "content": last_message.content,
            }
        ]

        reply = self.llm.invoke(messages)
        return {"messages": [{"role":"assistant", "content": reply.content}]}


    def logical_agent(self, state: State):
        last_message = state["messages"][-1]
        messages = [
            {
                "role": "system",
                "content": """You are a purely logical assistant. Focus only on facts and information.
                Provide clear, concise answers based on logic and evidence.
                Do not address emotions or provide emotional support.
                Be direct and straightforward in your responses.
                """
            },
            {
                "role": "user",
                "content": last_message.content,
            }
        ]

        reply = self.llm.invoke(messages)
        return {"messages": [{"role": "assistant", "content": reply.content}]}


    def create_graph(self):
        self.graph_builder.add_node("classifier", self.classify_message)
        self.graph_builder.add_node("router", self.router)
        self.graph_builder.add_node("emotional", self.emotional_agent)
        self.graph_builder.add_node("logical", self.logical_agent)

        self.graph_builder.add_edge(START, "classifier")
        self.graph_builder.add_edge("classifier", "router")
        self.graph_builder.add_conditional_edges(
            "router",
            lambda state: state.get("next"),
            {"emotional": "emotional", "logical": "logical"}
        )
        self.graph_builder.add_edge("emotional", END)
        self.graph_builder.add_edge("logical", END)

        self.graph = self.graph_builder.compile()



    def run_chatbot(self):
        state = {"messages": [], "message_type": None}
        self.create_graph()

        while True:
            user_input = input("> ")
            if user_input == "exit":
                print("End")
                break

            state["messages"].append({"role": "system", "content": user_input})
            state = self.graph.invoke(state)

            if state.get("messages") and len(state["messages"]) > 0:
                last_message = state["messages"][-1]
                print(f"Agent: {last_message.content}")
