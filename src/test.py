import sys
from IPython.display import Image, display

# Stage 1 - Foundation
# from config import llm
# print(llm.invoke('Say hello').content)
# sys.exit()

# Stage 2 - RAG — Product Search Tool
# from tools import search_product_catalog; 
# print(search_product_catalog.invoke({'query': 'wireless headphones'}))
# sys.exit()

# Stage 3 - Product Agent Subgraph
from langchain_core.messages import HumanMessage, SystemMessage
# from nodes import product_subgraph, PRODUCT_PROMPT


# result = product_subgraph.invoke({'messages': [
# SystemMessage(content=PRODUCT_PROMPT),
# HumanMessage(content='Show me headphones under 15000')]})
# print(result['messages'][-1].content)
# display(Image(product_subgraph.get_graph().draw_mermaid_png()))
# sys.exit()

# Stage 4 - Support Agent + Tools
# from langchain_core.messages import HumanMessage, SystemMessage
# from nodes import support_subgraph, SUPPORT_PROMPT
# result = support_subgraph.invoke({'messages': [
# SystemMessage(content=SUPPORT_PROMPT),
# HumanMessage(content='Status of order ORD102?')]})
# print(result['messages'][-1].content)
# display(Image(support_subgraph.get_graph().draw_mermaid_png()))
# sys.exit()


# Stage 5 - Orchestrator + Multi-Agent Routing
# from config import llm
# from state import ClassificationResult
# c = llm.with_structured_output(ClassificationResult)
# r = c.invoke('Classify: My order ORD102 is late show me headphones alternatives')
# print(r)
# print('Mixed:', [t.agent for t in r.tasks], 'synthesis:', r.requires_synthesis)
# sys.exit()

# Stage 6 - Synthesizer + Full Graph
# from langchain_core.messages import HumanMessage
# from graph import axiomcart_graph
# result = axiomcart_graph.invoke(
#     {'messages': [HumanMessage(content='Where is my order?')],
#      'user_query': 'ORD102 is late, show me headphones'},
#     {'configurable': {'thread_id': 'test-006'}}
# )
# print(result['final_answer'])
# sys.exit()

# Stage 7 - Human-in-the-Loop (HITL)
# from langchain_core.messages import HumanMessage
from langgraph.types import Command
from graph import axiomcart_graph
cfg = {'configurable': {'thread_id': 'test-hitl'}}
r = axiomcart_graph.invoke(
    {'messages': [HumanMessage(content='Where is my order?')],
    'user_query': 'Where is my order?'}, cfg)
print(axiomcart_graph.get_state(cfg))

print("-"*50)
if '__interrupt__' in r and r['__interrupt__']:
    print('Agent asks:', r['__interrupt__'][0].value)
    r = axiomcart_graph.invoke(Command(resume='ORD102'), cfg)
    print(r['final_answer'])

print("-"*50)
state = axiomcart_graph.get_state(cfg)
for m in state.values["messages"]:
    print(type(m).__name__, m.content)