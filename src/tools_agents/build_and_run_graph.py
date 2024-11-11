import os

from tqdm.autonotebook import tqdm, trange

from langchain_core.tools import tool
from langchain_core.agents import AgentAction

from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator

from langgraph.graph import StateGraph, END


from dotenv import load_dotenv, find_dotenv

# Load environment variables from.env file
load_dotenv(find_dotenv(), override=True)

from tools_agents.tools import (rag_search_filter,
                                rag_search,
                                fetch_abstract_from_arxiv,
                                web_search,
                                final_answer)

from tools_agents.agents import create_oracle, TOOLS_LIST

TOOL_STR_TO_FUNC = {
    'rag_search_filter': rag_search_filter,
    'rag_search': rag_search,
    'fetch_abstract_from_arxiv': fetch_abstract_from_arxiv,
    'web_search': web_search,
    'final_answer': final_answer
}

oracle = create_oracle()

def run_oracle(state: dict) -> dict:
    """
    Run the oracle and handle the interaction with the tools.
    Args:
        state (dict): The current state of the conversation.
    Returns:
        dict: The updated state of the conversation after the oracle's response.
    """
    print('run_oracle')
    print(f'intermediate_steps: {state["intermediate_steps"]}')
    print(state)
    
    out = oracle.invoke(state)
    print(out)

    tool_name = out.tool_calls[0]['name']
    tool_args = out.tool_calls[0]['args']

    print(f'tool_name: {tool_name}')
    print(f'tool_args: {tool_args}')

    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log='TBD'  # To be determined later after the tool runs.
    )

    return {'intermediate_steps': [action_out]}


# router() function
def router(state: dict) -> str:
    """
    Route the conversation to the appropriate tool based on the current state.
    Args:
        state (dict): The current state of the conversation.
    Returns:
        str: The name of the tool to be executed.
    """

    if isinstance(state['intermediate_steps'], list):
        print(f"running tool: {state['intermediate_steps'][-1].tool}")
        return state['intermediate_steps'][-1].tool
    else:
        print('Router invalid format')
        return 'final_answer'


# The run_tool() function executes the appropriate tool based on the current state.
def run_tool(state: dict) -> dict:
    """
    Run the appropriate tool based on the current state.
    Args:
        state (dict): The current state of the conversation.
    Returns:
        dict: The updated state of the conversation after the tool's execution.
    """

    tool_name = state['intermediate_steps'][-1].tool
    tool_args = state['intermediate_steps'][-1].tool_input

    print(f'{tool_name}.invoke(input={tool_args})')

    out = TOOL_STR_TO_FUNC[tool_name].invoke(input=tool_args)

    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log=str(out)
    )

    return {'intermediate_steps': [action_out]}


class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]



def build_graph(tools=TOOLS_LIST):
    print("running graph")
    workflow = StateGraph(AgentState)

    workflow.add_node('oracle', run_oracle)
    workflow.add_node('rag_search', run_tool)
    workflow.add_node('rag_search_filter', run_tool)
    workflow.add_node('web_search', run_tool)
    workflow.add_node('fetch_abstract_from_arxiv', run_tool)
    workflow.add_node('final_answer', run_tool)


    workflow.set_entry_point('oracle')
    workflow.add_conditional_edges(source='oracle',
                                path=router)

    for tool_object in tools:
        if tool_object.name != 'final_answer':
            workflow.add_edge(tool_object.name, 'oracle')

    workflow.add_edge('final_answer', END)

    graph = workflow.compile()

    return graph


def build_report(output: dict) -> str:
    
    research_steps = output['research_steps']
    if isinstance(research_steps, list):
        research_steps = '\n'.join([f'- {r}' for r in research_steps])
    
    sources = output['sources']
    if isinstance(sources, list):
        sources = '\n'.join([f'- {s}' for s in sources])
    
    return f"""
        INTRODUCTION
        ------------
        {output['introduction']}
        
        RESEARCH STEPS
        --------------
        {research_steps}
        
        REPORT
        ------
        {output['main_body']}
        
        CONCLUSION
        ----------
        {output['conclusion']}
        
        SOURCES
        -------
        {sources}
    """

graph = build_graph()
def run_oracle(query):
    print('run_oracle')

    output = graph.invoke(query)
    # print(build_report(output['intermediate_steps'][-1].tool_input ))

    return build_report(output['intermediate_steps'][-1].tool_input)

