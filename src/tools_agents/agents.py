from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.tools import tool
from langchain_core.messages import ToolCall


from dotenv import load_dotenv, find_dotenv

# Load environment variables from.env file
load_dotenv(find_dotenv(), override=True)

from tools_agents.tools import (rag_search_filter,
                         rag_search,
                         fetch_abstract_from_arxiv,
                         web_search,
                         final_answer)

# Create a model
LLM = ChatGroq(model="llama-3.1-70b-versatile",
                      stop_sequences="[end]",
                      temperature=0.)

# Define the list of tools available to the oracle.
TOOLS_LIST = [
    rag_search_filter,
    rag_search,
    fetch_abstract_from_arxiv,
    web_search,
    final_answer
]

def get_prompt():
    # Define the system prompt guiding the AI's decision-making process.
    system_prompt = (
        '''As the AI oracle, your primary role is to make informed decisions based on the user's query and the list of tools at your disposal.
        When processing a query, take note of the tools that have already been utilized in the scratchpad. 
        Avoid reusing a tool that has been previously applied to the same query. 
        Additionally, refrain from using any tool more than twice; if a tool has already appeared twice in the scratchpad, do not select it again.
        Your objective is to gather information from a wide range of sources before providing a response to the user. 
        Continue to collect and process data until you have accumulated sufficient information to answer the user's question. 
        Once you have compiled the necessary knowledge in the scratchpad, utilize the final_answer tool to provide a comprehensive response.
        Also keep track of different steps taken in terms of tools used in the scratchpad so that it can be used in research_steps in final respone from final_answer tool.
        The steps taken should be presented in a numbered list format in research_steps obtainef from the final_answer tool.
        Sources in the final_answer tool should also be mentioned in a numbered list format, mentioning the name and link if any of the sources.'''
    )


    # Create a prompt template for the conversation flow.
    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),  # Define the AI's role and rules.
        
        # Insert past chat messages to maintain context.
        MessagesPlaceholder(variable_name='chat_history'),
        
        # Insert user's input dynamically.
        ('user', '{input}'),
        
        # Include the assistant's scratchpad to track tool usage and intermediate steps.
        ('assistant', 'scratchpad: {scratchpad}'),
    ])

    return prompt


# Function to create the scratchpad from the intermediate tool calls.
def create_scratchpad(intermediate_steps: list[ToolCall]) -> str:
    research_steps = []
    
    # Loop over each step and process tool calls with actual outputs.
    for i, action in enumerate(intermediate_steps):
        if action.log != 'TBD':
            research_steps.append(
                f'Tool: {action.tool}, input: {action.tool_input}\n'
                f'Output: {action.log}'
            )
    
    # Join the research steps into a readable log.
    return '\n---\n'.join(research_steps)

def create_oracle(llm=LLM, tools=TOOLS_LIST):
    prompt = get_prompt()
    oracle = (
        {
            'input': lambda x: x['input'],
            'chat_history': lambda x: x['chat_history'],
            'scratchpad': lambda x: create_scratchpad(intermediate_steps=x['intermediate_steps']),
        }
        | prompt
        | llm.bind_tools(tools, tool_choice='any')
    )

    return oracle

def main(llm=LLM, tools=TOOLS_LIST):
    oracle = create_oracle(llm, tools)
    return oracle

if __name__ == '__main__':
    oracle = main()