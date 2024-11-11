import streamlit as st
import time

from src.tools_agents.build_and_run_graph import run_oracle


def run_research_assistant_app():
    # Title at the top of the app
    st.title("AI Research Assistant")

    # Text input box for entering a query
    query = st.text_input("Enter your query:", placeholder="Type your query here")

    # Button to submit the query
    if st.button("Generate Report"):
        if query:
            # Display a progress indicator while processing
            with st.spinner("Generating report... Please wait."):
                
                report = run_oracle(query={'input': query,
                                'chat_history': [],})
            # Display the generated report
            st.text_area("Generated Report:", report, height=300)
        else:
            st.warning("Please enter a query to generate a report.")

if __name__ == "__main__":
    run_research_assistant_app()