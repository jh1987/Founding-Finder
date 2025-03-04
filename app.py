import streamlit as st
import pandas as pd
from pathlib import Path
from rag_engine import RAGEngine
import os

# Initialize session state
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = RAGEngine()
if 'page' not in st.session_state:
    st.session_state.page = 0
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = {}

def load_or_refresh_data():
    """Load or refresh the funding data and embeddings."""
    with st.spinner("Loading and processing funding data..."):
        st.session_state.rag_engine.load_and_embed_data('data.csv')
    st.success("Data loaded successfully!")

def handle_basic_info_submit():
    """Handle the submission of basic information form."""
    st.session_state.quiz_data.update({
        'startup_name': st.session_state.startup_name,
        'industry': st.session_state.industry.lower(),
        'stage': st.session_state.stage
    })
    st.session_state.page = 1

def handle_funding_info_submit():
    """Handle the submission of funding information form."""
    st.session_state.quiz_data.update({
        'funding_type': st.session_state.funding_type.lower(),
        'funding_needed': st.session_state.funding_needed,
        'location': st.session_state.location.lower()
    })
    
    # Ensure data is loaded
    if st.session_state.rag_engine.index is None:
        load_or_refresh_data()
    
    with st.spinner("Analyzing funding opportunities and checking eligibility..."):
        # Get matches using RAG engine
        matches = st.session_state.rag_engine.search(st.session_state.quiz_data)
        
        # Generate recommendation
        recommendation = st.session_state.rag_engine.generate_recommendation(
            st.session_state.quiz_data, matches
        )
        
        # Store results
        st.session_state.matches = matches
        st.session_state.recommendation = recommendation
        st.session_state.page = 2

def render_quiz():
    """Render the funding finder quiz interface."""
    st.title('Funding Finder Quiz')
    st.write("""
    Welcome to the Funding Finder! Answer a few questions about your startup 
    to discover the most suitable funding opportunities using AI-powered matching.
    """)
    
    # Sidebar for data management
    with st.sidebar:
        st.header("Data Management")
        if st.button("Refresh Data"):
            load_or_refresh_data()
        
        st.write("---")
        st.write("### About")
        st.write("""
        This app uses advanced AI to match your startup with 
        the most relevant funding opportunities. It combines:
        - Semantic search
        - Eligibility analysis
        - Natural language processing
        - Personalized recommendations
        """)
    
    # Quiz questions
    if st.session_state.page == 0:
        st.subheader("Basic Information")
        with st.form("basic_info"):
            st.text_input(
                "What's your startup's name?",
                key="startup_name",
                value=st.session_state.quiz_data.get('startup_name', '')
            )
            st.selectbox(
                "What industry are you in?",
                options=['Technology', 'Green Energy', 'Healthcare', 'Digital Technology', 
                        'Fintech', 'E-commerce', 'Social Enterprise', 'Other'],
                key="industry",
                index=0
            )
            st.selectbox(
                "What stage is your startup in?",
                options=['Idea', 'Early-stage', 'Growth', 'Scaling'],
                key="stage",
                index=0
            )
            submit_button = st.form_submit_button("Next")
            if submit_button:
                handle_basic_info_submit()
                st.rerun()
    
    elif st.session_state.page == 1:
        st.subheader("Funding Requirements")
        with st.form("funding_info"):
            st.selectbox(
                "What type of funding are you looking for?",
                options=['Grant', 'Loan', 'Investment', 'Any'],
                key="funding_type",
                index=0
            )
            st.number_input(
                "How much funding do you need? (in EUR)",
                min_value=0,
                step=1000,
                value=50000,
                key="funding_needed"
            )
            st.selectbox(
                "Where is your startup based?",
                options=['Berlin', 'Hamburg', 'Munich', 'Other German City', 'Outside Germany'],
                key="location",
                index=0
            )
            
            submit_button = st.form_submit_button("Find Matches")
            if submit_button:
                handle_funding_info_submit()
                st.rerun()
    
    elif st.session_state.page == 2:
        st.subheader("Your Personalized Funding Matches")
        
        # Display startup profile
        st.write("### Your Startup Profile")
        st.write(f"**Startup Name:** {st.session_state.quiz_data['startup_name']}")
        st.write(f"**Industry:** {st.session_state.quiz_data['industry'].title()}")
        st.write(f"**Stage:** {st.session_state.quiz_data['stage']}")
        st.write(f"**Funding Type:** {st.session_state.quiz_data['funding_type'].title()}")
        st.write(f"**Amount Needed:** €{st.session_state.quiz_data['funding_needed']:,}")
        st.write(f"**Location:** {st.session_state.quiz_data['location'].title()}")
        
        # Display AI recommendation
        st.write("### AI-Generated Recommendation")
        st.write(st.session_state.recommendation)
        
        # Display detailed matches
        st.write("### Detailed Funding Opportunities")
        matches = st.session_state.matches
        
        if not matches:
            st.warning("No eligible funding opportunities found matching your criteria.")
        else:
            for match in matches:
                similarity = match['similarity_score'] * 100
                with st.expander(f"{match['Funding Program Name']} - {similarity:.0f}% Match"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("#### Program Details")
                        st.write(f"**Funding Type:** {match['Funding Type']}")
                        st.write(f"**Amount:** {match['Funding Amount/Range']}")
                        st.write(f"**Industry Focus:** {match['Industry Focus']}")
                        st.write(f"**Application Process:** {match['Application Process']}")
                        st.write(f"**Deadline:** {match['Application Deadline']}")
                        st.write(f"**Additional Benefits:** {match['Additional Benefits']}")
                    
                    with col2:
                        st.write("#### Eligibility Analysis")
                        st.write(f"**Status:** ✅ Eligible")
                        st.write(f"**Reason:** {match['eligibility_reason']}")
                        st.write("#### Contact Information")
                        st.write(f"**Website:** {match['Website/Link to Apply']}")
                        st.write(f"**Contact:** {match['Contact Information']}")
        
        if st.button("Start Over"):
            st.session_state.page = 0
            st.session_state.quiz_data = {}
            st.rerun()

def main():
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("""
        OpenAI API key not found! Please set your OPENAI_API_KEY environment variable.
        You can add it to a .env file in the root directory.
        """)
        return
    
    # Render the quiz interface
    render_quiz()

if __name__ == '__main__':
    main() 