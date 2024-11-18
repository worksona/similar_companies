import streamlit as st
import openai
from dotenv import load_dotenv
import os

load_dotenv()

# Updated model list to match requested models
OPENAI_MODELS = [
    "GPT-4o",
    "GPT-4o mini",
    "o1-preview",
    "o1-mini",
    "GPT-4 Turbo",
    "GPT-4",
    "GPT-3.5 Turbo"
]

AGENT_PROMPTS = {
    "company_analyst": """role: Company Analysis Specialist
goal: Analyze the target company to identify key characteristics for comparison
task: Conduct a comprehensive analysis of {target_company}. Focus on key characteristics such as industry, size, location, product offerings, target market, and any unique attributes. Consider how these characteristics might relate to their potential need for {our_product}.""",

    "market_researcher": """role: Market Research Expert
goal: Research potential companies similar to the target company
task: Using the profile of {target_company}, research and search for and identify other companies that share similar characteristics. Look for companies in the same or adjacent industries, of similar size, with comparable product offerings or target markets. Aim to find at least 10 potential matches.""",

    "similarity_evaluator": """role: Company Similarity Evaluator
goal: Assess and score the similarity between companies
task: Evaluate each of the potential similar companies identified. Compare them to the profile of {target_company}, considering all relevant factors. Assign a similarity score to each company on a scale of 0 to 1, where 1 is most similar. Provide a brief explanation for each score.""",

    "sales_strategist": """role: Sales Approach Strategist
goal: Develop recommendations for approaching similar companies
task: Based on the analysis of {target_company} and the identified similar companies, develop recommendations for how the sales team should approach these prospects about {our_product}. Consider common characteristics, potential pain points, and how our product might address their needs."""
}

def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {
            'company_profile': None,
            'company_list': None,
            'evaluated_list': None,
            'sales_strategy': None,
            'final_report': None
        }
    if 'current_chat_context' not in st.session_state:
        st.session_state.current_chat_context = 'all'
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []

def run_agent(client, model, prompt, context=""):
    messages = [{"role": "system", "content": prompt}]
    if context:
        messages.append({"role": "user", "content": context})
    
    # Map display names to API model names
    model_mapping = {
        "GPT-4o": "gpt-4-0125-preview",
        "GPT-4o mini": "gpt-4-0125-preview",
        "o1-preview": "gpt-4-0125-preview",
        "o1-mini": "gpt-3.5-turbo-0125",
        "GPT-4 Turbo": "gpt-4-turbo-preview",
        "GPT-4": "gpt-4",
        "GPT-3.5 Turbo": "gpt-3.5-turbo"
    }
    
    api_model = model_mapping.get(model, "gpt-3.5-turbo")
    
    response = client.chat.completions.create(
        model=api_model,
        messages=messages
    )
    return response.choices[0].message.content

def web_search(client, model, query):
    """
    Perform a web search using OpenAI's capabilities and return formatted results
    """
    search_prompt = f"""You are a web search assistant. Search the web for information about: {query}
    
    Please provide:
    1. A comprehensive summary of the findings
    2. Key facts and data points
    3. Relevant sources or references if available
    4. Any additional insights that might be valuable
    
    Format the response in a clear, organized manner."""

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",  # Using GPT-4 Turbo for better search capabilities
            messages=[
                {"role": "system", "content": "You are a web search assistant with access to current information."},
                {"role": "user", "content": search_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error performing web search: {str(e)}"

def run_analysis(api_key, model, company_name, product_desc):
    client = openai.Client(api_key=api_key)
    results = {}
    
    # Company Analysis Specialist
    analyst_prompt = AGENT_PROMPTS["company_analyst"].format(
        target_company=company_name,
        our_product=product_desc
    )
    results['company_profile'] = run_agent(client, model, analyst_prompt)
    
    # Market Research Expert
    researcher_prompt = AGENT_PROMPTS["market_researcher"].format(
        target_company=company_name
    )
    results['company_list'] = run_agent(client, model, researcher_prompt, results['company_profile'])
    
    # Company Similarity Evaluator
    evaluator_prompt = AGENT_PROMPTS["similarity_evaluator"].format(
        target_company=company_name
    )
    results['evaluated_list'] = run_agent(client, model, evaluator_prompt, results['company_list'])
    
    # Sales Approach Strategist
    strategist_prompt = AGENT_PROMPTS["sales_strategist"].format(
        target_company=company_name,
        our_product=product_desc
    )
    context = f"""Company Profile:
{results['company_profile']}

Evaluated Companies:
{results['evaluated_list']}"""
    
    results['sales_strategy'] = run_agent(client, model, strategist_prompt, context)
    
    results['final_report'] = format_final_report(
        company_name, 
        product_desc, 
        results['company_profile'],
        results['evaluated_list'],
        results['sales_strategy']
    )
    
    st.session_state.analysis_results = results
    return results

def format_final_report(company_name, product_desc, profile, evaluated_list, sales_strategy):
    return f"""# SimilarCompanyFinderResult

## {company_name} Profile

{profile}

## Similar Companies

{evaluated_list}

## Approach Recommendations

### Sales Strategies

{sales_strategy}

---

This detailed profile, similarity breakdown, and tailored sales approach recommendations offer your sales team a comprehensive strategy to engage similar companies effectively."""

def get_chat_context(context_type):
    results = st.session_state.analysis_results
    if context_type == 'all':
        return results['final_report']
    elif context_type == 'company_profile':
        return results['company_profile']
    elif context_type == 'company_list':
        return results['company_list']
    elif context_type == 'evaluated_list':
        return results['evaluated_list']
    elif context_type == 'sales_strategy':
        return results['sales_strategy']
    return results['final_report']

def chat_with_results(api_key, model, query, context_type='all'):
    if not st.session_state.analysis_results['final_report']:
        return "Please run analysis first."
    
    client = openai.Client(api_key=api_key)
    context = f"Analysis context: {get_chat_context(context_type)}"
    
    # Include chat history for context
    messages = [
        {"role": "system", "content": "You are a helpful assistant with expertise in analyzing companies and sales strategies. Use the provided analysis to answer questions."},
        {"role": "assistant", "content": context}
    ]
    
    # Add relevant chat history
    for chat in st.session_state.chat_history[-5:]:  # Include last 5 messages for context
        if chat['context_type'] == context_type:
            messages.append({"role": "user", "content": chat['query']})
            messages.append({"role": "assistant", "content": chat['response']})
    
    messages.append({"role": "user", "content": query})
    
    # Map display names to API model names
    model_mapping = {
        "GPT-4o": "gpt-4-0125-preview",
        "GPT-4o mini": "gpt-4-0125-preview",
        "o1-preview": "gpt-4-0125-preview",
        "o1-mini": "gpt-3.5-turbo-0125",
        "GPT-4 Turbo": "gpt-4-turbo-preview",
        "GPT-4": "gpt-4",
        "GPT-3.5 Turbo": "gpt-3.5-turbo"
    }
    
    api_model = model_mapping.get(model, "gpt-3.5-turbo")
    
    response = client.chat.completions.create(
        model=api_model,
        messages=messages
    )
    return response.choices[0].message.content

def main():
    st.title("Similar Company Finder")
    
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox("Model", OPENAI_MODELS)
        company_name = st.text_input("Company Name")
        product_desc = st.text_area("Product Description")
        
        if st.button("Run Analysis"):
            if not all([api_key, company_name, product_desc]):
                st.error("Please fill in all fields")
                return
                
            with st.spinner("Running analysis..."):
                results = run_analysis(api_key, model, company_name, product_desc)
                st.success("Analysis complete!")
    
    # Main content area with tabs
    tabs = st.tabs([
        "Company Analysis",
        "Market Research",
        "Similarity Evaluation",
        "Sales Strategy",
        "Final Report",
        "Chat with Results",  # Renamed from "Chat"
        "Open Chat"  # Renamed from "Web Search"
    ])
    
    if st.session_state.analysis_results['final_report']:
        # Company Analysis Tab
        with tabs[0]:
            st.header("Company Analysis Specialist")
            st.markdown("**Role**: Analyze target company characteristics")
            st.markdown("**Results**:")
            st.markdown(st.session_state.analysis_results['company_profile'])
        
        # Market Research Tab
        with tabs[1]:
            st.header("Market Research Expert")
            st.markdown("**Role**: Identify similar companies")
            st.markdown("**Results**:")
            st.markdown(st.session_state.analysis_results['company_list'])
        
        # Similarity Evaluation Tab
        with tabs[2]:
            st.header("Company Similarity Evaluator")
            st.markdown("**Role**: Score company similarities")
            st.markdown("**Results**:")
            st.markdown(st.session_state.analysis_results['evaluated_list'])
        
        # Sales Strategy Tab
        with tabs[3]:
            st.header("Sales Approach Strategist")
            st.markdown("**Role**: Develop sales recommendations")
            st.markdown("**Results**:")
            st.markdown(st.session_state.analysis_results['sales_strategy'])
        
        # Final Report Tab
        with tabs[4]:
            st.header("Final Report")
            st.markdown(st.session_state.analysis_results['final_report'])
        
        # Enhanced Chat Tab (renamed to "Chat with Results")
        with tabs[5]:
            st.header("Chat with Results")
            
            # Context selector
            context_options = {
                'all': 'All Content',
                'company_profile': 'Company Profile',
                'company_list': 'Similar Companies',
                'evaluated_list': 'Similarity Evaluation',
                'sales_strategy': 'Sales Strategy'
            }
            
            selected_context = st.selectbox(
                "Select what to chat about:",
                options=list(context_options.keys()),
                format_func=lambda x: context_options[x],
                key='chat_context'
            )
            
            st.session_state.current_chat_context = selected_context
            
            # Display the selected content
            st.markdown("### Selected Content:")
            with st.expander("View Content", expanded=True):
                st.markdown(get_chat_context(selected_context))
            
            # Chat interface
            query = st.text_input("Ask a question about the analysis")
            if st.button("Submit"):
                response = chat_with_results(api_key, model, query, selected_context)
                st.markdown("**Response:**")
                st.markdown(response)
                st.session_state.chat_history.append({
                    "query": query,
                    "response": response,
                    "context_type": selected_context
                })
            
            # Chat history with context labels
            with st.expander("Chat History", expanded=True):
                for chat in reversed(st.session_state.chat_history):
                    st.markdown(f"**Context:** {context_options[chat['context_type']]}")
                    st.markdown(f"**Q:** {chat['query']}")
                    st.markdown(f"**A:** {chat['response']}")
                    st.markdown("---")
    
    # Open Chat Tab (renamed from "Web Search")
    with tabs[6]:
        st.header("Open Chat")
        st.markdown("Chat openly about any topic using OpenAI's capabilities.")
        
        search_query = st.text_input("Enter your message")
        if st.button("Send"):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar settings.")
            elif not search_query:
                st.error("Please enter a message.")
            else:
                with st.spinner("Processing..."):
                    client = openai.Client(api_key=api_key)
                    search_result = web_search(client, model, search_query)
                    st.markdown("### Response")
                    st.markdown(search_result)
                    
                    # Store in search history
                    st.session_state.search_history.append({
                        "query": search_query,
                        "result": search_result
                    })
        
        # Chat History
        if st.session_state.search_history:
            with st.expander("Chat History", expanded=True):
                for search in reversed(st.session_state.search_history):
                    st.markdown(f"**You:** {search['query']}")
                    st.markdown(f"**Assistant:**")
                    st.markdown(search['result'])
                    st.markdown("---")

if __name__ == "__main__":
    main()
