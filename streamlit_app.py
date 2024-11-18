import streamlit as st
import openai
from dotenv import load_dotenv
import os
import requests
import json

# Load environment variables
load_dotenv()
DEFAULT_OPENAI_KEY = os.getenv('OPENAI_API_KEY', '')
DEFAULT_SERPER_KEY = os.getenv('SERPER_API_KEY', '')

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
task: Using the provided search results about {target_company}, conduct a comprehensive analysis. Focus on key characteristics such as industry, size, location, product offerings, target market, and any unique attributes. Consider how these characteristics might relate to their potential need for {our_product}.

Search Results:
{search_results}""",

    "market_researcher": """role: Market Research Expert
goal: Identify potential companies similar to the target company
task: Using the profile of {target_company} and the search results for similar companies, identify and analyze companies that share similar characteristics. Look for companies in the same or adjacent industries, of similar size, with comparable product offerings or target markets. Aim to find at least 10 potential matches.

Company Profile:
{company_profile}

Search Results for Similar Companies:
{search_results}""",

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

def serper_search(api_key, query, num_results=10):
    """
    Perform a search using the Serper API
    """
    url = "https://google.serper.dev/search"
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    payload = {
        'q': query,
        'num': num_results
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return f"Error performing search: {str(e)}"

def format_serper_results(results):
    """
    Format Serper API results into a readable string
    """
    if isinstance(results, str):  # If there was an error
        return results
    
    formatted = []
    if 'organic' in results:
        for item in results['organic']:
            formatted.append(f"Title: {item.get('title', 'N/A')}")
            formatted.append(f"Snippet: {item.get('snippet', 'N/A')}")
            formatted.append(f"Link: {item.get('link', 'N/A')}")
            formatted.append("---")
    
    return "\n".join(formatted)

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

def run_analysis(openai_key, serper_key, model, company_name, product_desc):
    client = openai.Client(api_key=openai_key)
    results = {}
    
    # Company Analysis Specialist with Serper search
    company_search_results = serper_search(serper_key, f"{company_name} company information business details")
    formatted_company_results = format_serper_results(company_search_results)
    
    analyst_prompt = AGENT_PROMPTS["company_analyst"].format(
        target_company=company_name,
        our_product=product_desc,
        search_results=formatted_company_results
    )
    results['company_profile'] = run_agent(client, model, analyst_prompt)
    
    # Market Research Expert with Serper search
    similar_companies_query = f"companies similar to {company_name} same industry competitors"
    similar_companies_results = serper_search(serper_key, similar_companies_query)
    formatted_similar_results = format_serper_results(similar_companies_results)
    
    researcher_prompt = AGENT_PROMPTS["market_researcher"].format(
        target_company=company_name,
        company_profile=results['company_profile'],
        search_results=formatted_similar_results
    )
    results['company_list'] = run_agent(client, model, researcher_prompt)
    
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

def open_chat(api_key, model, query):
    """
    Open chat functionality using OpenAI
    """
    client = openai.Client(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant ready to discuss any topic."},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in chat: {str(e)}"

def main():
    st.title("Similar Company Finder")
    
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        openai_key = st.text_input("OpenAI API Key", value=DEFAULT_OPENAI_KEY, type="password")
        serper_key = st.text_input("Serper API Key", value=DEFAULT_SERPER_KEY, type="password")
        model = st.selectbox("Model", OPENAI_MODELS)
        company_name = st.text_input("Company Name")
        product_desc = st.text_area("Product Description")
        
        if st.button("Run Analysis"):
            if not all([openai_key, serper_key, company_name, product_desc]):
                st.error("Please fill in all fields")
                return
                
            with st.spinner("Running analysis..."):
                results = run_analysis(openai_key, serper_key, model, company_name, product_desc)
                st.success("Analysis complete!")
    
    # Main content area with tabs
    tabs = st.tabs([
        "Company Analysis",
        "Market Research",
        "Similarity Evaluation",
        "Sales Strategy",
        "Final Report",
        "Chat with Results",
        "Open Chat"
    ])
    
    if st.session_state.analysis_results['final_report']:
        # Company Analysis Tab
        with tabs[0]:
            st.header("Company Analysis Specialist")
            st.markdown("**Role**: Analyze target company characteristics")
            st.markdown(st.session_state.analysis_results['company_profile'])
        
        # Market Research Tab
        with tabs[1]:
            st.header("Market Research Expert")
            st.markdown("**Role**: Identify similar companies")
            st.markdown(st.session_state.analysis_results['company_list'])
        
        # Similarity Evaluation Tab
        with tabs[2]:
            st.header("Company Similarity Evaluator")
            st.markdown("**Role**: Score company similarities")
            st.markdown(st.session_state.analysis_results['evaluated_list'])
        
        # Sales Strategy Tab
        with tabs[3]:
            st.header("Sales Approach Strategist")
            st.markdown("**Role**: Develop sales recommendations")
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
            st.markdown(get_chat_context(selected_context))
            
            # Chat interface
            query = st.text_input("Ask a question about the analysis")
            if st.button("Submit"):
                response = chat_with_results(openai_key, model, query, selected_context)
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
    
    # Open Chat Tab
    with tabs[6]:
        st.header("Open Chat")
        st.markdown("Chat openly about any topic using OpenAI's capabilities.")
        
        chat_query = st.text_input("Enter your message")
        if st.button("Send"):
            if not openai_key:
                st.error("Please enter your OpenAI API key in the sidebar settings.")
            elif not chat_query:
                st.error("Please enter a message.")
            else:
                with st.spinner("Processing..."):
                    chat_result = open_chat(openai_key, model, chat_query)
                    st.markdown(chat_result)
                    
                    # Store in search history
                    st.session_state.search_history.append({
                        "query": chat_query,
                        "result": chat_result
                    })
        
        # Chat History
        if st.session_state.search_history:
            with st.expander("Chat History", expanded=True):
                for chat in reversed(st.session_state.search_history):
                    st.markdown(f"**You:** {chat['query']}")
                    st.markdown(f"**Assistant:** {chat['result']}")
                    st.markdown("---")

if __name__ == "__main__":
    main()
