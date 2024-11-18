import streamlit as st
import openai
from dotenv import load_dotenv
import os

load_dotenv()

AGENT_PROMPTS = {
    "company_analyst": """role: Company Analysis Specialist
goal: Analyze the target company to identify key characteristics for comparison
task: Conduct a comprehensive analysis of {target_company}. Focus on key characteristics such as industry, size, location, product offerings, target market, and any unique attributes. Consider how these characteristics might relate to their potential need for {our_product}.""",

    "market_researcher": """role: Market Research Expert
goal: Identify potential companies similar to the target company
task: Using the profile of {target_company}, search for and identify other companies that share similar characteristics. Look for companies in the same or adjacent industries, of similar size, with comparable product offerings or target markets. Aim to find at least 10 potential matches.""",

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

def run_agent(client, model, prompt, context=""):
    messages = [{"role": "system", "content": prompt}]
    if context:
        messages.append({"role": "user", "content": context})
    
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

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

def chat_with_results(api_key, model, query):
    if not st.session_state.analysis_results['final_report']:
        return "Please run analysis first."
    
    client = openai.Client(api_key=api_key)
    context = f"Analysis result: {st.session_state.analysis_results['final_report']}"
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant with expertise in analyzing companies and sales strategies. Use the provided analysis to answer questions."},
            {"role": "assistant", "content": context},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content

def main():
    st.title("Similar Company Finder")
    
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox("Model", ["gpt-4-turbo-preview", "gpt-3.5-turbo"])
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
    if st.session_state.analysis_results['final_report']:
        tabs = st.tabs([
            "Company Analysis",
            "Market Research",
            "Similarity Evaluation",
            "Sales Strategy",
            "Final Report",
            "Chat"
        ])
        
        # Company Analysis Tab
        with tabs[0]:
            st.header("Company Analysis Specialist")
            st.markdown("**Role**: Analyze target company characteristics")
            st.markdown("**Prompt**:")
            st.code(AGENT_PROMPTS["company_analyst"].format(
                target_company=company_name,
                our_product=product_desc
            ))
            st.markdown("**Results**:")
            st.markdown(st.session_state.analysis_results['company_profile'])
        
        # Market Research Tab
        with tabs[1]:
            st.header("Market Research Expert")
            st.markdown("**Role**: Identify similar companies")
            st.markdown("**Prompt**:")
            st.code(AGENT_PROMPTS["market_researcher"].format(
                target_company=company_name
            ))
            st.markdown("**Results**:")
            st.markdown(st.session_state.analysis_results['company_list'])
        
        # Similarity Evaluation Tab
        with tabs[2]:
            st.header("Company Similarity Evaluator")
            st.markdown("**Role**: Score company similarities")
            st.markdown("**Prompt**:")
            st.code(AGENT_PROMPTS["similarity_evaluator"].format(
                target_company=company_name
            ))
            st.markdown("**Results**:")
            st.markdown(st.session_state.analysis_results['evaluated_list'])
        
        # Sales Strategy Tab
        with tabs[3]:
            st.header("Sales Approach Strategist")
            st.markdown("**Role**: Develop sales recommendations")
            st.markdown("**Prompt**:")
            st.code(AGENT_PROMPTS["sales_strategist"].format(
                target_company=company_name,
                our_product=product_desc
            ))
            st.markdown("**Results**:")
            st.markdown(st.session_state.analysis_results['sales_strategy'])
        
        # Final Report Tab
        with tabs[4]:
            st.header("Final Report")
            st.markdown(st.session_state.analysis_results['final_report'])
        
        # Chat Tab
        with tabs[5]:
            st.header("Chat with Results")
            query = st.text_input("Ask a question about the analysis")
            if st.button("Submit"):
                response = chat_with_results(api_key, model, query)
                st.markdown("**Response:**")
                st.markdown(response)
                st.session_state.chat_history.append({"query": query, "response": response})
            
            with st.expander("Chat History"):
                for chat in st.session_state.chat_history:
                    st.markdown(f"**Q:** {chat['query']}")
                    st.markdown(f"**A:** {chat['response']}")

if __name__ == "__main__":
    main()