import validators
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader



# Streamlit app
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website")
st.title("LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# Sidebar: API Key
import streamlit as st
Groq_api_key = st.secrets.get("GROQ_API_KEY", "")

# URL input
generic_url = st.text_input("URL", label_visibility="collapsed")

# Create LLM OUTSIDE the button (only if key exists)
llm = None
if Groq_api_key.strip():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=Groq_api_key,  
        temperature=0
    )

# Prompt template
prompt_template = """
You are a highly skilled summarization assistant.

Your task:
- Write a clear, accurate, and well-structured summary of the content below.
- The summary must be around 500 words.
- Focus on the main ideas, key arguments, important details, and essential insights.
- Do NOT add any information that is not present in the content.

Content to summarize:
{text}

Now provide the final ~500-word summary.
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Summarization button
if st.button("Summarize the content from YT or Website"):
    # Validate inputs
    if not Groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide both the Groq API key and a URL.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (Website or YouTube).")
    elif llm is None:
        st.error("Invalid API key. Please enter a valid Groq API key.")
    else:
        try:
            with st.spinner("Loading and summarizing..."):
                # Choose loader
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=False
                    )
                else:
                    headers = {
                        "User-Agent": "Mozilla/5.0"
                    }
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers=headers
                    )

                docs = loader.load()

                # Create summarization chain
                chain = load_summarize_chain(
                    llm,
                    chain_type="stuff",
                    prompt=prompt
                )

                # Run chain
                output_summary = chain.run(docs)
                st.success(output_summary)

        except Exception as e:
            st.exception(e)

