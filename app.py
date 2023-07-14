import streamlit as st
from bot import ChatBot, AIFactory

llm = AIFactory.llm_models('openai')
embedding = AIFactory.embeddings('openai')


# @st.cache_resource
def build_chatbot(url, llm, embedding, pattern, chunk_size=2000, chunk_overlap=500):
    return ChatBot(site_url=url,
                   llm=llm, embedding=embedding,
                   chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                   url_pattern=pattern)


st.set_page_config(page_title="Ask Me Anything", page_icon="🐍")
st.title("Ask Me Anything")

# Remove whitespace from the top of the page and sidebar
st.markdown("""<style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>""", unsafe_allow_html=True)

st.markdown("## Config")
url_col, filter_col, platform_col = st.columns(3)
with url_col:
    site_url = st.text_input("URL to the website", value="")
with filter_col:
    pattern = st.text_input("URL filter pattern (optional)", value="")

st.markdown("## Ask")
if site_url and pattern:
    with st.spinner("Getting the knowledge base ready ..."):
        cb = build_chatbot(site_url, llm, embedding, pattern)
    question = st.text_input("Question", value="")
    if question:
        with st.spinner("Getting the answer ..."):
            result = cb.ask(question)
        st.markdown("### Answer")
        st.markdown(result["answer"])
        st.markdown("### Sources")
        st.markdown("\n ".join([f"- {x}" for x in result["sources"].split("\n")]))
