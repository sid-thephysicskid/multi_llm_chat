import streamlit as st
from openai import OpenAI
import anthropic
import google.generativeai as genai
import os
from dotenv import load_dotenv
import asyncio
import fitz  # PyMuPDF

load_dotenv()


openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) if os.getenv("ANTHROPIC_API_KEY") else None
if os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-pro')
else:
    gemini_model = None

async def stream_openai_response(prompt, response_container):
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True
        # max_tokens=100000
    )
    full_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
            response_container.markdown(full_response)
        await asyncio.sleep(0)

async def stream_anthropic_response(prompt, response_container):
    with anthropic_client.messages.stream(
        # model="claude-3-opus-20240229",
        model='claude-3-5-sonnet-20240620',
        max_tokens=2000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    ) as stream:
        full_response = ""
        for text in stream.text_stream:
            full_response += text
            response_container.markdown(full_response)
            await asyncio.sleep(0)

async def stream_gemini_response(prompt, response_container):
    response = gemini_model.generate_content(prompt, stream=True)
    full_response = ""
    for chunk in response:
        full_response += chunk.text
        response_container.markdown(full_response)
        await asyncio.sleep(0)

def extract_text_from_pdf(pdf_file):
    try:
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

async def process_responses(prompt, openai_container, anthropic_container, gemini_container):
    tasks = []
    if openai_container and openai_client:
        tasks.append(stream_openai_response(prompt, openai_container))
    if anthropic_container and anthropic_client:
        tasks.append(stream_anthropic_response(prompt, anthropic_container))
    if gemini_container and gemini_model:
        tasks.append(stream_gemini_response(prompt, gemini_container))
    await asyncio.gather(*tasks)

st.set_page_config(layout="wide")

st.title("Multi-Model Chat and PDF Analyzer")

# Create two columns for the main layout
left_column, right_column = st.columns([1, 3])

with left_column:
    st.subheader("Select Models")
    use_openai = st.checkbox("OpenAI GPT-3.5", value=True, disabled=not openai_client)
    use_anthropic = st.checkbox("Anthropic Claude", value=True, disabled=not anthropic_client)
    use_gemini = st.checkbox("Google Gemini", value=True, disabled=not gemini_model)

    uploaded_file = st.file_uploader("Upload a PDF file (optional)", type="pdf")

with right_column:
    user_input = st.text_area("Enter your prompt or question about the PDF:", height=100)
    
    if st.button("Send", use_container_width=True):
        if user_input:
            # prepare prompt
            if uploaded_file:
                pdf_text = extract_text_from_pdf(uploaded_file)
                if pdf_text:
                    # Use a larger chunk of the PDF content, or consider summarizing it
                    prompt = f"PDF Content:\n\n{pdf_text[:10000]}\n\nUser Question: {user_input}"
                    st.info(f"PDF content length: {len(pdf_text)} characters")
                else:
                    prompt = f"Error occurred while processing the PDF. User Question: {user_input}"
            else:
                prompt = user_input

            #  response columns
            openai_col, anthropic_col, gemini_col = st.columns(3)

            with openai_col:
                if use_openai and openai_client:
                    st.subheader("OpenAI Response")
                    openai_response = st.empty()
                else:
                    openai_response = None

            with anthropic_col:
                if use_anthropic and anthropic_client:
                    st.subheader("Anthropic Response")
                    anthropic_response = st.empty()
                else:
                    anthropic_response = None

            with gemini_col:
                if use_gemini and gemini_model:
                    st.subheader("Gemini Response")
                    gemini_response = st.empty()
                else:
                    gemini_response = None

            # responses
            asyncio.run(process_responses(prompt, openai_response, anthropic_response, gemini_response))
        else:
            st.warning("Please enter a prompt or question.")

if __name__ == "__main__":
    pass
