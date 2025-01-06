import streamlit as st
from openai import OpenAI
import anthropic
import os
from dotenv import load_dotenv
import asyncio
import fitz  # PyMuPDF
from prompt_manager import PromptManager
from config import DEFAULT_SYSTEM_PROMPTS, MODEL_CONFIGS, MODEL_PROVIDERS

load_dotenv()

MAX_CONCURRENT_MODELS = 4

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) if os.getenv("ANTHROPIC_API_KEY") else None
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
) if os.getenv("NVIDIA_API_KEY") else None

# Initialize prompt manager
prompt_manager = PromptManager()

async def stream_openai_response(model_id, prompt, system_prompt, response_container):
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        response = openai_client.chat.completions.create(
            model=MODEL_CONFIGS[model_id]["model"],
            messages=messages,
            stream=True
        )
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                response_container.markdown(full_response)
            await asyncio.sleep(0)
    except Exception as e:
        response_container.error(f"Error from {MODEL_CONFIGS[model_id]['name']}: {str(e)}")

async def stream_anthropic_response(model_id, prompt, system_prompt, response_container):
    try:
        with anthropic_client.messages.stream(
            model=MODEL_CONFIGS[model_id]["model"],
            max_tokens=2000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        ) as stream:
            full_response = ""
            for text in stream.text_stream:
                full_response += text
                response_container.markdown(full_response)
                await asyncio.sleep(0)
    except Exception as e:
        response_container.error(f"Error from {MODEL_CONFIGS[model_id]['name']}: {str(e)}")

async def stream_nvidia_response(model_id, prompt, system_prompt, response_container):
    try:
        # For NVIDIA API, combine system prompt and user prompt
        combined_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        messages = [
            {"role": "user", "content": combined_prompt}
        ]
        response = nvidia_client.chat.completions.create(
            model=MODEL_CONFIGS[model_id]["model"],
            messages=messages,
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
            stream=True
        )
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                response_container.markdown(full_response)
            await asyncio.sleep(0)
    except Exception as e:
        response_container.error(f"Error from {MODEL_CONFIGS[model_id]['name']}: {str(e)}")

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

async def process_responses(prompt, system_prompts, selected_models, containers):
    tasks = []
    for model_id in selected_models:
        if not containers[model_id]:
            continue
            
        provider = MODEL_CONFIGS[model_id]["provider"]
        if provider == "openai" and openai_client:
            task = asyncio.create_task(
                stream_openai_response(model_id, prompt, system_prompts[model_id], containers[model_id])
            )
        elif provider == "anthropic" and anthropic_client:
            task = asyncio.create_task(
                stream_anthropic_response(model_id, prompt, system_prompts[model_id], containers[model_id])
            )
        elif provider == "nvidia" and nvidia_client:
            task = asyncio.create_task(
                stream_nvidia_response(model_id, prompt, system_prompts[model_id], containers[model_id])
            )
        else:
            continue
            
        tasks.append(task)
    
    # Wait for all tasks to complete, but don't let one failure stop others
    if tasks:
        await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

def initialize_session_state():
    if "system_prompts" not in st.session_state:
        st.session_state.system_prompts = DEFAULT_SYSTEM_PROMPTS.copy()
    if "use_same_prompt" not in st.session_state:
        st.session_state.use_same_prompt = True
    if "saved_prompts" not in st.session_state:
        st.session_state.saved_prompts = prompt_manager.list_prompts()
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []

def main():
    st.set_page_config(layout="wide")
    initialize_session_state()

    st.title("Multi-Model Chat and PDF Analyzer")
    
    # Create two columns for the main layout
    left_column, right_column = st.columns([1, 3])

    with left_column:
        st.subheader("Model Selection")
        st.caption(f"Select up to {MAX_CONCURRENT_MODELS} models")
        
        # Model selection by provider
        for provider, info in MODEL_PROVIDERS.items():
            with st.expander(f"{info['name']} Models"):
                for model_id in info['models']:
                    config = MODEL_CONFIGS[model_id]
                    # Determine if the model can be selected
                    client_available = (
                        (provider == "openai" and openai_client) or
                        (provider == "anthropic" and anthropic_client) or
                        (provider == "nvidia" and nvidia_client)
                    )
                    can_select = (
                        client_available and
                        (model_id in st.session_state.selected_models or
                         len(st.session_state.selected_models) < MAX_CONCURRENT_MODELS)
                    )
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if st.checkbox(
                            config["name"],
                            value=model_id in st.session_state.selected_models,
                            disabled=not can_select,
                            key=f"model_{model_id}"
                        ):
                            if model_id not in st.session_state.selected_models:
                                st.session_state.selected_models.append(model_id)
                        else:
                            if model_id in st.session_state.selected_models:
                                st.session_state.selected_models.remove(model_id)
                    with col2:
                        st.caption(config["description"])

        st.subheader("System Prompts")
        use_same_prompt = st.checkbox("Use same prompt for all models", value=st.session_state.use_same_prompt)
        
        if use_same_prompt:
            shared_prompt = st.text_area(
                "System Prompt (all models)",
                value=st.session_state.system_prompts[list(MODEL_CONFIGS.keys())[0]],
                height=100
            )
            for model_id in MODEL_CONFIGS:
                st.session_state.system_prompts[model_id] = shared_prompt
        else:
            with st.expander("Configure Individual Prompts"):
                for model_id in st.session_state.selected_models:
                    config = MODEL_CONFIGS[model_id]
                    st.session_state.system_prompts[model_id] = st.text_area(
                        f"{config['name']} Prompt",
                        value=st.session_state.system_prompts[model_id],
                        height=100
                    )

        st.subheader("Prompt Management")
        saved_prompts = prompt_manager.list_prompts()
        if saved_prompts:
            selected_prompt = st.selectbox("Load Saved Prompt", options=list(saved_prompts.keys()))
            if st.button("Load"):
                loaded_prompt = prompt_manager.get_prompt(selected_prompt)
                if use_same_prompt:
                    for model_id in MODEL_CONFIGS:
                        st.session_state.system_prompts[model_id] = loaded_prompt
                else:
                    st.session_state.system_prompts[list(MODEL_CONFIGS.keys())[0]] = loaded_prompt

        prompt_name = st.text_input("Save current prompt as:")
        if st.button("Save Prompt"):
            if prompt_name:
                prompt_to_save = (st.session_state.system_prompts[list(MODEL_CONFIGS.keys())[0]]
                                if use_same_prompt 
                                else st.session_state.system_prompts)
                prompt_manager.save_prompt(prompt_name, prompt_to_save)
                st.success(f"Prompt saved as '{prompt_name}'")

        uploaded_file = st.file_uploader("Upload a PDF file (optional)", type="pdf")

    with right_column:
        if not st.session_state.selected_models:
            st.warning("Please select at least 1 model from the left sidebar.")
        else:
            user_input = st.text_area("Enter your prompt or question:", height=100)
            
            if st.button("Send", use_container_width=True):
                if user_input:
                    if uploaded_file:
                        pdf_text = extract_text_from_pdf(uploaded_file)
                        if pdf_text:
                            prompt = f"PDF Content:\n\n{pdf_text[:10000]}\n\nUser Question: {user_input}"
                            st.info(f"PDF content length: {len(pdf_text)} characters")
                        else:
                            prompt = f"Error occurred while processing the PDF. User Question: {user_input}"
                    else:
                        prompt = user_input

                    # Create response columns based on selected models
                    num_models = len(st.session_state.selected_models)
                    if num_models > 0:
                        cols = st.columns(min(num_models, MAX_CONCURRENT_MODELS))
                        
                        # Initialize containers for all possible models
                        response_containers = {model_id: None for model_id in MODEL_CONFIGS}
                        
                        # Set up containers for selected models
                        for i, model_id in enumerate(st.session_state.selected_models):
                            with cols[i]:
                                st.subheader(f"{MODEL_CONFIGS[model_id]['name']}")
                                response_containers[model_id] = st.empty()

                        # Process responses
                        asyncio.run(process_responses(
                            prompt,
                            st.session_state.system_prompts,
                            st.session_state.selected_models,
                            response_containers
                        ))
                else:
                    st.warning("Please enter a prompt or question.")

if __name__ == "__main__":
    main()
