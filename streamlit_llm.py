import torch
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoModelForSeq2SeqLM,
    pipeline,
)
from langchain import PromptTemplate, HuggingFaceHub, LLMChain, HuggingFacePipeline

session_dict = {
    'generated': ['ChatBot in Streamlit, please send a prompt'],
    'past': ['Hello'],
    'model_loaded': False,
    'llm_chain': None,
}

for k, v in session_dict.items():
    if k not in st.session_state:
        st.session_state[k] = v


if st.session_state['model_loaded'] == False:
    # model_name = 'EleutherAI/pythia-2.8b'
    # model_name = 'meta-llama/Llama-2-7b-chat-hf'
    # model_name = 'tiiuae/falcon-7b'
    model_name = 'google/flan-t5-base'

    template = """
    Answer the question below in a very objective and polite way: \
    Question: {question} \
    """
    
    prompt = PromptTemplate(template=template, input_variables=["question"])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bits=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    
    device_map = {"": 0}
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    # model.to_bettertransformer()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    pipe = pipeline(
    'text2text-generation',
    model=model,
    tokenizer=tokenizer,
    max_length=100,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    llm_chain = LLMChain(prompt=prompt,
                     llm=local_llm)
    
    st.session_state['model_loaded'] = True
    st.session_state['llm_chain'] = llm_chain
    

# streamlit app
st.set_page_config(page_title="HugChat - An LLM-powered Streamlit app")

with st.sidebar:
    st.title('🤗💬 HugChat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](<https://streamlit.io/>)
    - [HugChat](<https://github.com/Soulter/hugging-chat-api>)
    - [OpenAssistant/oasst-sft-6-llama-30b-xor](<https://huggingface.co/OpenAssistant/oasst-sft-6-llama-30b-xor>) LLM model
    
    💡 Note: No API key required!
    ''')
    add_vertical_space(5)
    

input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

with input_container:
    user_input = st.text_input('User: ', "", key='input')

llm_chain = st.session_state['llm_chain']


def generate_response(prompt):

    text_return = llm_chain.run(prompt)

    # input_ids = tokenizer(prompt, truncation=True, max_length=1000, return_tensors='pt').to(device)
    # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    #     output_ids = model.generate(**input_ids, max_new_tokens=100,) # stop_sequences=["\nUser:", "<|endoftext|>"])
    # text_return = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return text_return

with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(response)
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i)+'_user')
            message(st.session_state['generated'][i], key=str(i))