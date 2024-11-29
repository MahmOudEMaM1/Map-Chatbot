import streamlit as st
import requests
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama

# Set page configuration
st.set_page_config(page_title="Llama 3.2 Chat", page_icon="🦙", layout="wide")

# Main chat interface
st.title("Place Info Chatbot 🦙")

with st.sidebar:
    st.header("Inference Settings")
    
    model = st.selectbox("Model", ["llama3.2:1b"], index=0)

    seed = st.slider("Seed", min_value=1, max_value=9007199254740991, value=1, step=1)
    temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )
    max_tokens = st.slider(
        "Max Tokens", min_value=100, max_value=128000, value=5000, step=100
    )

    st.session_state.model = model
    st.session_state.seed = seed
    st.session_state.temperature = temperature
    st.session_state.max_tokens = max_tokens

chat = ChatOllama(
    model=st.session_state.model,
    seed=st.session_state.seed,
    temperature=st.session_state.temperature,
    max_tokens=st.session_state.max_tokens,
)

msgs = StreamlitChatMessageHistory(key="special_app_key")

if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

chain = prompt | chat

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,  # Always return the instance created earlier
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Function to get coordinates from the HERE API
def get_here_coordinates(place_name):
    api_key = "8dcGBy0K1LCzTwyc0OWq7yNCR-SO69rcqFrccB1A2f4"
    url = f"https://geocode.search.hereapi.com/v1/geocode?q={place_name}&apiKey={api_key}"
    response = requests.get(url)

    if response.status_code == 200 and response.json()['items']:
        lat_lng = response.json()['items'][0]['position']
        return {'lat': lat_lng['lat'], 'lng': lat_lng['lng']}
    return None

# Function to generate map link
def get_here_map_link(lat_lng):
    return f"https://wego.here.com/directions/mix//{lat_lng['lat']},{lat_lng['lng']}"

# Generate detailed information about the place using Llama
def generate_place_description(place_name):
    place_description_prompt = f"Please generate a detailed description (100 words) about the following place: {place_name}. Include historical, cultural, or geographical details."

    # Adjust the prompt for the language model to generate the text
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"input": place_description_prompt}, config)

    return response.content

# Main chat logic
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input("Type your message here..."):
    st.chat_message("human").write(prompt)

    with st.spinner("Thinking..."):
        # Check if the input is a place name and generate a detailed description
        lat_lng = get_here_coordinates(prompt)

        if lat_lng:
            # Generate a detailed description of the place
            place_description = generate_place_description(prompt)

            # Generate the map link
            map_link = get_here_map_link(lat_lng)
            
            # Combine place description and map link
            response_content = f"{place_description}\nHere's the location: {map_link}"
        else:
            # Otherwise, process as a normal chatbot response
            config = {"configurable": {"session_id": "any"}}
            response = chain_with_history.invoke({"input": prompt}, config)
            response_content = response.content

        # Display the response
        st.chat_message("ai").write(response_content)

# Add a button to clear chat history
if st.button("Clear Chat History"):
    msgs.clear()
    st.rerun()
