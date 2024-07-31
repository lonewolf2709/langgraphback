import streamlit as st
import requests
import json

API_URL = "http://localhost:8000/query/"

def invoke_workflow(query):
    response = requests.post(API_URL, json={"query": query}, stream=True)
    # if response.status_code == 200:
    #     for chunk in response.iter_content(chunk_size=100):
    #         response_content = chunk.decode('utf-8')
    #         yield response_content
    # else:
    #     yield "Error: Unable to get the answer. Please try again later."
    return response

# Initialize session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define CSS for layout and styling
css = """
<style>
body {
    display: flex;
    flex-direction: column;
    height: 100vh;
    margin: 0;
}

.sidebar {
    width: 300px;
    position: fixed;
    top: 0;
    left: 0;
    height: 100%;
    background-color: #f5f5f5;
    padding: 20px;
    border-right: 1px solid #ddd;
    overflow-y: auto;
}

.main-content {
    margin-left: 10px;
    display: flex;
    flex-direction: column;
    height: 90%;
    align-items:flex-start;
}

.header {
    padding: 20px;
}

.chat-history {
    overflow-y: auto;
    flex: 1;
    padding: 20px;
    background-color: #f5f5f5;
    margin-bottom: 60px; /* Make space for input area */
}

.st-emotion-cache-4uzi61 {
    position: fixed;
    bottom: 0;
    left: 400px;
    width: 60%;
    padding:10px;
    padding-top:30px;
    display: flex;
    align-items: center;
    flex-direction:row;
    left:460px;
    padding-bottom:20px;
    border:none;
    z-index:1;
}
.st-emotion-cache-18cjxy2{
   display:flex;
   flex-direction:row;
   gap:10px;
   width:100%;
}
.st-emotion-cache-1wmy9hl{
    width:100%;
    display:flex;
    flex-direction:row;
}
.st-emotion-cache-e2y4gp {
    width: 100%;
    display: flex;
    flex-direction: row;
    gap: 1rem;
}
.user-message {
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
    width: fit-content;
    max-width: 100%;
    align-self: flex-start;
}

.agent-message {
    background-color: #F1F0F0;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
    width: fit-content;
    max-width: 100%;
    align-self: flex-end;
}

input[type="text"] {
    flex: 1;
    padding: 10px;
    margin-right: 10px;
}

button {
    padding: 10px 20px;
    border: none;
    background-color: #007bff;
    color: white;
    border-radius: 5px;
    cursor: pointer;
}
button:hover {
    background-color: #0056b3;
}
.st-emotion-cache-bm2z3a{
    display: flex;
    flex-direction: column;
    width: 100%;
    overflow: auto;
    top:100px;
    margin-left:100px;
    -webkit-box-align: center;
    align-items: flex-start;
    height:80%;
}
.st-emotion-cache-13ln4jf{
    padding:2rem 0rem 2rem
}
.st-emotion-cache-v7xzqe{
    width:100%;
    display:flex;
    flex-direction: column;
    gap: 1rem;
}
.st-emotion-cache-0{
   width:100%;
}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("History")
    st.write("Your past interactions:")

# Main content
st.markdown("<div class='main-content'>", unsafe_allow_html=True)
# Header
st.markdown("<div class='header'><h1>Conversational Agent</h1></div>", unsafe_allow_html=True)

# Chat history section
chat_history_container = st.empty()
with chat_history_container.container():
    for msg in st.session_state.messages:
        if msg["role"] == "human":
            st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="agent-message">{msg["content"]}</div>', unsafe_allow_html=True)

# Input form at the bottom
with st.form(key="message_form", clear_on_submit=True):
    user_input = st.text_input("Your Message", key="input")
    submit_button = st.form_submit_button(label="Send")
    if submit_button and user_input:
        # Add user message to the history
        st.session_state.messages.append({"role": "human", "content": user_input})
        # Placeholder for agent message
        agent_message = ""
        st.session_state.messages.append({"role": "agent", "content": agent_message})
        # Invoke the API and stream the response
        for chunk in invoke_workflow(user_input):
            agent_message += chunk
            st.session_state.messages[-1]["content"] = agent_message
            # Refresh the display with updated conversation history
            with chat_history_container.container():
                for msg in st.session_state.messages:
                    if msg["role"] == "human":
                        st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="agent-message">{msg["content"]}</div>', unsafe_allow_html=True)
            # JavaScript to scroll to the bottom of the chat history
            # st.markdown("""
            #     <script>
            #     var chat_history = window.parent.document.querySelector('.chat-history');
            #     chat_history.scrollTop = chat_history.scrollHeight;
            #     </script>
            # """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
