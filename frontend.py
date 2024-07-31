import streamlit as st
import requests
def stream_response(url, json_data):
    response = requests.post(url, json=json_data, stream=True)
    if response.status_code == 200:
        # response_content = ""
        for chunk in response.iter_content(chunk_size=100):
            response_content = chunk.decode('utf-8')
            yield response_content
    else:
        yield "Error: Unable to get the answer. Please try again later."
def main():
    st.header("Chat with PDF 💬")
    query = st.text_input("Enter Your Query Here")
    
    if st.button("Submit"):
        if query:
            # st.write("Fetching the answer...")
            response_stream = stream_response("http://localhost:8001/query/", {"query": query})
            answer_placeholder = st.empty()
            answer = ""
            for chunk in response_stream:
                answer += chunk
                answer_placeholder.write(answer)
            if "send_email" in answer:
                st.write("Could not find the answer to the query in the context. An email has been sent to the team and they will look into it")
if __name__ == "__main__":
    main()
