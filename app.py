import tempfile
from PIL import Image
import os
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo

# Set the title and subtitle of the app
st.title('🦜🔗 Document_Chat: Chat with Your PDFs')
st.subheader('Input the PDF and questions pertaining to the document')

# Load the app logo
logo_image = Image.open('Document_Chat.png')
st.image(logo_image)

# Prompt user to upload a PDF file
st.subheader('Upload your PDF')
uploaded_file = st.file_uploader('', type=(['pdf', 'tsv', 'csv', 'txt', 'tab', 'xlsx', 'xls']))

# Set the default temporary file path
temp_file_path = os.getcwd()

# Wait for the user to upload a file
while uploaded_file is None:
    x = 1

# If a file is uploaded
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    st.write("Full path of the uploaded file:", temp_file_path)

# Set API key for OpenAI Service
# This can be substituted for other language model providers
os.environ['OPENAI_API_KEY'] = # Your OpenAI API Key

# Create an instance of OpenAI language model
language_model = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

# Create and load the PDF Loader
pdf_loader = PyPDFLoader(temp_file_path)
# Split pages from the PDF
pages = pdf_loader.load_and_split()

# Load documents into the vector database (ChromaDB)
vector_store = Chroma.from_documents(pages, embeddings, collection_name='Pdf')

# Create a vector store info object
vector_store_info = VectorStoreInfo(
    name="Pdf",
    description="A PDF file to answer your questions",
    vectorstore=vector_store
)

# Convert the document store into a langchain toolkit
vector_store_toolkit = VectorStoreToolkit(vectorstore_info=vector_store_info)

# Add the toolkit to an end-to-end language chain
agent_executor = create_vectorstore_agent(
    llm=language_model,
    toolkit=vector_store_toolkit,
    verbose=True
)

# Create a text input box for the user's prompt
user_prompt = st.text_input('Input your prompt here')

# If the user enters a prompt
if user_prompt:
    # Pass the prompt to the language model
    response = agent_executor.run(user_prompt)
    # Display the response
    st.write(response)

    # Use a streamlit expander for Document Similarity Search
    with st.expander('Document Similarity Search'):
        # Find relevant pages based on similarity
        similarity_search_results = vector_store.similarity_search_with_score(user_prompt) 
        # Display the content of the most relevant page
        st.write(similarity_search_results[0][0].page_content)
