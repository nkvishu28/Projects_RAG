# Projects_RAG
Mini project on RAG using Open source LLM hosted on GROQ cloud platform - Chatbot with PDF files.

This project is to establish a conversational chat with the PDF files. The PDF files will be providing the context for the user query.

This is developed using the langchain framework, open source LLM - llama3-groq-70b-8192-tool-use-preview (this can be changed based on the evaluation).

The solution involves the below steps,

1. Data ingestion - PDF files.
2. Splitting the PDF files into chunks and convert them into the embeddings using open source embeddings offered by Huggingface.
3. Store the embeddings into the vector database - Chroma.
4. Define chat template as offered by Groq.
5. Select the Open source model offered by Groq - https://groq.com/
6. Define the chain with retriever.
7. Take the input from the user - question/query from the end user.
8. Generate response and maintain the chat history using the ChatMessage History provided by Langchain.
9. Streamlit is used to provide a simple web application.
10. Langsmith is used to monitor the application

Special thanks to Krish Naik for providing a detailed session on developing this.
