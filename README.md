# OpenAI RAG Chat With GitHub Topic

This prototype contains:

- `rendergit-topic.py`: Takes a GitHub Topic as a CLI argument and calls `rendergit` on each repo - turning each repo into a potentially large (>100MB) HTML file.
- `openai-file-search.py`: Creates a RAG index of all of these files using OpenAI. This can take a substantial amount of time for a large topic.
- `rag-chat-multi.py`: A familiar Streamlit chatbot UI. Automatically connects to an OpenAI file search (RAG) ID if available. Supports file uploads, specification of the API key and model selection. Has some quirks.
