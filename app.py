import streamlit as st
import os
import tempfile
import google.generativeai as genai
from pypdf import PdfReader
from pinecone import Pinecone
import uuid
import time
import json
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

# Configuration
st.set_page_config(page_title="PDF Learning Assistant", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_pdf_content" not in st.session_state:
    st.session_state.current_pdf_content = ""
if "current_pdf_name" not in st.session_state:
    st.session_state.current_pdf_name = ""
if "index_name" not in st.session_state:
    st.session_state.index_name = "index1"  # Using your specific index name
if "is_initialized" not in st.session_state:
    st.session_state.is_initialized = False
if "index_dimensions" not in st.session_state:
    st.session_state.index_dimensions = 1024  # Set this based on your Pinecone index

# Functions for PDF processing
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length and end - start == chunk_size:
            # Find the last period or newline to make more natural chunks
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            if last_period > start + chunk_size // 2:
                end = last_period + 1
            elif last_newline > start + chunk_size // 2:
                end = last_newline + 1
        
        chunks.append(text[start:end])
        start = end - overlap if end < text_length else text_length
    
    return chunks

# Embeddings and Vector Store functions
@st.cache_resource
def get_embedding_model():
    # Using a model that produces 1024-dimensional embeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-roberta-large-v1")


def initialize_pinecone():
    # Get API key from environment variables
    api_key = os.getenv("PINECONE_API_KEY")
    
    if not api_key:
        st.error("Pinecone API key not found. Please add it to your .env file as PINECONE_API_KEY=your_api_key")
        return False
    
    try:
        # Initialize Pinecone with your specific configuration
        pc = Pinecone(api_key=api_key)
        
        # Store Pinecone client in session state
        st.session_state.pinecone_client = pc
        
        # Check if your index exists
        index_list = [idx.name for idx in pc.list_indexes()]
        if st.session_state.index_name not in index_list:
            st.error(f"Index '{st.session_state.index_name}' not found in your Pinecone account.")
            st.info("Available indexes: " + ", ".join(index_list))
            return False
        
        # Get index details to check dimensions
        try:
            index = pc.Index(st.session_state.index_name)
            index_stats = index.describe_index_stats()
            if 'dimension' in index_stats:
                st.session_state.index_dimensions = index_stats['dimension']
                st.info(f"Detected index dimension: {st.session_state.index_dimensions}")
            else:
                st.warning("Could not detect index dimensions. Using default: 1024")
        except Exception as e:
            st.warning(f"Could not get index details: {str(e)}")
        
        return True
    except Exception as e:
        st.error(f"Error initializing Pinecone: {str(e)}")
        return False

def get_pinecone_index():
    # Connect to your existing index
    return st.session_state.pinecone_client.Index(st.session_state.index_name)

def embed_chunks(chunks):
    model = get_embedding_model()
    embeddings = []
    for chunk in chunks:
        # HuggingFaceEmbeddings returns a list with a single embedding
        embed = model.embed_documents([chunk])[0]
        embeddings.append(embed)
    return embeddings

def store_embeddings(chunks, embeddings, pdf_name):
    index = get_pinecone_index()
    batch_size = 100
    
    for i in range(0, len(chunks), batch_size):
        i_end = min(i + batch_size, len(chunks))
        ids = [f"{pdf_name}-{uuid.uuid4()}" for _ in range(i, i_end)]
        metadata = [{"text": chunks[j], "pdf_name": pdf_name, "chunk_id": j} for j in range(i, i_end)]
        vectors = [(ids[j-i], embeddings[j], metadata[j-i]) for j in range(i, i_end)]
        
        try:
            index.upsert(vectors=vectors)
            st.success(f"Successfully stored batch {i//batch_size + 1} of chunks to Pinecone")
        except Exception as e:
            st.error(f"Error storing embeddings: {str(e)}")
            # Display the first embedding's dimension for debugging
            if embeddings and len(embeddings) > 0:
                st.info(f"Embedding dimension: {len(embeddings[0])}")
            return False
    
    st.success(f"Successfully stored all {len(chunks)} chunks to Pinecone")
    return True

def search_similar_chunks(query, top_k=5, pdf_name=None):
    model = get_embedding_model()
    query_embedding = model.embed_query(query)
    index = get_pinecone_index()
    
    filter_query = {"pdf_name": pdf_name} if pdf_name else None
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_query
    )
    
    return results.matches

# Gemini LLM Integration
@st.cache_resource
def initialize_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("Google API key not found. Please add it to your .env file as GOOGLE_API_KEY=your_api_key")
        return False
    
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error initializing Google Generative AI: {str(e)}")
        return False

def get_gemini_response(prompt, context=None, temperature=0.7):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        if context:
            full_prompt = f"""
            Context information: 
            {context}
            
            Question: {prompt}
            
            Please provide a helpful, accurate response based on the context information provided. 
            If the answer cannot be determined from the context, please state that clearly.
            """
        else:
            full_prompt = prompt
            
        response = model.generate_content(full_prompt, generation_config={"temperature": temperature})
        return response.text
    except Exception as e:
        st.error(f"Error getting response from Gemini: {str(e)}")
        return "Sorry, I couldn't generate a response at this time."

# Quiz and Assignment Generation
def generate_quiz(pdf_content, num_questions=5):
    prompt = f"""
    Based on the following content, generate a quiz with {num_questions} multiple-choice questions. 
    For each question, provide 4 options and indicate the correct answer.
    Format the response as a JSON array of question objects with the structure:
    [
        {{
            "question": "Question text", 
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": "Correct option (A, B, C, or D)",
            "explanation": "Brief explanation of why this is the correct answer"
        }},
        // more questions...
    ]
    
    Content: {pdf_content[:2000]}... (truncated for brevity)
    """
    
    response = get_gemini_response(prompt, temperature=0.2)
    
    try:
        # Extract JSON from response if it's embedded in markdown or text
        if "```json" in response:
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            json_str = response[json_start:json_end].strip()
        elif "```" in response:
            json_start = response.find("```") + 3
            json_end = response.find("```", json_start)
            json_str = response[json_start:json_end].strip()
        else:
            json_str = response
            
        quiz_data = json.loads(json_str)
        return quiz_data
    except Exception as e:
        st.error(f"Error parsing quiz response: {str(e)}")
        return []

def generate_assignment(pdf_content, assignment_type="short_answer", num_questions=3):
    prompt = f"""
    Based on the following content, generate a {assignment_type} assignment with {num_questions} questions.
    If the assignment type is 'short_answer', create questions that require brief explanations.
    If the assignment type is 'essay', create deeper questions that require longer responses.
    If the assignment type is 'research', create questions that encourage further exploration of the topics.
    
    Format the response as a JSON array with the structure:
    [
        {{
            "question": "Question text",
            "hints": ["Hint 1", "Hint 2"],
            "key_points": ["Key point 1", "Key point 2", "Key point 3"]
        }},
        // more questions...
    ]
    
    Content: {pdf_content[:2000]}... (truncated for brevity)
    """
    
    response = get_gemini_response(prompt, temperature=0.3)
    
    try:
        # Extract JSON from response if it's embedded in markdown or text
        if "```json" in response:
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            json_str = response[json_start:json_end].strip()
        elif "```" in response:
            json_start = response.find("```") + 3
            json_end = response.find("```", json_start)
            json_str = response[json_start:json_end].strip()
        else:
            json_str = response
            
        assignment_data = json.loads(json_str)
        return assignment_data
    except Exception as e:
        st.error(f"Error parsing assignment response: {str(e)}")
        return []

# Streamlit UI
def main():
    st.title("📚 PDF Learning Assistant")
    
    # Initialize services
    if not st.session_state.is_initialized:
        with st.spinner("Initializing services..."):
            pinecone_init = initialize_pinecone()
            gemini_init = initialize_gemini()
            
            if pinecone_init and gemini_init:
                st.session_state.is_initialized = True
                st.success("Services initialized successfully!")
                
                # Display Pinecone connection info
                st.info(f"""
                Connected to Pinecone index:
                - Index name: {st.session_state.index_name}
                - Dimension: {st.session_state.index_dimensions}
                - Host: https://index1-mwog0w0.svc.aped-4627-b74a.pinecone.io
                - Region: us-east-1
                - Type: Dense
                - Capacity: Serverless
                """)
            else:
                st.error("Failed to initialize all required services. Please check your API keys in the .env file.")
                
                # Show .env file template
                st.code("""
# Create a .env file in the same directory with the following content:
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_google_api_key
                """)
                return
    
    # Sidebar for PDF upload and main actions
    with st.sidebar:
        st.header("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file:
            with st.spinner("Processing PDF..."):
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name
                
                # Extract text
                pdf_text = extract_text_from_pdf(temp_path)
                os.unlink(temp_path)  # Delete temp file
                
                # Store in session state
                st.session_state.current_pdf_content = pdf_text
                st.session_state.current_pdf_name = uploaded_file.name
                
                # Chunk and embed
                chunks = chunk_text(pdf_text)
                embeddings = embed_chunks(chunks)
                
                # Store in Pinecone
                success = store_embeddings(chunks, embeddings, st.session_state.current_pdf_name)
                
                if success:
                    st.success(f"Successfully processed {uploaded_file.name}")
                else:
                    st.error(f"Failed to process {uploaded_file.name}")
        
        st.divider()
        st.header("Learning Tools")
        
        # Only enable these buttons if a PDF is loaded
        if st.session_state.current_pdf_content:
            # Quiz generation
            quiz_questions = st.slider("Number of quiz questions", min_value=3, max_value=10, value=5)
            if st.button("Generate Quiz"):
                with st.spinner("Generating quiz..."):
                    quiz_data = generate_quiz(st.session_state.current_pdf_content, num_questions=quiz_questions)
                    st.session_state.quiz_data = quiz_data
            
            # Assignment generation
            assignment_type = st.selectbox(
                "Assignment Type", 
                ["short_answer", "essay", "research"]
            )
            assignment_questions = st.slider("Number of assignment questions", min_value=1, max_value=5, value=3)
            
            if st.button("Generate Assignment"):
                with st.spinner("Generating assignment..."):
                    assignment_data = generate_assignment(
                        st.session_state.current_pdf_content, 
                        assignment_type,
                        num_questions=assignment_questions
                    )
                    st.session_state.assignment_data = assignment_data
        else:
            st.info("Please upload a PDF first to use these features")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Chatbot", "Quiz", "Assignment"])
    
    # Tab 1: Chatbot
    with tab1:
        st.header("Chat with your PDF")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if st.session_state.current_pdf_content:
            user_input = st.chat_input("Ask a question about your PDF...")
            
            if user_input:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.write(user_input)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Search for relevant context
                        similar_chunks = search_similar_chunks(
                            user_input, 
                            top_k=3, 
                            pdf_name=st.session_state.current_pdf_name
                        )
                        
                        # Extract text from results
                        context = "\n\n".join([match.metadata["text"] for match in similar_chunks])
                        
                        # Get response from Gemini
                        response = get_gemini_response(user_input, context)
                        
                        st.write(response)
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.info("Please upload a PDF to start chatting")
    
    # Tab 2: Quiz
    with tab2:
        st.header("Quiz")
        
        if "quiz_data" in st.session_state and st.session_state.quiz_data:
            quiz_data = st.session_state.quiz_data
            
            if "quiz_answers" not in st.session_state:
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
            
            if not st.session_state.quiz_submitted:
                for i, question in enumerate(quiz_data):
                    st.subheader(f"Question {i+1}")
                    st.write(question["question"])
                    
                    options = question["options"]
                    option_labels = ["A", "B", "C", "D"]
                    
                    # Create radio buttons for options
                    answer = st.radio(
                        "Select your answer:",
                        options=option_labels[:len(options)],
                        key=f"q{i}",
                        index=None
                    )
                    
                    # Display options
                    for j, option in enumerate(options):
                        st.write(f"{option_labels[j]}: {option}")
                    
                    st.session_state.quiz_answers[i] = answer
                    st.divider()
                
                if st.button("Submit Quiz"):
                    st.session_state.quiz_submitted = True
                    st.experimental_rerun()
            else:
                # Show results
                correct_count = 0
                
                for i, question in enumerate(quiz_data):
                    st.subheader(f"Question {i+1}")
                    st.write(question["question"])
                    
                    options = question["options"]
                    option_labels = ["A", "B", "C", "D"]
                    correct_letter = question["correct_answer"]
                    user_answer = st.session_state.quiz_answers.get(i)
                    
                    # Display options with correct/incorrect indicators
                    for j, option in enumerate(options):
                        current_label = option_labels[j]
                        if current_label == correct_letter:
                            st.success(f"{current_label}: {option} ✓")
                            if user_answer == current_label:
                                correct_count += 1
                        elif user_answer == current_label:
                            st.error(f"{current_label}: {option} ✗")
                        else:
                            st.write(f"{current_label}: {option}")
                    
                    # Show explanation
                    st.info(f"Explanation: {question['explanation']}")
                    st.divider()
                
                st.subheader(f"Your Score: {correct_count}/{len(quiz_data)}")
                
                if st.button("Retake Quiz"):
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_answers = {}
                    st.experimental_rerun()
        else:
            st.info("Generate a quiz from the sidebar to see it here")
    
    # Tab 3: Assignment
    with tab3:
        st.header("Assignment")
        
        if "assignment_data" in st.session_state and st.session_state.assignment_data:
            assignment_data = st.session_state.assignment_data
            
            for i, question in enumerate(assignment_data):
                with st.expander(f"Question {i+1}", expanded=True):
                    st.write(question["question"])
                    
                    if "hints" in question and question["hints"]:
                        st.subheader("Hints")
                        for hint in question["hints"]:
                            st.write(f"- {hint}")
                    
                    # Input area for answers
                    st.text_area("Your Answer:", key=f"assignment_q{i}", height=150)
                    
                    # Reveal key points button
                    if st.button("Show Key Points", key=f"key_points_btn_{i}"):
                        st.subheader("Key Points to Include")
                        for point in question["key_points"]:
                            st.write(f"- {point}")
        else:
            st.info("Generate an assignment from the sidebar to see it here")

if __name__ == "__main__":
    main()