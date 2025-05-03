import spacy
import streamlit as st

# Set custom title and background color
st.set_page_config(
    page_title="POS Tagging and NER",  # Page title
    page_icon="📝",  # Page icon (emoji or image path)
    layout="wide",  # Layout configuration (can be 'centered' or 'wide')
)

# Apply custom CSS for a better theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F0F8FF;  # Light background color
        font-family: 'Arial', sans-serif;
    }

    .stButton>button {
        background-color: #4CAF50;  # Green button
        color: white;
        font-size: 16px;
        border-radius: 8px;
        height: 50px;
        width: 200px;
    }

    .stButton>button:hover {
        background-color: #45a049;
    }

    .stTitle {
        color: #2C3E50;  # Dark title color
        font-size: 36px;
        font-weight: bold;
    }

    .stSubheader {
        color: #2980B9;  # Blue color for subheaders
        font-size: 24px;
    }

    .stMarkdown {
        color: #34495E;  # Text color for markdown
    }

    .stTextInput>div>input {
        background-color: #ecf0f1;  # Light grey input field
        font-size: 18px;
        color: #2C3E50;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Function for POS tagging
def pos_tagging(text):
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

# Function for Named Entity Recognition (NER)
def named_entity_recognition(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Streamlit UI
st.title('📝 POS Tagging and Named Entity Recognition (NER)')

# Get user input
sentence = st.text_area('Enter a sentence:', '', height=150)

# Button to trigger the analysis
if st.button('Analyze'):
    if sentence:
        # POS tagging
        pos_tags = pos_tagging(sentence)
        # NER analysis
        ner_results = named_entity_recognition(sentence)
        
        # Display the results
        st.subheader("POS Tags:")
        st.write(pos_tags)

        st.subheader("Named Entities:")
        st.write(ner_results)
    else:
        st.warning("Please enter a sentence.")
