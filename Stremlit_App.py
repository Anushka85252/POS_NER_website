import spacy
import streamlit as st
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import time
import matplotlib.pyplot as plt

# Function to load models based on user selection
def load_model(model_choice):
    if model_choice == "Small Model (en_core_web_sm)":
        return spacy.load("en_core_web_sm")
    elif model_choice == "Medium Model (en_core_web_md)":
        return spacy.load("en_core_web_md")
    elif model_choice == "Large Model (en_core_web_lg)":
        return spacy.load("en_core_web_lg")
    else:
        return spacy.load("en_core_web_sm")

# Set page config and title
st.set_page_config(page_title="Explore POS and NER", page_icon="📝", layout="wide")
st.title("✨ Explore POS and NER 🧠")

# Add custom CSS for improved UI with lighter background
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #A7C7E7, #A1D8F1); /* Light gradient background */
        font-family: 'Arial', sans-serif;
    }

    .stButton>button {
        background-color: #4CAF50;  /* Green button */
        color: white;
        font-size: 16px;
        border-radius: 8px;
        height: 50px;
        width: 200px;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #45a049;
    }

    .stTitle {
        color: #FFFFFF;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }

    .stSubheader {
        color: #FFFFFF;  /* White color for subheaders */
        font-size: 24px;
    }

    .stTextInput>div>input {
        background-color: #ecf0f1;  /* Light grey input field */
        font-size: 18px;
        color: #2C3E50;
        border-radius: 10px;
    }

    .stTextInput>div>textarea {
        background-color: #ecf0f1;
        font-size: 18px;
        color: #2C3E50;
    }
    </style>
    """, unsafe_allow_html=True
)

# Add a description with emoji
st.markdown("""
    🎉 **POS Tagging** and **NER** (Named Entity Recognition) allows you to explore text and discover its structure and entities.
    ✨ With just a sentence, explore **Part-of-Speech (POS)** tagging and **Named Entities** to unlock hidden insights.
    🧠 Choose the model that best suits your needs and let’s analyze the text! 
""")

# Allow the user to select the model
model_choice = st.selectbox(
    "Choose a model for POS tagging and NER 🌐",
    ["Small Model (en_core_web_sm)", "Medium Model (en_core_web_md)", "Large Model (en_core_web_lg)"]
)

# Load the chosen model
nlp = load_model(model_choice)

# Function for POS tagging
def pos_tagging(text):
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

# Function for Named Entity Recognition (NER)
def named_entity_recognition(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Streamlit UI for text input
sentence = st.text_area('Enter a sentence 📝', '', height=150)

# Add a slider to limit the number of NER entities displayed
num_entities = st.slider("Select number of entities to display:", min_value=1, max_value=10, value=5)

# Button to trigger the analysis
if st.button('Analyze 🔍'):
    with st.spinner('Analyzing...'):
        time.sleep(1)  # Simulate processing delay

        if sentence:
            # POS tagging
            pos_tags = pos_tagging(sentence)
            # NER analysis
            ner_results = named_entity_recognition(sentence)

            # Display POS Tags
            st.subheader("🔑 POS Tags:")
            st.write(pos_tags)

            # Display Named Entities
            st.subheader("💡 Named Entities:")
            st.write(ner_results[:num_entities])  # Display only the selected number of entities

            # Generate Word Cloud for Named Entities, only if there are named entities
            if ner_results:
                text = ' '.join([ent[0] for ent in ner_results[:num_entities]])
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
                st.image(wordcloud.to_array(), caption="Word Cloud for Named Entities", use_column_width=True)
            else:
                st.warning("⚠️ No named entities found in the sentence.")

            # Visualize POS tag distribution with color and style
            pos_list = [pos for _, pos in pos_tags]
            pos_counter = Counter(pos_list)

            # Plot POS distribution
            fig, ax = plt.subplots()
            ax.bar(pos_counter.keys(), pos_counter.values(), color="#56B4E9")
            ax.set_xlabel('POS Tag')
            ax.set_ylabel('Frequency')
            ax.set_title('📊 Distribution of POS Tags', fontsize=16, color="#2980B9")
            st.pyplot(fig)

            # Show POS and NER in a nicely formatted table
            pos_df = pd.DataFrame(pos_tags, columns=["Word", "POS Tag"])
            ner_df = pd.DataFrame(ner_results, columns=["Entity", "NER Label"])

            st.subheader("📋 POS and NER Results in Table Format")
            st.write("### POS Tags Table")
            st.table(pos_df)
            st.write("### Named Entities Table")
            st.table(ner_df)

            # Highlight Named Entities in the sentence with colored background
            def highlight_entities(text, entities):
                for ent in entities:
                    text = text.replace(ent[0], f"<span style='background-color: #FFFF00'>{ent[0]}</span>")
                return text

            highlighted_text = highlight_entities(sentence, ner_results)
            st.markdown(f"### Highlighted Sentence: <br> {highlighted_text}", unsafe_allow_html=True)

        else:
            st.warning("⚠️ Please enter a sentence to analyze.")
