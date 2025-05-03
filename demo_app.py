import spacy
import streamlit as st

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
st.title('POS Tagging and Named Entity Recognition')

# Get user input
sentence = st.text_area('Enter a sentence:', '')

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
