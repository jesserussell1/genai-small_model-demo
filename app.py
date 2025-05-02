import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

tokenizer, model = load_model()

# Sentence generator with creative option
def generate_sentences(words, model, tokenizer, creative=False, max_length=50, num_return_sequences=3):
    word_str = ', '.join(words)

    # Create the prompt depending on the checkbox
    if creative_mode:
        prompt = f"Write a fun, creative, and whimsical sentence using the words: {word_str}."
    else:
        prompt = f"Write a sentence using the words: {word_str}."

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    if creative:
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_p=0.95,
            temperature=1.2,
        )
    else:
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_p=0.8,
            temperature=0.7,
        )

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Streamlit UI
st.title("Controlled Text Generation Demo")
st.markdown("_Give it a few words. Watch a small language model craft a sentence — from literal to lyrical._")

user_input = st.text_input("Enter a few words separated by commas (e.g., cat, jumped, vase):")
creative_mode = st.checkbox("🎨 Creative Mode", value=False)



if user_input:
    words = [w.strip() for w in user_input.split(",") if w.strip()]
    if words:
        with st.spinner("Generating sentences..."):
            sentences = generate_sentences(words, model, tokenizer, creative=creative_mode)
        st.subheader("✨ Generated Sentences")
        for i, sentence in enumerate(sentences, 1):
            st.write(f"**{i}.** {sentence}")
    else:
        st.warning("Please enter at least one word.")
