from tkinter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from faq_data import faqs

# Download tokenizer
nltk.download('punkt')

# Prepare questions and answers
questions = list(faqs.keys())
answers = list(faqs.values())

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words='english')
question_vectors = vectorizer.fit_transform(questions)

# Chatbot response function
def get_response():
    user_input = user_entry.get()
    if user_input.strip() == "":
        chat_area.insert(END, "Bot: Please ask a question.\n\n")
        return

    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, question_vectors)

    best_match = similarity.argmax()
    best_score = similarity[0][best_match]

    if best_score > 0.35:
        response = answers[best_match]
    else:
        response = "Sorry, I couldn't understand your question. Please ask a related FAQ."


    chat_area.insert(END, f"You: {user_input}\n")
    chat_area.insert(END, f"Bot: {response}\n\n")
    user_entry.delete(0, END)

# GUI setup
root = Tk()
root.title("CodeAlpha FAQ Chatbot")
root.geometry("600x450")

Label(root, text="FAQ Chatbot", font=("Arial", 16, "bold")).pack(pady=10)

chat_area = Text(root, height=15, width=70)
chat_area.pack(pady=10)

user_entry = Entry(root, width=50)
user_entry.pack(pady=5)

Button(root, text="Ask", command=get_response, bg="blue", fg="white").pack()

root.mainloop()
