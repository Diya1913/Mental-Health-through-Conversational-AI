from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

nltk.download("vader_lexicon")
sentiment_analyzer = SentimentIntensityAnalyzer()

def generate_response(prompt, max_length=100):
    inputs = tokenizer([prompt], return_tensors="pt")
    output = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def analyze_sentiment(text):
    scores = sentiment_analyzer.polarity_scores(text)
    return scores["compound"]

def get_sentiment_category(sentiment_score):
    if sentiment_score >= 0.5:
        return "positive"
    elif sentiment_score <= -0.5:
        return "negative"
    else:
        return "neutral"

def provide_resources(sentiment_category):
    if sentiment_category == "negative":
        return "I'm sorry to hear that you're feeling down. Here are some resources that may help:\n" \
               "- National Suicide Prevention Lifeline: 1-800-273-8255\n" \
               "- Crisis Text Line: Text 'HOME' to 741741\n" \
               "- Find a mental health professional: https://www.psychologytoday.com/us/therapists\n"
    else:
        return ""

def handle_crisis(user_input):
    crisis_keywords = ["suicide", "kill myself", "self-harm", "hopeless"]
    if any(keyword in user_input.lower() for keyword in crisis_keywords):
        return "I'm concerned about your safety. If you're having thoughts of suicide or self-harm, " \
               "please reach out for help immediately:\n" \
               "- National Suicide Prevention Lifeline: 1-800-273-8255\n" \
               "- Crisis Text Line: Text 'HOME' to 741741\n" \
               "Remember, you are not alone, and help is available."
    return ""

print("Welcome to the Mental Health Chatbot!")
print("Type 'quit' to exit.")

chat_history = []

while True:
    user_input = input("User: ")
    if user_input.lower() == "quit":
        break

    chat_history.append(f"User: {user_input}")

    crisis_response = handle_crisis(user_input)
    if crisis_response:
        print(f"\nAI: {crisis_response}\n")
        chat_history.append(f"AI: {crisis_response}")
        continue

    sentiment_score = analyze_sentiment(user_input)
    sentiment_category = get_sentiment_category(sentiment_score)
    resources = provide_resources(sentiment_category)

    if sentiment_category == "negative":
        prompt = f"User: {user_input}\nAI: I'm sorry to hear that you're feeling down. Can you tell me more about what's been bothering you?\nUser: "
    else:
        prompt = f"User: {user_input}\nAI:"

    response = generate_response(prompt)
    response = f"{resources}\n{response}"
    print(f"\nAI: {response}\n")
    chat_history.append(f"AI: {response}")
    