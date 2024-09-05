import json
import os
import random
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from datetime import datetime, timedelta

# Initialize the Lemmatizer
lemmatizer = WordNetLemmatizer()

# File to store reminders
REMINDER_FILE = "reminders.json"

# Load existing reminders from a file
def load_reminders():
    if os.path.exists(REMINDER_FILE):
        with open(REMINDER_FILE, "r") as f:
            return json.load(f)
    return {}

# Save reminders to a file
def save_reminders(reminders):
    with open(REMINDER_FILE, "w") as f:
        json.dump(reminders, f)

# Function to handle greetings
def greet():
    greetings = [
        "Hello! How can I assist you today?",
        "Hi there! What can I do for you?",
        "Hey! Need help with something?"
    ]
    return random.choice(greetings)

def preprocess_input(user_input):
    # Tokenize and lemmatize input
    tokens = word_tokenize(user_input.lower())
    return [lemmatizer.lemmatize(token) for token in tokens]

def respond_to_query(user_input):
    tokens = preprocess_input(user_input)

    # Simple keyword-based responses
    if any(token in tokens for token in ['hello', 'hi', 'hey']):
        return greet()
    elif 'how' in tokens and 'you' in tokens:
        return "I'm just a computer program, but thanks for asking!"
    elif 'name' in tokens:
        return "I'm ChatBot, your virtual assistant."
    elif 'reminder' in tokens:
        if 'set' in tokens:
            return set_reminder()
        elif 'list' in tokens:
            return list_reminders()
    elif 'exit' in tokens or 'bye' in tokens:
        return "Goodbye! Have a great day!"
    else:
        return handle_small_talk(tokens)

def handle_small_talk(tokens):
    small_talk_responses = {
        'weather': "I don't have a weather app, but I hope it's nice out!",
        'joke': "Why did the robot go on a diet? Because it had too many bytes!",
        'thank': "You're welcome! I'm here to help.",
        'love': "I appreciate the sentiment! But I'm just a bot.",
        'food': "I can't eat, but I can suggest recipes if you need!",
    }

    for keyword, response in small_talk_responses.items():
        if keyword in tokens:
            return response

    return "I'm sorry, I don't understand. Can you ask something else?"

# Function for setting reminders
def set_reminder():
    reminder_text = input("What would you like to be reminded about? ")
    time_str = input("When do you want to be reminded? (e.g., 'in 5 minutes', 'at 15:30'): ")

    time_to_remember = None
    try:
        if 'in' in time_str:
            amount, unit = time_str.split()[1], time_str.split()[2]
            if unit.startswith('minute'):
                time_to_remember = datetime.now() + timedelta(minutes=int(amount))
            elif unit.startswith('hour'):
                time_to_remember = datetime.now() + timedelta(hours=int(amount))
        elif 'at' in time_str:
            hour, minute = map(int, time_str.split()[1].split(':'))
            now = datetime.now()
            time_to_remember = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if time_to_remember < now:
                time_to_remember += timedelta(days=1)
    except (ValueError, IndexError):
        return "Invalid time format. Please use 'in X {minutes/hours}' or 'at HH:MM'."

    # Save the reminder
    reminders = load_reminders()
    reminders[str(time_to_remember)] = reminder_text
    save_reminders(reminders)

    return f"Okay, I will remind you to '{reminder_text}' at {time_to_remember}."

# Function to list reminders
def list_reminders():
    reminders = load_reminders()
    if not reminders:
        return "You have no reminders set."

    reminder_list = "\n".join(f"At {time_key}: {text}" for time_key, text in reminders.items())
    return f"Here are your reminders:\n{reminder_list}"

# Main loop to run the chatbot
def main():
    print(greet())

    while True:
        user_input = input("\nYou: ")
        response = respond_to_query(user_input)
        print("ChatBot:", response)

        if "exit" in user_input.lower() or "bye" in user_input.lower():
            break

if __name__ == "__main__":
    main()
