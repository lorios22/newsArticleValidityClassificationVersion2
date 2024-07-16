import json
import pandas as pd
from sklearn.model_selection import train_test_split
import re

file_path = 'data/fine_data.jsonl'

def clean_text(text):
    #Delete "Is this a valid article Article" at the beggining of the text
    text = re.sub(r'^Is this a valid article Article\s*', '', text, flags=re.IGNORECASE)
    #Delete URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    #Delete mentions (@users) and hashtags (#)
    text = re.sub(r'@\w+|#\w+', '', text)
    #Delete characters non alphanumeric (except white spaces)
    text = re.sub(r'\W', ' ', text)
    #Delete multiple spaces and spaces at the end and beginning 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#Load and process data from JSONL 
def load_and_process_data(file_path):
    data = []
    try:
        #Delete the JSONL line by line 
        with open(file_path, 'r') as file:
            for line in file:
                #Load each line as a JSON object
                try:
                    json_data = json.loads(line)
                    
                    #Get the content of'messages'
                    messages = json_data.get('messages', [])

                    #Get the content the role of 'user' and 'assistant'
                    user_message = None
                    assistant_message = None
                    
                    for message in messages:
                        role = message.get('role', 'No role')
                        content = message.get('content', 'No content')
                        if role == 'user':
                            user_message = content
                        elif role == 'assistant':
                            assistant_message = content
                    
                    if user_message and assistant_message:
                        if 'Invalid' in assistant_message:
                            label = 0  
                        elif 'Valid' in assistant_message:
                            label = 1  
                        else:
                            label = -1  #Label by default in case of error
                        
                        data.append({'content': user_message, 'label': label})
                        
                except json.JSONDecodeError:
                    print("Error decoding JSON from a line in the file.")
    except FileNotFoundError:
        print(f"The archive in {file_path} was not found.")
    except Exception as e:
        print(f"It occurs an unexpected error: {e}")
    
    df = pd.DataFrame(data)
    
    df = df[df['label'].isin([0, 1])]

    df['content'] = df['content'].apply(lambda x: clean_text(clean_text(x)))

    return df

#Split the data of training and validation
def split_data(df, test_size=0.2, random_state=42):
    X = df['content']
    y = df['label']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
