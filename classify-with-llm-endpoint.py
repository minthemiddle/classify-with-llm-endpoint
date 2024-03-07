import click
import pandas as pd
import requests
import configparser
import json
import os
from openai import OpenAI

# Determine the directory of the current script
script_dir = os.path.dirname(__file__)

# Construct the absolute path to the config.ini file
config_path = os.path.join(script_dir, 'config.ini')

# Load the configuration file
config = configparser.ConfigParser()
config.read(config_path)

# Set up LLM
# Check if 'API' section exists in the configuration
if 'API' not in config:
    raise Exception(f"Section 'API' not found in the configuration file at {config_path}")

llm_key = config['API']['key']

client = OpenAI(
    api_key=llm_key
)

# Load prompt
with open('prompt.txt', 'r') as file:
    prompt = file.read()

@click.command()
@click.argument('input_csv', type=click.Path(exists=True))
@click.option('--output-csv', type=click.Path(), default=None, help='Output CSV file path')
def classify_messages(input_csv, output_csv):
    """
    This script reads messages from a CSV file, sends them to an API for classification,
    and appends the response to the CSV file in a new column 'label_classification_llm'.
    """
    df = pd.read_csv(input_csv)

    if 'label_classification_llm' not in df.columns:
        df['label_classification_llm'] = None
    
    df['label_classification_llm'] = df['label_classification_llm'].astype('object')
    

    for index, row in df.iterrows():
        # Replace {{TEXT}} with row['message'] in the prompt
        formatted_prompt = prompt.replace("{{INPUT}}", row['message'])
        
        # print(formatted_prompt)
        # break
        
        # Send the request
        response = client.chat.completions.create(
          model="gpt-3.5-turbo-0125",
          response_format={ "type": "json_object" },
          messages=[
              {"role": "user", "content": formatted_prompt}
            ]
        )
        
        # Access the first choice
        first_choice = response.choices[0]
        
        # Extract the message content
        message_content = first_choice.message.content
        
        # Attempt to parse the message content as JSON
        try:
            data = json.loads(message_content)
            # Extract the 'classification'
            classification = data['classification']
        except json.JSONDecodeError:
            # If JSON parsing fails, log the message_content and its corresponding id to log.txt
            with open('log.txt', 'a') as log_file:
                log_file.write(f"ID: {index}, Output: {message_content}\n")
            print(f"Error parsing JSON for ID: {index}")
            classification = None
        
        # Update the 'label_classification_llm' column with the classification
        df.at[index, 'label_classification_llm'] = classification

    # Determine the output CSV file path
    output_csv = output_csv if output_csv else input_csv

    # Save the updated DataFrame to a CSV file
    df.to_csv(output_csv, index=False)

    print(f"Classification completed. Updated CSV saved to {output_csv}")

if __name__ == '__main__':
    classify_messages()
