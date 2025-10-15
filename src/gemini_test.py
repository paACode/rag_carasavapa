
#Failing to make free tier work with litellm
from litellm import completion


import os

# Set your Gemini API key
os.environ["GEMINI_API_KEY"] = "AIzaSyD1XyDLX8BbkpPH6_xc5CSB4QZYvGsfXiQ"

# Call the completion function with the correct model
response = completion(
    model="gemini-2.5-pro-exp-03-25:free",  # Use the exact free-tier model name
    messages=[{"role": "user", "content": "Hello, Gemini!"}],
    provider="google"  # Specify the provider as 'google'
)

print(response)