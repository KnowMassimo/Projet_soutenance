import os
from dotenv import load_dotenv
import streamlit as st
from langchain.llms import OpenAI

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,  SequentialChain 

## Setting up the Environment
###############################################

load_dotenv()  # take environment variables from .env.
API_KEY = os.environ['OPENAI_API_KEY'] 

llm = OpenAI(openai_api_key= API_KEY,temperature=0.9)


## Defining the Prompt Templates
###############################################

# I'll create two prompt-templates:
# # 1_Meal template
# "Takes a list of ingredients and formats a prompt that asks the language-model to generate some recipes from this list."
# 2_Tips template
# "Takes the output of the language-model's response from the above template/API call, and give some tips for preserving this ingredients."


# Meal template.
prompt_template = PromptTemplate(
    input_variables=["ingredients"],
    template="Give me an example of 3 meals that could be made using the following ingredients: {ingredients}",
)

# Tips template for Preserving Ingredients.

tips_template = """"Gives advice on how to better preserve ingredients:

Meals:  
{meals}

"""
tips_template_prompt = PromptTemplate(
    template=tips_template,
    input_variables=['meals']

) 

## Creating LLMChain & SequentialChain objects
###############################################

meal_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    output_key="meals",  # the output from this chain will be called 'meals'.
    verbose=True
)

   
tips_chain = LLMChain(
    llm=llm,
    prompt=tips_template_prompt, 
    output_key="tips_meals",  # the output from this chain will be called 'tips_meals'.
    verbose=True
)

overall_chain = SequentialChain(
    chains=[meal_chain, tips_chain],
    input_variables=["ingredients"],
    output_variables=["meals", "tips_meals"],
    verbose=True
)


## Streamlit UI
###############################################

st.title("Leftovers Chef")
user_prompt = st.text_input("Enter a comma-separated list of ingredients")

if st.button("Generate") and user_prompt:
    with st.spinner("Generating..."):
        output = overall_chain({'ingredients': user_prompt})
     
       
        col1, col2 = st.columns(2)
        col1.write(output['meals'])
        col2.write(output['tips_meals'])



