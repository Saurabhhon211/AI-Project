from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


load_dotenv()

llm = ChatGoogleGenerativeAI(
     model="gemini-3.8-flash",
     temperature=0.2
 )
#llm = ChatOllama(temperature=0,model="gemma3:270m")

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer clearly and accurately."),
    ("human", "{question}\n\nText:\n{information}")
])

chain = chat_template | llm


information = '''US President Donald Trump on Sunday blamed "immigration of people" for the widespread unrest in France as violent protests broke out in Paris, Nice, Marseille and other cities over rising inflation.

The US president's comment came as the police in France reportedly arrested at least 1,800 people across the country after violent groups clashed with officers and looted shops during protests over rising living costs.

Trump, in a post on Truth Social, said, "France is burning! The immigration of people from an entirely different world, people with hatred in their hearts, has shattered that once great country."

"Close your Borders and take control, before it is too late," the US President added.'''


question = input("Enter what you want to ask:: ")

response = chain.invoke({
    "question": question,
    "information": information
})

print("\n--- Answer ---")
print("Input Tokens Details::",response.usage_metadata["input_token_details"])
print("Input Tokens::",response.usage_metadata["input_tokens"])
print("Output Tokens::",response.usage_metadata["output_tokens"])
print("Total Tokens::",response.usage_metadata["total_tokens"])

print(response.text)
