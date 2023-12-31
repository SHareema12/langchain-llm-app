from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType
from dotenv import load_dotenv

# loads .env file to get the environmental variables without having to hard code here
load_dotenv()

def generate_pet_name(animal_type, pet_color):
    # initialize our LLM, here we're initializing openai with a temperature of 0.7 (0 being not creative, 1 being most creative)
    llm=OpenAI(temperature=0.5)

    # initialize a prompt_template where we pass in the user parameters and create a dynamic prompt template
    prompt_template_name = PromptTemplate(input_variables=['animal_type', 'pet_color'], template="I have a {animal_type} pet and I want a cool name for it, it is {pet_color} in color. Suggest five cool names for my pet.")
    # name = llm("I have a dog pet and I want a cool name for it. Suggest five cool names for my pet.")

    # initialize the LLM chain where we pass in the llm and the prompt
    # output key is the part of the response we want that we can refer to later
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="pet_name")

    # call the chain with the user's parameters, this reponse will be in json format
    response = name_chain({'animal_type': animal_type, 'pet_color': pet_color})
    return response

def langchain_agent():
    llm = OpenAI(temperature=0.5)
    tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    result = agent.run(
        "Could you ELI5 how the internet works? Please explain in a very simple way using every day concepts."
    )

    print(result)

if __name__ == "__main__":
    # print(generate_pet_name("cow", "purple"))
    langchain_agent()

  