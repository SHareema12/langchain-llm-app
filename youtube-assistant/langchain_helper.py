from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

video_url = "https://www.youtube.com/watch?v=MTJZpO3bTpg"

def create_vector_db_from_youtube(video_url: str) -> FAISS:
  ## load transcript, split into chunks, and save those chunks into vector stores
  loader = YoutubeLoader.from_youtube_url(video_url)
  transcript = loader.load()

  # openai's token size is limitied so we need to send the text in chunks
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
  docs = text_splitter.split_documents(transcript)

  # create vector db 
  db = FAISS.from_documents(docs, embeddings)

  return db

def get_response_from_query(db, query, k=4):
  # text-davinci can handle 4096 tokens
  # sending 4 docs at a time 
  docs = db.similarity_search(query, k=k)
  docs_page_content = " ".join([d.page_content for d in docs])

  llm = OpenAI(model="text-davinci-003")
  prompt = PromptTemplate(
    input_variables=["question", "docs"],
    template="""
    You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
     """
  )

  chain = LLMChain(llm=llm, prompt=prompt)

  response = chain.run(question=query, docs=docs_page_content)
  response = response.replace("\n", "")
  return response, docs