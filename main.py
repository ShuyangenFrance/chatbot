import os
import pickle

from google.auth.transport.requests import Request

from google_auth_oauthlib.flow import InstalledAppFlow
from llama_index import (
    GPTSimpleVectorIndex,
    download_loader,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
)

from langchain.chat_models import ChatOpenAI


def authorize_gdocs():
    google_oauth2_scopes = [
        "https://www.googleapis.com/auth/documents.readonly"
    ]
    cred = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", 'rb') as token:
            cred = pickle.load(token)
    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", google_oauth2_scopes)
            cred = flow.run_local_server(port=0)
        with open("token.pickle", 'wb') as token:
            pickle.dump(cred, token)

def get_index():
    # define LLM
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo"))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    index = GPTSimpleVectorIndex.from_documents(
        documents, service_context=service_context
    )
    return index

if __name__ == '__main__':

    os.environ['OPENAI_API_KEY'] = "OPENAI_API_KEY"

    # change knowledge base resource here:
    kb="google"
    if kb == "google":
        authorize_gdocs()
        GoogleDocsReader = download_loader('GoogleDocsReader')
        gdoc_ids = ['ID_1','ID_2']
        loader = GoogleDocsReader()
        documents = loader.load_data(document_ids=gdoc_ids)
    elif kb=="local":
        documents= SimpleDirectoryReader('data').load_data()
    index=get_index()

    query = "What is the limitation of CSVs?"
    response = index.query(query, response_mode="compact")
    print(response)
