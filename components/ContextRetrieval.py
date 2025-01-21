
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings


class ContextRetriever:
    def __init__(self, endpoint, apikey, deployment_name, model_name, vs_path=None):
        text_embedding = AzureOpenAIEmbeddings(
                                azure_endpoint=endpoint,
                                api_key=apikey,
                                azure_deployment=deployment_name,
                                model=model_name
                            )

        self.store = FAISS.load_local(vs_path, text_embedding, allow_dangerous_deserialization=True)        

    def get_context(self, input, topk=5):
        return self.store.similarity_search(input, k=topk)
        
