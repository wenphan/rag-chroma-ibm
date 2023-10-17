# Environment variables and keys
import os
credentials = {
	"url": "https://us-south.ml.cloud.ibm.com",
	"apikey": os.environ["WXA_API_KEY"]
}

wxa_project_id = os.environ["WXA_PROJECT_ID"]

# Query
query = "What did the president say about Ketanji Brown Jackson?"

# IBM Model Completion with LangChain
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

parameters = {
	GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
	GenParams.MIN_NEW_TOKENS: 1,
	GenParams.MAX_NEW_TOKENS: 100
}

model = Model(
	model_id=ModelTypes.GRANITE_13B_INSTRUCT,
	params=parameters,
	credentials=credentials,
	project_id=wxa_project_id
)

from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
granite_llm_ibm = WatsonxLLM(model=model)
result = granite_llm_ibm(query)

print("Query: " + query)
print("Result:" + result)
