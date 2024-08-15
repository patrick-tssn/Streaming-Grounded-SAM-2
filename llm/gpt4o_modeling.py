from .api_wrap import OpenAIAPIWrapper

class GPT4o():
    def __init__(self, model="gpt-4o-2024-05-13", device="cuda") -> None:

        self.model =  OpenAIAPIWrapper(model)
        


    async def generate(self, query):

        response, _ = self.model.get_completion(query)
        
        return response
