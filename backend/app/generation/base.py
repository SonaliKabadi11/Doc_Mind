from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system:str, user:str):
         """Return the model's text reply.

        Raises
        ------
        LLMTimeoutError
            if the provider did not respond in time
        LLMError
            on any other upstream failure or an empty response
        """