import os
from groq import Groq
from ..core.constants import SYSTEM_PROMPTS

class AIService:
    def __init__(self):
        self.api_keys = [
            os.getenv("GROQ_API_KEY", ""),
            os.getenv("GROQ_API_KEY_1", ""),
            os.getenv("GROQ_API_KEY_2", ""),
            os.getenv("GROQ_API_KEY_3", ""),
            os.getenv("GROQ_API_KEY_4", ""),
            os.getenv("GROQ_API_KEY_5", "")
        ]
        self.api_keys = [key for key in self.api_keys if key.strip()]
        if not self.api_keys:
            print("Warning: No GROQ API keys found.")
        self.current_key_idx = 0
            
    def get_client(self) -> Groq:
        if not self.api_keys:
            raise Exception("No Groq API keys available")
        return Groq(api_key=self.api_keys[self.current_key_idx])
        
    def rotate_key(self):
        if self.api_keys:
            self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
            print(f"Rotated to API Key index {self.current_key_idx}")

    def generate_response(self, query: str, context: str, task_type: str, language: str) -> str:
        prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nPlease answer the question in {language}."
        system_prompt = SYSTEM_PROMPTS.get(task_type, "You are a helpful assistant. Answer questions truthfully and clearly.")
        
        for _ in range(len(self.api_keys) + 1):
            try:
                client = self.get_client()
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                content = completion.choices[0].message.content
                
                # Clean JSON output if requested
                if task_type == "convert_to_json":
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]
                
                return content.strip()
            except Exception as e:
                print(f"Groq API Error: {str(e)}")
                self.rotate_key()
                
        return "I'm sorry, I'm having trouble retrieving a specific answer from the documents right now. However, I can still help you with a **Summary**, **JSON Conversion**, or **Compliance Check**. Would you like me to try one of those instead?"

    def generate_response_stream(self, query: str, context: str, task_type: str, language: str):
        prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nPlease answer the question in {language}."
        system_prompt = SYSTEM_PROMPTS.get(task_type, "You are a helpful assistant. Answer questions truthfully and clearly.")
        
        for _ in range(len(self.api_keys) + 1):
            try:
                client = self.get_client()
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    stream=True
                )
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
            except Exception as e:
                print(f"Groq API Stream Error: {str(e)}")
                self.rotate_key()
        
        yield "I encountered an error while streaming the response. Please try again."
        
    def detect_task_type(self, query: str) -> str:
        prompt = f"Categorize the following query into one of these tasks: convert_to_json, invoice_compare, summary, documentation, answer.\nOnly output the exact category name.\nQuery: {query}"
        
        try:
            client = self.get_client()
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a strict classifier. Return only the task name."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            content = completion.choices[0].message.content.strip().lower()
            if content in ["convert_to_json", "invoice_compare", "summary", "documentation", "answer"]:
                return content
        except Exception:
            pass
            
        return "answer"
