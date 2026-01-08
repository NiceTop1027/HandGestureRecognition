"""
Generative AI Translator using Google Gemini
Translates gesture tokens into natural Korean sentences
"""
import google.generativeai as genai
import os

class GeminiTranslator:
    def __init__(self, api_key=None):
        if not api_key:
            api_key = os.environ.get('GEMINI_API_KEY')
            
        if not api_key:
            print("⚠️ GEMINI_API_KEY not found. AI Translation will not work.")
            self.model = None
            return
            
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            print("✨ Gemini AI successfully initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Gemini: {e}")
            self.model = None

    def translate(self, tokens):
        """
        Translate a list of tokens into a Korean sentence
        Args:
            tokens: List of strings (e.g., ['HELLO', 'NAME', 'ANTIGRAVITY'])
        Returns:
            str: Translated sentence
        """
        if not self.model or not tokens:
            return ""
            
        # Filter out invalid tokens
        valid_tokens = [t for t in tokens if t != '?' and t is not None]
        if not valid_tokens:
            return ""
            
        prompt = f"""
        Translate the following sequence of Sign Language Glosses (tokens) into a natural, polite Korean sentence.
        
        Tokens: {', '.join(valid_tokens)}
        
        Rules:
        1. Output ONLY the Korean sentence.
        2. Make it sound natural and polite (honorifics).
        3. If the meaning is ambiguous, guess the most likely context.
        
        Korean Sentence:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"❌ Translation error: {e}")
            return "번역 오류"

if __name__ == "__main__":
    # Test
    translator = GeminiTranslator(api_key="AIzaSyCKnJURvK-oOWNzfV3PN0SO8vare900lDw") # User needs to set this
    if translator.model:
        res = translator.translate(["HELLO", "NAME", "AI"])
        print(f"Result: {res}")
