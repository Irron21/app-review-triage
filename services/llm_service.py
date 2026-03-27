import os
import pandas as pd
import google.generativeai as genai

class LLMService:
    def __init__(self, metadata_path, gemini_api_key):
        self.app_metadata = {}
        self.unique_apps = []
        self.load_metadata(metadata_path)
        
        genai.configure(api_key=gemini_api_key)
        # Pointing to Gemini 2.5 natively 
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')

    def load_metadata(self, metadata_path):
        try:
            if os.path.exists(metadata_path):
                df_meta = pd.read_csv(metadata_path)
                df_meta = df_meta.drop_duplicates(subset=['App'])
                
                for _, row in df_meta.iterrows():
                    app_name_str = str(row['App'])
                    self.app_metadata[app_name_str] = {
                        'Category': str(row.get('Category', 'Unknown')),
                        'Rating': str(row.get('Rating', 'Unknown'))
                    }
                self.unique_apps = sorted(list(self.app_metadata.keys()))
                print(f"Loaded store metadata for {len(self.unique_apps)} applications via LLMService.")
            else:
                print(f"Metadata file '{metadata_path}' not found.")
        except Exception as e:
            print(f"Error loading Kaggle dataset metadata: {e}")

    def get_executive_summary(self, app_name, negative_reviews_list):
        meta = self.app_metadata.get(str(app_name), {'Category': 'Unknown', 'Rating': 'Unknown'})
        category = str(meta['Category'])
        rating = str(meta['Rating'])
        
        # Sample strict arrays dynamically
        sliced_reviews = negative_reviews_list[:10]
        reviews_text = "\n".join([f"- {r}" for r in sliced_reviews])
        
        prompt = f"""
        You are an expert Mobile App Product Manager.
        Review Context:
        - App Name: {app_name}
        - Category: {category}
        - Current Rating: {rating} stars
        
        Analyze this batch of critical user feedback for the app '{app_name}' and provide a single, cohesive 3-bullet-point executive action plan for its product team.
        
        Critical Feedback:
        {reviews_text}
        
        Do not include any pleasantries, headers, or conversational text. Return just the 3 bullet points.
        """
        try:
            response = self.gemini_model.generate_content(prompt)
            return str(response.text).strip()
        except Exception as e:
            return f"AI Action Plan could not be generated at this time. (Error: {e})"
