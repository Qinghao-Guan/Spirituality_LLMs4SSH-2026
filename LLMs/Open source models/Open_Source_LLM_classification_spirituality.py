import pandas as pd
from langchain_community.chat_models.ollama import ChatOllama
import sys

class Zero_shot_classifier:
    def __init__(self, model_name, file_path, column_name):
        self.llm = ChatOllama(model=model_name)
        self.df = pd.read_excel(file_path)
        self.column_name = column_name

    def zero_classify_spirituality(self, post):
        response = self.llm.invoke(f"""
            You are a human annotator, and you need to determine whether the sentence below is related to spirituality: "Yes" (1), "No" (0), or "hard to classify" (-1).
            "Yes" (1) means this sentence is related to spirituality. "No" (0) means the row is not related to spirituality. The "hard to classify" (-1) means that it is hard to classify the sentence into "Yes" or "No".
            For each sentence, simply return "1", "0", or "-1" and then give me the reason.
            Sentence: "{post}"
        """)
        return response.content

    def zero_classify_connectedness(self, post):
        response = self.llm.invoke(f"""
            You are a human annotator and need to classify the sentence below into one of the following five labels:  
            "connectedness to self" (0), "connectedness to others" (1), "connectedness to nature" (2), "connectedness to Transcendence" (3), or "hard to classify" (-1).  
            For each sentence, simply return "0", "1", "2", "3", or "-1". 
            Sentence: "{post}"
        """)
        return response.content

    def zero_definition_classify_spirituality(self, post):
        response = self.llm.invoke(f"""
            You are a human annotator and now I give you the definition of spirituality.
            "Spirituality is the pursuit and practice of experiences, beliefs, and values that influence and nurture the spirit, fostering personal growth, meaning, and a sense of connection to something greater than oneself". 
            Based on this definition, re-evaluate the sentence below to determine whether it is related to spirituality: "Yes" (1), "No" (0), or "hard to classify"(-1).
            Yes (1) means the sentence is related to spirituality. No (0) means the sentence is not related to spirituality. The "hard to classify" (-1) means that the sentence is hard to classify the sentence into "Yes" or "No".
            For each sentence, simply return "1", "0", or "-1" and then give me the reason.
            Sentence: "{post}"
        """)
        return response.content

    def zero_definition_classify_connectedness(self, post):
        response = self.llm.invoke(f"""
            You are a human annotator. Now I give you the definition of four types of connectedness.
            Connectedness to nature refers to the deep sense of relationship that individuals feel with the natural world and understanding of humanity’s place within the broader ecological system.
            Connectedness to self includes authenticity, inner harmony/inner peace, consciousness, self-knowledge and experiencing and searching for meaning in life.
            Connectedness to others emphasizes empathy, compassion, and a sense of community, recognizing that interpersonal connections contribute to personal growth, a deeper understanding of oneself and others, and overall spiritual well-being.
            Connectedness with Transcendence pertains to something or someone beyond the human level, such as the universe, transcendent reality, a higher power or God.
            Based on the definitions above, classify the sentence below into one of the following five labels:
            "connectedness to self" (0), "connectedness to others" (1), "connectedness to nature" (2), "connectedness to Transcendence" (3), or "hard to classify" (-1).  
            For each sentence, simply return "0", "1", "2", "3", or "-1".
            Sentence: "{post}"
        """)
        return response.content

    def apply_classifications_and_save(self, output_path):
        # Initialize new DataFrame to save results incrementally
        results_df = pd.DataFrame(columns=self.df.columns.tolist() + [
            'zero_spirituality_classification', 'zero_connectedness_classification',
            'zero_definition_spirituality_classification', 'zero_definition_connectedness_classification'
        ])

        for index, row in self.df.iterrows():
            post = row[self.column_name]
            spirituality = self.zero_classify_spirituality(post)
            connectedness = self.zero_classify_connectedness(post)
            def_spirituality = self.zero_definition_classify_spirituality(post)
            def_connectedness = self.zero_definition_classify_connectedness(post)

            # Append the classifications to the row
            #row['zero_spirituality_classification'] = spirituality
            row['zero_connectedness_classification'] = connectedness
            #row['zero_definition_spirituality_classification'] = def_spirituality
            row['zero_definition_connectedness_classification'] = def_connectedness

            # Append row to results DataFrame using concat
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

            # Save the results DataFrame to Excel after every row processed
            results_df.to_excel(output_path, index=False)

            # Print the processed line
            print(f"Processed and saved row {index+1}: {row.to_dict()}")

if __name__ == "__main__":
    model_name = sys.argv[1]
    input_filename = sys.argv[2]
    column_name = sys.argv[3]
    output_filename = f"{model_name}_zero_shot_classification_{input_filename}"

    classifier = Zero_shot_classifier(model_name, input_filename, column_name)
    classifier.apply_classifications_and_save(output_filename)

# python LLMclassification_spirituality.py llama3 quora.xlsx 'Answer'
# python LLMclassification_spirituality.py llama3 Instagram_caption.xlsx 'Caption'
