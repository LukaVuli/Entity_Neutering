
from CommonTools.prompts import de_neuter_name
from CommonTools.GPT_agent import Ollama_Agent
import pandas as pd
import time
import gc
import concurrent.futures
from CommonTools.CommonTools import extract_it_features, StringInputWrapper, StringOutputWrapper
pd.options.mode.chained_assignment = None  # Disable chained assignment warning

def process_llm_responses(df, column_name, input_string,prompt, model_to_use = 'llama3.2'):
    df[column_name] = None
    counter = -1

    for i, row in df.iterrows():
        agent = Ollama_Agent(prompt, model_to_use, 60)
        def get_response(text):
            return agent.response(text, StringInputWrapper, StringOutputWrapper)

        counter += 1
        next_article = row[input_string]
        if isinstance(next_article, str):
            next_article_length = len(row[input_string])
        else:
            next_article_length = "N/A"

        print(f"Processed {counter}/{len(df)} rows, roughly {round(100*counter/len(df),2)}% completed. Next Article Length : {next_article_length} chars", end='\r')

        if pd.isna(row[input_string]) or row[input_string] == '':
            response = None
        else:
            # Timeout mechanism
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(get_response, row[input_string])
                try:
                    response = future.result(timeout=60*5)  # Set timeout
                except concurrent.futures.TimeoutError:
                    print(f"\nTimeout occurred at row {counter}. Moving to next entry.")
                    response = None
                except Exception as e:
                    print(f"\nUnexpected error at row {counter}: {e}")
                    response = None
            gc.collect()

        df.at[i, column_name] = response.strip() if response else None
    return df

if __name__ == "__main__":

    output_path = '/Users/User/Desktop/ParProcess_Output/'
    alldata = pd.read_csv('data.csv')
    alldata['Date'] = pd.to_datetime(alldata['Date'],format='mixed')
    random_sample = alldata.sample(50)

    #llama
    print("Starting LLM Processing")
    start_time = time.time()
    random_sample = process_llm_responses(random_sample,'LLM_DENEUTER_GUESS_LLAMA_3B', 'Body Text Neutered', de_neuter_name, model_to_use = 'llama3.2')
    random_sample = extract_it_features(random_sample, 'LLM_DENEUTER_GUESS_LLAMA_3B',['Ticker_Guess_llama_3B', 'Name_Guess_llama_3B', 'Date_Guess_llama_3B', 'Industry_Guess_llama_3B'])
    random_sample.to_csv(output_path + 'Dual_Submission_LLAMA_random_sample.csv')
    end_time = time.time()
    optimized_time = end_time - start_time
    minutes = optimized_time / 60
    print(f"This calculation took: {minutes:.2f} minutes")




    #gemma
    print("Starting LLM Processing")
    start_time = time.time()
    random_sample = process_llm_responses(random_sample,'LLM_DENEUTER_GUESS_GEMMA_4B', 'Body Text Neutered', de_neuter_name, model_to_use = 'gemma3:4b')
    random_sample = extract_it_features(random_sample, 'LLM_DENEUTER_GUESS_GEMMA_4B',['Ticker_Guess_GEMMA_4B', 'Name_Guess_GEMMA_4B', 'Date_Guess_GEMMA_4B', 'Industry_Guess_GEMMA_4B'])
    random_sample.to_csv(output_path + 'Dual_Submission_GEMMA_random_sample.csv')
    end_time = time.time()
    optimized_time = end_time - start_time
    minutes = optimized_time / 60
    print(f"This calculation took: {minutes:.2f} minutes")


