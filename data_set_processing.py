import pandas as pd
# import pyspark
from RSA import rsa_rdm


df = pd.read_csv("mtgeneval_a_M.es.tsv", sep='\t')

df = df.sample(n=200, random_state=42)
df = df.reset_index(drop=True)

source_en_sentences = df['segment']
out_es_sentences = df['tgt']
ref_es_sentences = df['last_translation']

# define model name
model_name = "FacebookAI/xlm-roberta-base"

results = rsa_rdm(source_en_sentences, out_es_sentences, ref_es_sentences, model_name)
# save results to a csv file
results_df = pd.DataFrame(results)  
results_df.to_csv('results_ES_50.csv', index=False)

