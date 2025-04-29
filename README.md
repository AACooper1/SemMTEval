Update Apr 28:
  files added:
  1. improved_RSA_pipeline.py: I've found that the previous implementation might use the wrong data embedding for constructing the RDMs.(the RDM for a sentence with n tokens might not yield a NxN RDM)
  2.   es_results_bert-base-multilingual-cased.csv: The csv file showing the results of RSA from embeddings extracted from the multilingual bert.
     data columns mapping: source:segment, output:tgt, reference: last_translation, HTER:HTER
  3.   es_results_bert-base-spanish-wwm-cased-filtered: the csv file showing the results of RSA from embeddings extracted from Spanish monolingual bert(therefore there is only the out_ref RSA score)
     data columns mapping: output: suggestion, reference:last_translation, HTER:HTER
  4. data_analysis.py: script used to make the plot comparing HTER to kendal-tau score
  5. the plot
