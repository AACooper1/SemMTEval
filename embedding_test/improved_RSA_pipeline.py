import numpy 
from rsatoolbox.rdm.rdms import RDMs
from torch import Tensor, eye, matmul
from torch.nn.functional import pad
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
import rsatoolbox

def define_model_tokenizer(model_name: str):
    """
    Define the model and tokenizer based on the provided model name.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer,model



"""Function that loads the data from csvfile, extracts the output, reference and source sentences(optional),
extracts the embeddings from the model and return the embeddings in the correct format (N x D) tensor for RDM calculation"""
def process_data (
    csvfile: str,
    model_name: str,
    output_col: str,
    ref_col: str,
    src_col: str = None,
    max_length: int = 512,
):
    """
    Process the data from a CSV file and extract embeddings using a specified model.
    
    returns a list of tuples containing the embeddings for the output, reference and source sentences(optionally).
    """
    # Load the data
    try:
        data = pd.read_csv(csvfile, sep='\t')
    except FileNotFoundError:
        return f"Error: The file {csvfile} was not found."  
    except Exception as e:
        return f"Error: An unexpected error occurred: {e}"

    """select a subset of the data for testing purposes"""
    # data = data.head(200)
    
    # Extract the sentences and HTER scores
    hter_scores = data['HTER'].tolist()
    out_sents = data['tgt'].tolist()
    ref_sents = data['last_translation'].tolist()
    
    if src_col:
        src_sents = data["segment"].tolist()
    else:
        src_sents = None
    
    # Define the model and tokenizer
    tokenizer, model = define_model_tokenizer(model_name)
    
    # Check if the 3 lists are of the same length
    try:
        assert len(out_sents) == len(ref_sents)
        if src_col:
            assert len(out_sents) == len(src_sents)
    except AssertionError:
        print("Error: The output, reference, and source sentences lists are not of the same length.")
        return None
    
    # set the model to evaluation mode
    model.eval()

    # Create a list of triples (out_sents, ref_sents, src_sents)
    if src_col:
        inputs_list = list(zip(out_sents, ref_sents, src_sents))
    else:
        inputs_list = list(zip(out_sents, ref_sents))


    # Tokenize the sentences

    encoded_inputs = [tokenizer(
        tuples,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length
    ) for tuples in inputs_list]
    
    # Get model output
    with torch.no_grad():
        model_outputs = [model(**encoded_input) for encoded_input in encoded_inputs]

    """extract the embeddings from the model output using the last hidden state
    the last_hidden_states should be a list(length = number of data rows) of tensors shape (3 (or 2), N, D)
    """
    last_hidden_states = [model_output.last_hidden_state for model_output in model_outputs]

    numpy_last_hidden_states = [last_hidden_state.numpy() for last_hidden_state in last_hidden_states]

    return numpy_last_hidden_states, hter_scores


def rsa (numpy_last_hidden_states: list[Tensor], hter:list, csvfile: str = None):


    """create RDMs for the numpy arrays and compare them using the kendall tau correlation
    return a list of tuples containing the kendall tau correlation for each pair of RDMs"""

    results = []
    for i,embeddings in enumerate(numpy_last_hidden_states):
        src_data = rsatoolbox.data.Dataset(embeddings[2])
        out_data = rsatoolbox.data.Dataset(embeddings[0])
        ref_data = rsatoolbox.data.Dataset(embeddings[1])
        src_rdm = rsatoolbox.rdm.calc_rdm(src_data)
        out_rdm = rsatoolbox.rdm.calc_rdm(out_data)
        ref_rdm = rsatoolbox.rdm.calc_rdm(ref_data)
        # print(f"src_rdm shape: {src_rdm.get_matrices().shape} \n")
        # print(f"out_rdm shape: {out_rdm.get_matrices().shape} \n")
        # print(f"ref_rdm shape: {ref_rdm.get_matrices().shape} \n")

        src_out_result = rsatoolbox.rdm.compare_kendall_tau(src_rdm, out_rdm)[0]
        src_ref_result = rsatoolbox.rdm.compare_kendall_tau(src_rdm, ref_rdm)[0]
        out_ref_result = rsatoolbox.rdm.compare_kendall_tau(out_rdm, ref_rdm)[0]

        # print(f"src_out_result: {src_out_result} \n")
        # print(f"src_ref_result: {src_ref_result} \n")
        # print(f"out_ref_result: {out_ref_result} \n")

        hter_score = hter[i] if hter else None

        results.append((i, src_out_result, src_ref_result, out_ref_result, hter_score))


        """save the results to a csv file"""
        results_df = pd.DataFrame(results, columns=['index', 'src_out_result', 'src_ref_result', 'out_ref_result', 'HTER'])
        results_df.to_csv(f"{csvfile}_results.csv", index=False)

if __name__ == "__main__":
    """example usage of the functions"""
    data_input, hter_scores = process_data(
        csvfile='spanish_merged.tsv',
        model_name='bert-base-multilingual-cased',
        output_col='tgt',
        ref_col='last_translation',
        src_col='segment',
        max_length=512,
    )
    
    results = rsa(
        numpy_last_hidden_states=data_input,
        hter=hter_scores,
        csvfile='spanish_merged.tsv',
    )
