"""
Given a pair of sentences, this script creates the RDMs for each sentence using the RSA toolbox and return the Kendall tau correlation between them.
"""

import numpy 
import rsatoolbox
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from torch import eye, matmul
from rsatoolbox.vis.rdm_plot import show_rdm
from rsatoolbox.rdm.rdms import RDMs
from torch.nn.functional import pad



def project_embedding(embeddings: Tensor, dim : int =2, start=0):
    projectionMatrix = pad(eye(dim), (0, embeddings.shape[0], start, (embeddings.shape[1] - start) - dim), value=0)

    result = matmul(embeddings, projectionMatrix)[:, :dim]

    return result.numpy(force=True)

def define_model_tokenizer(model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(f"{model_name}")
        model = AutoModel.from_pretrained(f"{model_name}")
        return tokenizer, model

def rsa_rdm(src_sent: list, out_sent: list, ref_sent:list,  model_name: str):

        Results = []
        # Check if tokenizer and model are already defined
        tokenizer, model = define_model_tokenizer(model_name)
        for i in range(len(src_sent)):
                la_example_out = tokenizer(out_sent[i], return_tensors='pt', padding='max_length', max_length=100)
                la_example_ref = tokenizer(ref_sent[i], return_tensors='pt', padding='max_length', max_length=100)
                la_example_src = tokenizer(src_sent[i], return_tensors='pt', padding='max_length', max_length=100)

                la_labels_out = [tokenizer.decode(i.item()) for i in la_example_out['input_ids'][0]]
                la_labels_ref = [tokenizer.decode(i.item()) for i in la_example_ref['input_ids'][0]]
                la_labels_src = [tokenizer.decode(i.item()) for i in la_example_src['input_ids'][0]]

                max_len = max(len([i for i in la_labels_out if i != "[PAD]"]), len([i for i in la_labels_ref if i != "[PAD]"]))
                la_labels_out = la_labels_out[:max_len]
                la_labels_ref = la_labels_ref[:max_len]
                la_labels_src = la_labels_src[:max_len]

                embeddings_out = model(**la_example_out, output_hidden_states=True)['hidden_states'][0][0][:max_len]
                embeddings_ref = model(**la_example_ref, output_hidden_states=True)['hidden_states'][0][0][:max_len]
                embeddings_src = model(**la_example_src, output_hidden_states=True)['hidden_states'][0][0][:max_len]

                rand_dim = numpy.random.randint(low=0, high=embeddings_out.shape[1] - 2)

                projection_out = project_embedding(embeddings_out, start=rand_dim)
                projection_ref = project_embedding(embeddings_ref, start=rand_dim)
                projection_src = project_embedding(embeddings_src, start=rand_dim)

                out_x = projection_out[:, 0]
                out_y = projection_out[:, 1]

                ref_x = projection_ref[:, 0]
                ref_y = projection_ref[:, 1]

                src_x = projection_src[:, 0]
                src_y = projection_src[:, 1]

                # ax.scatter(out_x, out_y, c='orange')
                # ax.scatter(ref_x, ref_y, c='purple')

                # for i, txt in enumerate(la_labels_out):
                #     ax.annotate(txt, (out_x[i], out_y[i]))

                # for i, txt in enumerate(la_labels_ref):
                #     ax.annotate(txt, (ref_x[i], ref_y[i]))

                # # plt.show()

                rdm_out = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(embeddings_out.detach().numpy()))
                rdm_ref = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(embeddings_ref.detach().numpy()))
                rdm_src = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(embeddings_src.detach().numpy()))

                rdm_out.pattern_descriptors = {'index': la_labels_out}
                rdm_ref.pattern_descriptors = {'index': la_labels_ref}
                rdm_src.pattern_descriptors = {'index': la_labels_src}


                src_out = float(rsatoolbox.rdm.compare_kendall_tau(rdm_src, rdm_out)[0][0]) 
                src_ref = float(rsatoolbox.rdm.compare_kendall_tau(rdm_src, rdm_ref)[0][0])
                out_ref = float(rsatoolbox.rdm.compare_kendall_tau(rdm_out, rdm_ref)[0][0])
                tuple = (i, src_out, src_ref, out_ref)
                Results.append(tuple)

        # return the results
        return Results
