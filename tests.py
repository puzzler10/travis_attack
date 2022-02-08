#### PLACEHOLDER until i refactor this out. 
import torch
from pprint import pprint

    
#### LOGGING
def get_tokens_from_token_ids_batch(tokenizer, ids_batch):
    l = []
    for i in range(ids_batch.shape[0]): 
        l.append(tokenizer.convert_ids_to_tokens(ids_batch[i,:]))
    return l
    

def pretty_print_pp_batch_and_next_token_probabilities(pp_output, tkn_kmaxidx, tkn_kmaxprob, generated_length): 
    """Goes through each paraphrase and shows at each timestep the next likely tokens. 
    Only will work for greedy search. 
    e.g. [
    "<pad> ['▁My, 0.289', '▁I, 0.261', '▁Hello, 0.07'] | Entropy: 4.23 ",
     "<pad> My ['▁name, 0.935', '▁Name, 0.005', 'name, 0.002'] | Entropy: 0.80 "
    ]
    """
    str_d = defaultdict(list)
    for i_tkn in range(0, generated_length-1): 
        ids = pp_output.sequences[:, :(i_tkn+1)]
        partial_pp = pp_tokenizer.batch_decode(ids)
        kth_ids,kth_probs = tkn_kmaxidx[:, i_tkn, :], tkn_kmaxprob[:, i_tkn, :]
        kth_tkns = get_tokens_from_token_ids_batch(pp_tokenizer, kth_ids)

        # enumerates examples in batch
        z = zip(partial_pp, kth_tkns, kth_probs, ent.detach())
        for i_ex, (ex_sen, ex_next_tkns, ex_next_probs, ex_e) in enumerate(z): 
            # Form nice formatted string mixing together tokens and probabilities
            tkn_tuples_l = [(tkn, round_t(prob,3)) for tkn, prob in zip(ex_next_tkns, ex_next_probs)]
            tkn_str = ['%s, %s' % t for t in tkn_tuples_l]
            # Add to dict of lists and add on entropy term. 
            str_d[i_ex].append(f"{ex_sen} {tkn_str} | Entropy: {ex_e[i_tkn]:.2f} ")

    for v in str_d.values():  pprint(v)