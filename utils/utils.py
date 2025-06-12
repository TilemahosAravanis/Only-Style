from diffusers import StableDiffusionXLPipeline

def find_matching_indices(list1, list2):
    dict1 = {}
    for index, element in enumerate(list1):
        dict1[element] = index

    matching_indices = []
    for element in list2:
        if element in dict1:
            matching_indices.append(dict1[element])

    return matching_indices

def convert_tokens_to_indices(pipeline: StableDiffusionXLPipeline, subjects: str, candidate_tokens: str):
    token_indices = []
    for sub, tokens in zip(subjects, candidate_tokens):
        out1 = pipeline.tokenizer(sub, padding=True, return_tensors="pt")
        out2 = pipeline.tokenizer(tokens, padding=True, return_tensors="pt")
        indices = find_matching_indices(out1['input_ids'][0].tolist(), out2['input_ids'][0].tolist())
        token_indices.append(indices[1:-1])

    return token_indices