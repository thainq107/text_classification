import json
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

label2id = {
    'negative': '0',
    'positive': '1'
}

model = SentenceTransformer('all-MiniLM-L12-v2')

def compute_similarity_sentence(sentences, top_k=5):
    embeddings = model.encode(sentences)
    cos_sim = util.cos_sim(embeddings, embeddings)
    values, indices = cos_sim.topk(top_k+1)
    top_k_indices = {}
    for index in tqdm(range(len(indices))):
        data = {
            'top_index':indices[index][1:].tolist(),
            'score':values[index][1:].tolist()
        }
        top_k_indices[index] = data
    return top_k_indices


def preprocess_prompt(examples, top_k_indices, corpus, labels, id2label):
    sentences = examples["text"]
    prompts = []
    for index, _ in enumerate(zip(sentences)):
        prompt = 'Here are examples of texts and their sentiments'
        top_indexs = top_k_indices[index]['top_index']
        for top_index in top_indexs:
            top_sentence = corpus[top_index]
            top_label = id2label[str(labels[top_index])]
            prompt = " ".join(
                [
                    prompt,
                    ". Text: ", 
                    top_sentence,
                    ". Sentiment: ",
                    top_label
                ]
            )
        prompts.append(prompt)
    
    examples["prompt"] = prompts
    return examples

def get_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used//1024**2

def postprocess_text(predictions, labels):
    predictions = [prediction.strip() for prediction in predictions]
    labels = [label2id[label.strip()] for label in labels]

    for idx in range(len(predictions)):
        if predictions[idx] in label2id:
           predictions[idx] = label2id[predictions[idx]]
        else:
            predictions[idx] = '-100'
    return predictions, labels