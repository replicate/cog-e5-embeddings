# Cog Wrapper for E5-Mistral-7B-Instruct

This model originates from Mistral-7B-v0.1 and has been enhanced through fine-tuning with a diverse set of multilingual datasets, granting it multilingual capabilities. However, given Mistral-7B-v0.1's primary training on English data, it is recommended to primarily use this model for English text processing tasks.

It takes a single task definition, a query and a document to output text embeddings and also a relevancy score. For example,

- *task:* "Given a web search query, retrieve relevant passages that answer the query"  
- *query:* "What is the recommended sugar intake for women"  
- *document:* "As a general guideline, the CDC’s average recommended sugar intake for women ages 19 to 70 is 20 grams per day."

To generate embeddings, you have the option to process a single document directly. However, if you intend to include a query, it's essential to specify the task simultaneously, as the model's training aligns with this approach. Failing to do so may result in a decline in performance. The task should be defined by a concise, one-sentence instruction that clearly articulates the task at hand. This method allows for the customization of text embeddings to suit various scenarios by leveraging natural language instructions.

See the original [paper](https://arxiv.org/pdf/2401.00368.pdf), [model page](https://huggingface.co/intfloat/e5-mistral-7b-instruct) and Github [repo](https://github.com/microsoft/unilm/tree/master/e5) for more details.

# API Usage

You need to have Cog and Docker installed to run this model locally. 

To build the docker image with cog and run a prediction:

```
cog predict -i task="Given a web search query, retrieve relevant passages that answer the query" -i query="how much protein should a female eat" -i document="As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day." -i normalize=True
```

To start a server and send requests to your locally or remotely deployed API:

```
cog run -p 5000 python -m cog.server.http
```

To generate text embeddings, you must supply a document in the form of a string. Additionally, you can input a query to calculate the similarity score, but it's advisable to include a task definition as previously mentioned. When both a query (combined with a task definition) and a document are provided, the output will include text embeddings (.npy) for each, along with a similarity score. The API input arguments are as follows:

- *task:* The task definition for the LLM.  
- *query:* The query used to calculate the relevancy score in comparison to the document.  
- *document:* The document for which to generate text embeddings.  
- *normalize:* Specifies whether to output normalized embeddings. The default value is False. If set to True, normalized embeddings will be output.  

# Limitations

Using this model for inputs longer than 4096 tokens is not recommended.

# Citation
```
@article{wang2023improving,
  title={Improving Text Embeddings with Large Language Models},
  author={Wang, Liang and Yang, Nan and Huang, Xiaolong and Yang, Linjun and Majumder, Rangan and Wei, Furu},
  journal={arXiv preprint arXiv:2401.00368},
  year={2023}
}

@article{wang2022text,
  title={Text Embeddings by Weakly-Supervised Contrastive Pre-training},
  author={Wang, Liang and Yang, Nan and Huang, Xiaolong and Jiao, Binxing and Yang, Linjun and Jiang, Daxin and Majumder, Rangan and Wei, Furu},
  journal={arXiv preprint arXiv:2212.03533},
  year={2022}
}
```
