class EmbeddingClient:
    def __init__(self, model_path, max_length=2048):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, padding_side='left')
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to('cpu').eval()
        self.max_length = max_length

    def embed(self, text, query=False):
        if query:
            text = 'Instruct: Retrieve relevant memory conditions for the query\nQuery: ' + text
        tokens = self.tokenizer([text], padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        with self.torch.inference_mode():
            hidden = self.model(**tokens).last_hidden_state[:, -1, :]
            return self.torch.nn.functional.normalize(hidden, p=2, dim=1)[0].tolist()
