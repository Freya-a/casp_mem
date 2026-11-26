class NLIScorer:
    def __init__(self, model_path, max_length=512):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True).to('cpu').eval()
        self.max_length = max_length
        labels = {str(v).lower(): int(k) for k,v in self.model.config.id2label.items()}
        self.labels = {}
        for name in ('entailment','contradiction','neutral'):
            matches = [idx for label,idx in labels.items() if name in label]
            if len(matches) != 1:
                raise ValueError(f'NLI model does not identify {name} label: {labels}')
            self.labels[name] = matches[0]

    def score(self, premise, hypothesis):
        inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, max_length=self.max_length)
        with self.torch.inference_mode():
            probabilities = self.torch.softmax(self.model(**inputs).logits, dim=-1)[0].tolist()
        return {name:probabilities[index] for name,index in self.labels.items()}
