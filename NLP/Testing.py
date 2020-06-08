from transformers import BertForSequenceClassification, AdamW


model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    use_cdn=True,
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False
)

model.cuda()

sentences = train.Review.values
labels = train.Sentiment.values