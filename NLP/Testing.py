from transformers import BertTokenizer, BertForSequenceClassification


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

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case = True)

# Tokenize the Dataset:

max_length = 0
for sent in sentences:
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    max_length = max(max_length, len(input_ids))
print(f"Max sentence length: {max_length}")















