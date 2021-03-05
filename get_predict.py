import torch

device = "cpu"

def forward_with_softmax(inp, model):
    logits = model(inp)
    return torch.softmax(logits, 0)[0][1]


def forward_with_sigmoid(input, model):
    return torch.sigmoid(model(input))


# accumalate couple samples in this array for visualization purposes
vis_data_records_ig = []


def interpret_sentence(model, sentence, TEXT, min_len=7):
    model.eval()
    text = [tok for tok in TEXT.tokenize(sentence)]
    if len(text) < min_len:
        text += ['pad'] * (min_len - len(text))
    indexed = [TEXT.vocab.stoi[t] for t in text]

    model.zero_grad()

    input_indices = torch.tensor(indexed, device=device)
    input_indices = input_indices.unsqueeze(0)

    # input_indices dim: [sequence_length]
    seq_length = min_len

    # predict
    pred = forward_with_sigmoid(input_indices, model).item()

    return '%.2f' % pred
