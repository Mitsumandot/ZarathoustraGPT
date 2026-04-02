import torch
from  train import BigramLanguageModel, decode

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = BigramLanguageModel()
model.to(device)


model.load_state_dict(torch.load("./model/model.pt", map_location=device))


model.eval()


start_context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_indices = model.generate(start_context, max_new_tokens=2000)[0].tolist()

print(decode(generated_indices))