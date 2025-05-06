from huggingface_hub import login
from transformers import AutoModel 

# login(token=huggingface_token)  # login with your User Access Token, found at https://huggingface.co/settings/tokens

# titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
titan = AutoModel.from_pretrained('../TITAN', trust_remote_code=True)
conch, eval_transform = titan.return_conch()

print("a")
