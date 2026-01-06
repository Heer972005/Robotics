import torch

file_path = "G:\vs\robotics\Cart\cartpole_reinforce.pth"
checkpoint = torch.load(file_path)

# Print the keys to see what's inside (e.g., 'model_state_dict', 'optimizer_state_dict', etc.)
print(checkpoint.keys()) 

# If you want to access specific parts, like the model weights:
# model.load_state_dict(checkpoint['model_state_dict']) 
