wandb-login:
	@echo "Logging in to wandb..."
	@wandb login "$(KEY)" || (echo "WandB login failed. Check your API key." && exit 1)

wandb-logout:
	@echo "Logging out from wandb..."
	@-rm -f ~/.netrc  
	@-rm -f ~/.config/wandb/settings 
hf-login:
	@echo "Logging in to Hugging Face..."
	@huggingface-cli login --token "$(HF_TOKEN)" || (echo "Hugging Face login failed. Check your token." && exit 1)

hf-logout:
	@echo "Logging out from Hugging Face..."
	@huggingface-cli logout
