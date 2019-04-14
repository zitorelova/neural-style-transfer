build:
	@echo "Installing requirements..."
	@pip install -r requirements.txt
	@echo "Training model..."
	@python main.py
