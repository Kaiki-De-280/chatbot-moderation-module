# Chatbot Moderation Module

Software module for automatic moderation of Russian-language chatbot user messages.

The module analyzes user messages and returns a structured moderation result that can be used by a chatbot or another external system to decide whether a message should be allowed, flagged, or additionally reviewed.

The project was developed as part of a bachelor's graduation thesis.

## Main features

- rule-based detection of undesirable content using configurable regular expressions;
- machine learning moderation layer;
- transformer-based moderation layer;
- aggregation of moderation results from several moderation approaches;
- structured JSON-like moderation output;
- Streamlit web interface for testing and demonstrating the module;
- editable external configuration for rule-based moderation.

## Moderation pipeline

The moderation module is based on a multi-layer approach:

1. **Rule-based layer**  
   Detects obvious violations using predefined rules and regular expressions.

2. **Machine learning layer**  
   Uses a trained classical ML model for binary text classification.

3. **Transformer-based layer**  
   Uses a transformer model for Russian-language toxicity classification.

4. **Aggregation layer**  
   Combines results from different moderation layers and forms the final decision.

The final result contains information about the moderation status, detected indicators, risk level, and recommended reaction.

## Project structure

```text
chatbot-moderation-module/
├── app/                         # Streamlit utility functions
├── models/                      # Model registry configuration
│   └── model_registry.json
├── pages/                       # Streamlit application pages
├── rules/                       # Rule-based moderation configuration
│   └── moderation_rules.json
├── src/                         # Core moderation module source code
│   ├── ml_moderator.py
│   ├── moderation_pipeline.py
│   ├── preprocessing.py
│   ├── rule_based_moderator.py
│   └── transformer_moderator.py
├── streamlit_app.py             # Streamlit application entry point
├── requirements.txt             # Python dependencies
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Kaiki-De-280/chatbot-moderation-module.git
cd chatbot-moderation-module
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

For Windows:

```bash
.venv\Scripts\activate
```

For Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the application

Run the Streamlit interface:

```bash
streamlit run streamlit_app.py
```

After launch, the application will open in a browser and provide access to the moderation interface.

## Configuration

Rule-based moderation rules are stored in:

```text
rules/moderation_rules.json
```

The model registry is stored in:

```text
models/model_registry.json
```

The registry describes available models, their types, paths, and additional metadata used by the application.

## Important notes

Trained models and datasets are not included in this repository because of file size limitations and dataset distribution restrictions.

To use the ML and transformer layers, trained model files must be placed locally according to the paths specified in:

```text
models/model_registry.json
```

The repository contains the source code, configuration structure, and interface required to demonstrate the architecture and logic of the moderation module.

## Datasets

The datasets used for model training are not distributed with this repository.

They were used only during the research and training stages of the graduation thesis. Dataset sources are referenced separately in the thesis text.

## Technologies

- Python
- Streamlit
- pandas
- scikit-learn
- joblib
- PyTorch
- Transformers

## Author

Lev Petrov

Bachelor's graduation thesis project:  
**Software module for automatic moderation of chatbot user messages**