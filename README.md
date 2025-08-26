# Research System

A tool to support the process of searching and analyzing scientific publications. The application allows searching sources such as arXiv and CORE, evaluating the relevance of the results using a local language model, and then extracting structured data from selected documents.

## Main Features

*   **Publication Search**: Searches arXiv and CORE databases based on a given query.
*   **Local Database**: Results are saved in a local SQLite database, allowing you to work with the data without re-running searches.
*   **Web User Interface**: A web application based on Streamlit (`src/app.py`).
*   **AI-Powered Relevance Analysis**: Ability to evaluate search results for relevance to a given research thesis using a locally run language model (via an LMStudio server).
*   **Publication Management**: A process for selecting and "promoting" the most interesting results to a separate list of publications.
*   **Downloading and Processing**: The application can download PDF files for publications and extract their content into Markdown format.
*   **Data Extraction**: Using the `langextract` library powered by Google Gemini, it is possible to extract specific, structured information from the text (e.g., tool names, use cases), according to a given configuration.

## Installation and Setup

### 1. Prerequisites

*   Python version 3.8 or newer.
*   Access to an LMStudio server (for the relevance analysis feature).
*   API keys for CORE and Google AI services.

### 2. Environment Setup

Using a Python virtual environment is recommended.

```powershell
# Create and activate a virtual environment
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
# Make sure you have the latest version of pip
python -m pip install --upgrade pip

# Install the project in editable mode (this will also install dependencies from pyproject.toml)
python -m pip install -e .
```

### 4. API Key Configuration

Create a `.env` file by copying the `.env.example` template and filling in your values.

```powershell
Copy-Item .env.example .env
```

Then, edit the `.env` file with your API keys:
```
CORE_API_KEY="your_core_api_key"
GOOGLE_API_KEY="your_google_api_key"

# Optionally, if LMStudio is running on a different address
# LMSTUDIO_ENDPOINT="http://another-address:1234/v1/chat/completions"
```

### 5. Running the Application

To run the application:

```powershell
streamlit run src/app.py
```

## Project Structure

*   `src/app.py`: The main Streamlit web application file.
*   `src/services.py`: Core business logic, connecting the individual components.
*   `src/db/`: SQLite database schema definition and operations.
*   `src/sources/`: Implementations for searching sources (arXiv, CORE).
*   `src/evaluators/`: Client for communicating with LMStudio for relevance assessment.
*   `src/extractors/`: Logic for data extraction from text using `langextract` and Gemini.
*   `src/config.py`: Default configurations and constants, e.g., prompt templates.
*   `research.db`: Default name for the SQLite database file.

## LangExtract Extraction Configuration

The process of extracting specific, structured information from text using LangExtract is configurable. The application uses default settings defined in `src/config.py`.

You can override these settings through the "Extraction Config" tab in the web interface. When you save your changes, a `src/config/extraction_config.json` file is created, which will be used for future extractions. This allows you to define:
*   `prompt`: The instruction for the language model.
*   `allowed_classes`: A list of entity categories to be extracted.
*   `examples`: Examples for the model (few-shot learning) to improve extraction quality.

## License

MIT
