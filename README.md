# Research System

A tool for searching academic papers from arXiv and CORE, with basic AI relevance scoring.

## Features

- Search papers from arXiv and CORE APIs
- Filter results using Gemini AI (relevance score 0-100)
- Download PDFs and extract text
- Simple GUI interface (Tkinter)

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys in `.env`:
```
GEMINI_API_KEY=your_gemini_api_key_here
CORE_API_KEY=your_core_api_key_here  # Optional
```

## Usage

```bash
python -m src.app
```

## Project Structure

```
research_system/
├── src/
│   ├── app.py              # GUI (Tkinter)
│   ├── orchestrator.py     # Main workflow
│   ├── db.py               # SQLite operations
│   ├── models.py           # Data models
│   ├── sources/
│   │   ├── source.py       # Base source class
│   │   ├── arxiv_source.py # arXiv API
│   │   └── core_source.py  # CORE API
│   └── ai/
│       └── gemini_analyzer.py  # Gemini AI scoring
├── papers/                 # Downloaded PDFs (created automatically)
├── research.db            # SQLite database (created automatically)
├── requirements.txt       # Dependencies
├── .env.example          # Example environment variables
└── README.md            # This file
```

## Database

SQLite with two tables:
- **raw_results**: Search results with AI scores
- **publications**: Papers marked for keeping

## Configuration

### Search Options
- `query`: Search string
- `max_results`: Results per source (1-50)
- `use_arxiv`: Enable/disable arXiv
- `use_core`: Enable/disable CORE
- `arxiv_in_title`: Search in titles
- `arxiv_in_abstract`: Search in abstracts

### AI Analysis
- `threshold`: Minimum score to keep (0-100)
- `research_title`: Optional context for scoring

## Requirements

- Python 3.8+
- Gemini API key (required)
- CORE API key (optional)
- Internet connection

## License

MIT
### AI Analysis
- `threshold`: Minimum score to keep (0-100)
- `research_title`: Optional context for scoring

## Requirements

- Python 3.8+
- Gemini API key (required)
- CORE API key (optional)
- Internet connection

## License

MIT
- `arxiv_in_abstract`: Search in abstracts (arXiv)

### AI Analysis
- `threshold`: Minimum relevance score (0-100) to keep papers
- `research_title`: Optional context for better relevance scoring

## Requirements

- Python 3.8+
- Google Gemini API key (required)
- CORE API key (optional, for higher rate limits)
- Internet connection for API access

## License

MIT
