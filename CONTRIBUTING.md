# Contributing to AI-Based Single Image Dehazing System

Thank you for your interest in contributing to the **AI-Based Single Image Dehazing System**! We welcome bug reports, feature proposals, model architecture additions, and pull requests.

## Development Setup

1. **Fork and Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/image-dehazing.git
   cd image-dehazing
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Run Test Suite:**
   ```bash
   pytest tests/ -v
   ```

## Pull Request Guidelines

- Ensure all new features include unit test coverage in `tests/`.
- Maintain PEP 8 coding standards and type hint annotations.
- Verify zero syntax or compilation errors before opening a PR.
