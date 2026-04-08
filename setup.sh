#!/usr/bin/env bash
# UniMentor Setup Script for Linux/macOS
# This script sets up everything needed to run UniMentor locally.

set -e

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# --- Helper functions ---
print_header() {
    echo ""
    echo -e "${BOLD}${BLUE}========================================${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}========================================${NC}"
    echo ""
}

print_step() {
    echo -e "${CYAN}[STEP $1/$TOTAL_STEPS]${NC} ${BOLD}$2${NC}"
}

print_success() {
    echo -e "  ${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "  ${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "  ${RED}✗${NC} $1"
}

print_info() {
    echo -e "  ${BLUE}ℹ${NC} $1"
}

TOTAL_STEPS=7
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_header "UniMentor Setup"
echo -e "Setting up your local AI course assistant..."
echo -e "Estimated total time: ${BOLD}5-15 minutes${NC} (depends on model download speed)"
echo ""

# --- Step 1: Check Python ---
print_step 1 "Checking Python installation..."

PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
fi

if [ -z "$PYTHON_CMD" ]; then
    print_error "Python is not installed."
    echo ""
    echo "  Please install Python 3.10 or higher:"
    echo "    macOS:  brew install python@3.12"
    echo "    Ubuntu: sudo apt install python3 python3-venv python3-pip"
    echo "    Fedora: sudo dnf install python3 python3-pip"
    echo ""
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    print_error "Python $PYTHON_VERSION found, but 3.10+ is required."
    echo "  Please upgrade Python to version 3.10 or higher."
    exit 1
fi

print_success "Python $PYTHON_VERSION found ($PYTHON_CMD)"

# --- Step 2: Check Ollama ---
print_step 2 "Checking Ollama installation..."

if ! command -v ollama &> /dev/null; then
    print_warning "Ollama is not installed."
    echo ""
    echo "  Install Ollama with one of these commands:"
    echo ""
    echo "    macOS:   brew install ollama"
    echo "    Linux:   curl -fsSL https://ollama.com/install.sh | sh"
    echo ""
    echo "  Or visit: https://ollama.com/download"
    echo ""
    read -p "  Press Enter after installing Ollama to continue (or Ctrl+C to exit)... "

    if ! command -v ollama &> /dev/null; then
        print_error "Ollama still not found. Please install it and re-run this script."
        exit 1
    fi
fi

print_success "Ollama is installed"

# Check if Ollama is running
if ! ollama list &> /dev/null 2>&1; then
    print_warning "Ollama does not appear to be running."
    print_info "Starting Ollama in the background..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open -a Ollama 2>/dev/null || ollama serve &>/dev/null &
    else
        ollama serve &>/dev/null &
    fi
    sleep 3
    if ollama list &> /dev/null 2>&1; then
        print_success "Ollama is now running"
    else
        print_warning "Could not auto-start Ollama. Please start it manually: ollama serve"
    fi
else
    print_success "Ollama is running"
fi

# --- Step 3: Pull LLM model ---
print_step 3 "Pulling language model (this may take a few minutes)..."
print_info "Model: mistral:7b-instruct-v0.3-q4_K_M (~4.1 GB)"

if ollama list 2>/dev/null | grep -q "mistral:7b-instruct-v0.3-q4_K_M"; then
    print_success "Model already downloaded"
else
    print_info "Downloading... (estimated time: 2-10 minutes depending on connection)"
    if ollama pull mistral:7b-instruct-v0.3-q4_K_M; then
        print_success "Model downloaded successfully"
    else
        print_warning "Could not pull mistral model. You can do this manually later:"
        print_info "  ollama pull mistral:7b-instruct-v0.3-q4_K_M"
        print_info "  Or try a smaller model: ollama pull phi3:mini"
    fi
fi

# --- Step 4: Pull embedding model ---
print_step 4 "Pulling embedding model..."
print_info "Model: nomic-embed-text (~274 MB)"

if ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
    print_success "Embedding model already downloaded"
else
    print_info "Downloading..."
    if ollama pull nomic-embed-text; then
        print_success "Embedding model downloaded successfully"
    else
        print_warning "Could not pull embedding model. You can do this manually:"
        print_info "  ollama pull nomic-embed-text"
    fi
fi

# --- Step 5: Create virtual environment ---
print_step 5 "Setting up Python virtual environment..."

if [ -d "venv" ]; then
    print_success "Virtual environment already exists"
else
    print_info "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created"
fi

# Activate venv
source venv/bin/activate
print_success "Virtual environment activated"

# --- Step 6: Install Python dependencies ---
print_step 6 "Installing Python dependencies..."
print_info "This may take 1-3 minutes..."

pip install --upgrade pip --quiet 2>/dev/null
if pip install -r requirements.txt --quiet 2>&1; then
    print_success "All dependencies installed"
else
    print_error "Some dependencies failed to install. Check the output above."
    print_info "Try running: pip install -r requirements.txt"
fi

# --- Step 7: Initialize data directories and config ---
print_step 7 "Initializing UniMentor..."

# Ensure directories exist
mkdir -p data/chroma_db
mkdir -p data/profiles
mkdir -p knowledge_base

# Copy default config if user config doesn't exist
if [ ! -f "config/user_config.yaml" ]; then
    cp config/default_config.yaml config/user_config.yaml
    print_success "Default configuration created (config/user_config.yaml)"
else
    print_success "User configuration already exists"
fi

# Copy sample profile if no profiles exist
if [ -z "$(ls -A data/profiles/ 2>/dev/null)" ]; then
    cp config/sample_profile.yaml data/profiles/default_profile.yaml
    print_success "Sample student profile created"
else
    print_success "Student profiles already exist"
fi

print_success "Data directories initialized"

# --- Done ---
print_header "Setup Complete!"

echo -e "${GREEN}UniMentor is ready to use!${NC}"
echo ""
echo -e "To start UniMentor, run:"
echo ""
echo -e "  ${BOLD}source venv/bin/activate${NC}"
echo -e "  ${BOLD}streamlit run app/main.py${NC}"
echo ""
echo -e "Or in one command:"
echo -e "  ${BOLD}source venv/bin/activate && streamlit run app/main.py${NC}"
echo ""
echo -e "${BLUE}Tips:${NC}"
echo -e "  - Drop your course PDFs, slides, and notes into the ${BOLD}knowledge_base/${NC} folder"
echo -e "  - Or use the drag-and-drop uploader in the web UI"
echo -e "  - Edit ${BOLD}config/user_config.yaml${NC} to customize settings"
echo -e "  - Your data stays 100% local on your machine"
echo ""
echo -e "${BOLD}Happy studying! 🎓${NC}"
