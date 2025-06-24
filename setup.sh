#!/bin/bash

# Ditto Container Setup Script
# This script handles git submodules and Docker build for remote deployment

set -e

echo "🚀 Setting up Ditto TalkingHead Container..."

# Function to check if we're in a git repository
check_git_repo() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        echo "❌ Error: Not in a git repository"
        echo "Please clone the repository first:"
        echo "git clone --recursive <your-repo-url>"
        exit 1
    fi
}

# Function to setup submodules
setup_submodules() {
    echo "📂 Setting up git submodules..."
    
    if [ ! -f .gitmodules ]; then
        echo "❌ Error: No .gitmodules file found"
        exit 1
    fi
    
    # Initialize and update submodules
    git submodule update --init --recursive
    
    # Check if src directory has content
    if [ ! -f src/inference.py ]; then
        echo "❌ Error: Submodule src/ appears to be empty"
        echo "Trying to fetch submodule content..."
        git submodule update --remote --recursive
        
        if [ ! -f src/inference.py ]; then
            echo "❌ Error: Failed to fetch submodule content"
            echo "Please check your internet connection and git credentials"
            exit 1
        fi
    fi
    
    echo "✅ Submodules setup complete"
}

# Function to build Docker image
build_docker() {
    echo "🐳 Building Docker image..."
    
    # Check if src directory exists and has content
    if [ ! -d src ] || [ ! -f src/inference.py ]; then
        echo "❌ Error: src/ directory missing or empty"
        echo "Run setup_submodules first"
        exit 1
    fi
    
    # Build the Docker image
    if command -v docker-compose &> /dev/null; then
        echo "Using docker-compose..."
        docker-compose build
    else
        echo "Using docker build..."
        docker build -t ditto-talkinghead .
    fi
    
    echo "✅ Docker build complete"
}

# Function to run the container
run_container() {
    echo "🏃 Starting container..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d
        echo "✅ Container started with docker-compose"
        echo "Access the web interface at: http://localhost:8000"
        echo ""
        echo "To enter the container:"
        echo "docker-compose exec ditto-talkinghead bash"
    else
        docker run -d --gpus all \
            -v $(pwd)/checkpoints:/app/checkpoints \
            -v $(pwd)/data:/app/data \
            -v $(pwd)/output:/app/output \
            -p 8000:8000 \
            --name ditto-container \
            ditto-talkinghead
        echo "✅ Container started with docker run"
        echo "Access the web interface at: http://localhost:8000"
        echo ""
        echo "To enter the container:"
        echo "docker exec -it ditto-container bash"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  setup     - Setup git submodules"
    echo "  build     - Build Docker image"
    echo "  run       - Run the container"
    echo "  all       - Setup, build, and run (default)"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0          # Setup, build, and run everything"
    echo "  $0 setup    # Only setup submodules"
    echo "  $0 build    # Only build Docker image"
    echo "  $0 run      # Only run container"
}

# Main script logic
case "${1:-all}" in
    setup)
        check_git_repo
        setup_submodules
        ;;
    build)
        check_git_repo
        build_docker
        ;;
    run)
        run_container
        ;;
    all)
        check_git_repo
        setup_submodules
        build_docker
        run_container
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "❌ Unknown command: $1"
        show_usage
        exit 1
        ;;
esac

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download model checkpoints to ./checkpoints/"
echo "2. Add source images to ./data/"
echo "3. Start streaming: http://localhost:8000" 