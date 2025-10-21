#!/bin/bash

echo "🚀 Multi-Modal AI Chat - Quick Start Script"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your API keys before continuing."
    echo "   Run: nano .env"
    exit 1
fi

# Check if API keys are set
if ! grep -q "sk-" .env && ! grep -q "AIza" .env; then
    echo "⚠️  Warning: No API keys found in .env"
    echo "   Please add your OPENAI_API_KEY or GOOGLE_API_KEY"
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""
echo "🔨 Building and starting services..."
docker-compose up -d --build

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo ""
    echo "✅ Application is running!"
    echo ""
    echo "🌐 Access the application at: http://localhost:8000"
    echo "📚 API Documentation at: http://localhost:8000/docs"
    echo ""
    echo "📊 View logs:"
    echo "   docker-compose logs -f app"
    echo ""
    echo "🛑 Stop the application:"
    echo "   docker-compose down"
else
    echo ""
    echo "❌ Failed to start services. Check logs:"
    echo "   docker-compose logs"
fi