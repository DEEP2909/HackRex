# LLM-Powered Intelligent Query-Retrieval System

A production-ready document processing and intelligent query system that combines semantic search with LLM-powered contextual analysis for insurance, legal, HR, and compliance domains.

## 🚀 Features

- **Multi-format Document Processing**: PDF, DOCX, HTML, Email, and plain text
- **Semantic Search**: FAISS and Pinecone vector databases for fast similarity search
- **LLM Integration**: OpenAI GPT-4 and Anthropic Claude support
- **Domain-Specific Intelligence**: Specialized prompts for Insurance, Legal, HR, and Compliance
- **Clause Matching**: Advanced pattern recognition for legal and insurance clauses  
- **Explainable Results**: Detailed rationale and confidence scoring
- **Production Ready**: Docker containerization, async processing, comprehensive logging
- **RESTful API**: FastAPI with automatic documentation
- **Analytics**: Query tracking and performance metrics

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client App    │───▶│   FastAPI API    │───▶│   Document      │
└─────────────────┘    └──────────────────┘    │   Processor     │
                                │               └─────────────────┘
                                ▼                        │
┌─────────────────┐    ┌──────────────────┐             ▼
│   PostgreSQL    │◀───│  Database        │    ┌─────────────────┐
│   Database      │    │  Manager         │    │   Embedding     │
└─────────────────┘    └──────────────────┘    │   Engine        │
                                │               └─────────────────┘
                                ▼                        │
┌─────────────────┐    ┌──────────────────┐             ▼
│   OpenAI/       │◀───│   LLM Query      │    ┌─────────────────┐
│   Anthropic     │    │   Processor      │    │   FAISS/        │
└─────────────────┘    └──────────────────┘    │   Pinecone      │
                                               └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (optional)
- OpenAI API key or Anthropic API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd llm-query-retrieval-system
   ```

2. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the database**
   ```bash
   python -c "from src.database.manager import DatabaseManager; import asyncio; asyncio.run(DatabaseManager().initialize())"
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

### Docker Deployment

1. **Using Docker Compose**
   ```bash
   docker-compose up --build
   ```

This will start:
- The main application on port 8000
- PostgreSQL database on port 5432
- Redis for caching on port 6379
- Nginx reverse proxy on port 80

## 📚 API Usage

### Main Query Endpoint (Problem Statement Specification)

```http
POST /api/v1/hackrx/run
Content-Type: application/json
Authorization: Bearer your-token

{
  "documents": ["https://example.com/policy.pdf"],
  "questions": [
    "Does this policy cover knee surgery, and what are the conditions?",
    "What is the waiting period for pre-existing conditions?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "Yes, the policy covers knee surgery under the surgical benefits section. Coverage includes both arthroscopic and open knee procedures with a deductible of $500. Pre-authorization is required for all knee surgeries.",
    "The waiting period for pre-existing conditions is 24 months from the policy effective date. However, conditions disclosed and accepted at the time of application may have a reduced waiting period of 12 months."
  ]
}
```

### Document Upload

```http
POST /upload-document
Content-Type: multipart/form-data

file: <document-file>
domain: insurance
language: en
```

### Advanced Query Processing

```http
POST /api/v1/query/detailed
Content-Type: application/json

{
  "query": "What are the liability limitations in this contract?",
  "domain": "legal",
  "max_chunks": 10,
  "include_rationale": true,
  "temperature": 0.1
}
```

## 🏢 Domain-Specific Features

### Insurance
- Policy coverage analysis
- Claims procedures and requirements
- Premium and payment terms
- Exclusions and waiting periods
- Regulatory compliance checking

### Legal
- Contract term analysis
- Liability and indemnification review
- Termination and dispute resolution clauses
- Governing law identification
- Risk assessment

### HR
- Employment policy interpretation
- Benefits and compensation analysis
- Performance management procedures
- Compliance with labor laws
- Workplace safety requirements

### Compliance
- Regulatory requirement identification
- Audit procedure documentation
- Risk assessment and mitigation
- Reporting obligation tracking
- Training requirement analysis

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# API Keys (required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname

# Vector Database (optional)
PINECONE_API_KEY=...
PINECONE_ENV=us-west1-gcp

# Processing
MAX_CHUNK_SIZE=500
SIMILARITY_THRESHOLD=0.5
MAX_CHUNKS_PER_QUERY=10
```

### Model Configuration

The system supports multiple embedding and LLM models:

- **Embedding Models**: sentence-transformers models
- **LLM Models**: GPT-4, GPT-3.5-turbo, Claude-3-sonnet
- **Vector Databases**: FAISS (local), Pinecone (cloud)

## 📊 Performance & Scalability

### Benchmarks

- **Document Processing**: 1-5 seconds per PDF (depends on size)
- **Query Response Time**: 2-8 seconds (includes LLM processing)
- **Throughput**: 100+ concurrent queries with proper scaling
- **Accuracy**: 90%+ confidence on domain-specific queries

### Scaling Options

1. **Horizontal Scaling**: Multiple API instances behind load balancer
2. **Database Scaling**: PostgreSQL read replicas, connection pooling
3. **Vector Database**: Pinecone for cloud-scale vector operations
4. **Caching**: Redis for frequently accessed results
5. **Async Processing**: Background document processing

## 🧪 Testing

Run the test suite:

```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/

# API tests
pytest tests/api/

# Coverage report
pytest --cov=src tests/
```

## 📁 Project Structure

```
llm-query-retrieval-system/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
├── docker-compose.yml     # Multi-service deployment
├── config/
│   └── settings.py        # Configuration management
├── src/
│   ├── models/            # Data models and schemas
│   ├── processors/        # Document and query processing
│   ├── database/          # Database management
│   ├── api/              # API routes and middleware
│   └── utils/            # Utility functions
├── tests/                # Test suite
├── data/                 # Data storage
└── logs/                 # Application logs
```

## 🛡️ Security

- **API Authentication**: Bearer token support
- **Input Validation**: Comprehensive request validation
- **File Upload Security**: Type and size restrictions
- **Rate Limiting**: Configurable request limits
- **SQL Injection Protection**: Parameterized queries
- **CORS Configuration**: Configurable origins

## 📈 Monitoring & Analytics

### Built-in Analytics

- Query performance metrics
- Confidence score tracking
- Domain usage statistics
- Error rate monitoring
- User feedback collection

### Health Checks

```http
GET /health
```

Returns system status and component health.

## 🚨 Error Handling

The system provides comprehensive error handling with:

- Structured error responses
- Detailed logging
- Graceful degradation
- Retry mechanisms for external APIs
- Circuit breaker patterns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:

- Create an issue in the GitHub repository
- Check the documentation at `/docs`
- Review the API documentation at `/docs` (when running)

## 🎯 Roadmap

- [ ] Multi-language document support
- [ ] Advanced OCR integration
- [ ] Real-time document collaboration
- [ ] Custom model fine-tuning
- [ ] Advanced analytics dashboard
- [ ] GraphQL API support
- [ ] Mobile SDK

---

Built with ❤️ for intelligent document processing and query retrieval.
