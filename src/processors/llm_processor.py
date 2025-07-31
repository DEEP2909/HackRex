"""
LLM Query Processor for intelligent query understanding and response generation
"""

import json
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
import openai
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from ..models.query import QueryResult, QueryType, Domain
from ..utils.logger import get_logger
from config.settings import Settings

logger = get_logger(__name__)

class LLMQueryProcessor:
    """Handles LLM interactions for query processing and decision making"""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.settings = Settings()
        
        # Initialize API clients
        self.openai_client = None
        self.anthropic_client = None
        
        # Model configuration
        self.model_config = {
            'temperature': 0.1,
            'max_tokens': 2000,
            'top_p': 0.95
        }
        
        # Domain-specific prompts
        self.domain_prompts = {
            Domain.INSURANCE: self._get_insurance_prompt(),
            Domain.LEGAL: self._get_legal_prompt(),
            Domain.HR: self._get_hr_prompt(),
            Domain.COMPLIANCE: self._get_compliance_prompt(),
            Domain.GENERAL: self._get_general_prompt()
        }
        
        # Query type patterns
        self.query_type_patterns = {
            QueryType.FACTUAL: ['what is', 'define', 'explain', 'describe'],
            QueryType.COMPARATIVE: ['compare', 'difference', 'versus', 'vs', 'better'],
            QueryType.ANALYTICAL: ['analyze', 'evaluate', 'assess', 'why', 'how'],
            QueryType.PROCEDURAL: ['how to', 'steps', 'process', 'procedure'],
            QueryType.COMPLIANCE: ['compliant', 'regulation', 'requirement', 'rule']
        }
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize API clients"""
        try:
            if self.settings.openai_api_key:
                openai.api_key = self.settings.openai_api_key
                self.openai_client = openai
                logger.info("OpenAI client initialized")
            
            if self.settings.anthropic_api_key:
                self.anthropic_client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
                logger.info("Anthropic client initialized")
            
            if not self.openai_client and not self.anthropic_client:
                raise ValueError("No LLM API keys provided")
                
        except Exception as e:
            logger.error(f"Error initializing LLM clients: {str(e)}")
            raise
    
    def _get_insurance_prompt(self) -> str:
        """Get insurance domain-specific prompt"""
        return """You are an expert insurance policy analyst with deep knowledge of insurance products, coverage details, and claims procedures. 

When analyzing insurance documents and answering questions:

1. **Coverage Analysis**: Carefully examine policy terms, coverage limits, deductibles, and exclusions
2. **Claims Procedures**: Explain claim filing processes, required documentation, and timelines
3. **Premium Details**: Clarify payment schedules, grace periods, and renewal terms
4. **Exclusions**: Clearly identify what is NOT covered and under what circumstances
5. **Waiting Periods**: Explain any waiting periods that apply to specific coverages
6. **Legal Requirements**: Reference applicable insurance regulations when relevant

Always provide:
- Specific clause references with page numbers when available
- Clear explanations of complex insurance terminology
- Practical implications for the policyholder
- Confidence level in your analysis
- Any limitations or conditions that apply

Focus on accuracy and clarity, as insurance decisions have significant financial implications."""
    
    def _get_legal_prompt(self) -> str:
        """Get legal domain-specific prompt"""
        return """You are a legal document analyst with expertise in contract law, corporate agreements, and legal compliance.

When reviewing legal documents and answering questions:

1. **Contract Terms**: Analyze key provisions, obligations, rights, and remedies
2. **Liability Issues**: Identify liability limitations, indemnification clauses, and risk allocation
3. **Termination Provisions**: Explain termination rights, notice requirements, and consequences
4. **Dispute Resolution**: Detail arbitration, mediation, or litigation procedures
5. **Governing Law**: Identify applicable jurisdiction and governing law provisions
6. **Compliance Requirements**: Highlight regulatory compliance obligations

Always provide:
- Exact clause citations and section references
- Plain language explanations of legal concepts
- Potential risks and implications
- Recommendations for further legal review when appropriate
- Clear distinction between analysis and legal advice

Remember: Provide analysis, not legal advice. Recommend consulting qualified legal counsel for specific legal matters."""
    
    def _get_hr_prompt(self) -> str:
        """Get HR domain-specific prompt"""
        return """You are an HR policy expert specializing in employment law, benefits administration, and workplace procedures.

When analyzing HR documents and answering questions:

1. **Employment Terms**: Review job descriptions, compensation, benefits, and working conditions
2. **Policy Compliance**: Ensure alignment with labor laws and employment regulations
3. **Benefits Analysis**: Explain eligibility, enrollment, and benefit calculations
4. **Performance Management**: Detail evaluation processes, improvement plans, and disciplinary procedures
5. **Leave Policies**: Clarify vacation, sick leave, FMLA, and other time-off policies
6. **Workplace Procedures**: Explain grievance processes, harassment policies, and safety requirements

Always provide:
- Clear explanations of employee rights and responsibilities
- Reference to applicable employment laws (FLSA, ADA, Title VII, etc.)
- Practical guidance for both employees and managers
- Attention to diversity, equity, and inclusion considerations
- Compliance with federal, state, and local employment laws

Focus on creating fair, compliant, and employee-friendly interpretations while protecting organizational interests."""
    
    def _get_compliance_prompt(self) -> str:
        """Get compliance domain-specific prompt"""
        return """You are a compliance specialist with expertise in regulatory requirements, audit procedures, and risk management.

When analyzing compliance documents and answering questions:

1. **Regulatory Requirements**: Identify applicable regulations and compliance obligations
2. **Risk Assessment**: Evaluate compliance risks and potential consequences of non-compliance
3. **Audit Procedures**: Explain audit requirements, documentation needs, and reporting obligations
4. **Monitoring Systems**: Detail ongoing compliance monitoring and reporting processes
5. **Remediation Plans**: Outline corrective actions for compliance deficiencies
6. **Training Requirements**: Identify staff training and certification needs

Always provide:
- Specific regulatory citations and requirements
- Clear compliance timelines and deadlines
- Risk levels and potential penalties for non-compliance
- Practical implementation guidance
- Documentation and record-keeping requirements

Focus on proactive compliance management and risk mitigation strategies."""
    
    def _get_general_prompt(self) -> str:
        """Get general domain prompt"""
        return """You are an intelligent document analyst capable of understanding and interpreting various types of business documents.

When analyzing documents and answering questions:

1. **Document Understanding**: Carefully read and interpret the provided content
2. **Context Analysis**: Consider the business context and implications
3. **Accurate Information**: Provide precise, fact-based responses
4. **Clear Communication**: Use clear, professional language
5. **Comprehensive Coverage**: Address all aspects of the question
6. **Source Attribution**: Reference specific sections or clauses when possible

Always provide:
- Direct answers to the questions asked
- Supporting evidence from the documents
- Clear explanations of complex concepts
- Confidence levels in your responses
- Acknowledgment of any limitations or uncertainties

Focus on providing helpful, accurate, and actionable information."""
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query being asked"""
        query_lower = query.lower()
        
        type_scores = {}
        for query_type, patterns in self.query_type_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            type_scores[query_type] = score
        
        # Return the type with the highest score, or FACTUAL as default
        if any(type_scores.values()):
            return max(type_scores, key=type_scores.get)
        return QueryType.FACTUAL
    
    def _prepare_context(self, relevant_chunks: List[Tuple[Dict[str, Any], float]], max_context_length: int = 4000) -> str:
        """Prepare context from relevant chunks with length limiting"""
        context_parts = []
        current_length = 0
        
        for chunk_data, score in relevant_chunks:
            content = chunk_data.get('content', '')
            chunk_info = f"[Relevance: {score:.3f}] {content}"
            
            # Check if adding this chunk would exceed the limit
            if current_length + len(chunk_info) > max_context_length:
                if context_parts:  # If we have some context already, break
                    break
                else:  # If this is the first chunk and it's too long, truncate it
                    chunk_info = chunk_info[:max_context_length]
            
            context_parts.append(chunk_info)
            current_length += len(chunk_info)
        
        return "\n\n".join(context_parts)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _call_openai(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        """Call OpenAI API with retry logic"""
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.model_config['max_tokens'],
                top_p=self.model_config['top_p']
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _call_anthropic(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        """Call Anthropic API with retry logic"""
        try:
            # Convert messages to Anthropic format
            system_message = ""
            user_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)
            
            # Combine system message with user message
            if system_message and user_messages:
                combined_content = f"{system_message}\n\n{user_messages[0]['content']}"
            else:
                combined_content = user_messages[0]['content'] if user_messages else ""
            
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=self.model_config['max_tokens'],
                temperature=temperature,
                messages=[{"role": "user", "content": combined_content}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API call failed: {str(e)}")
            raise
    
    async def _call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        """Call available LLM API"""
        # Try OpenAI first, then Anthropic
        if self.openai_client:
            try:
                return await self._call_openai(messages, temperature)
            except Exception as e:
                logger.warning(f"OpenAI call failed, trying Anthropic: {str(e)}")
                if self.anthropic_client:
                    return await self._call_anthropic(messages, temperature)
                raise
        elif self.anthropic_client:
            return await self._call_anthropic(messages, temperature)
        else:
            raise ValueError("No LLM client available")
    
    async def process_query(
        self,
        query: str,
        relevant_chunks: List[Tuple[Dict[str, Any], float]],
        domain: str,
        temperature: float = 0.1
    ) -> QueryResult:
        """Process query using LLM with relevant context"""
        try:
            start_time = time.time()
            
            # Detect query type
            query_type = self._detect_query_type(query)
            
            # Convert domain string to enum
            domain_enum = Domain(domain) if domain in [d.value for d in Domain] else Domain.GENERAL
            
            # Prepare context from relevant chunks
            context = self._prepare_context(relevant_chunks)
            
            # Get domain-specific prompt
            system_prompt = self.domain_prompts.get(domain_enum, self.domain_prompts[Domain.GENERAL])
            
            # Construct the full prompt
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"""
Context from relevant documents:
{context}

Question: {query}

Please provide a comprehensive answer with the following JSON structure:
{{
    "answer": "Direct, comprehensive answer to the question",
    "evidence": "Supporting evidence and reasoning from the documents",
    "references": ["specific clause or section references"],
    "caveats": "Important limitations, conditions, or exceptions",
    "confidence": 0.95,
    "implications": "Practical implications or next steps",
    "related_topics": ["related topics that might be relevant"]
}}

Ensure your response is a valid JSON object. Be thorough but concise, and always base your answer on the provided context.
"""
                }
            ]
            
            # Call LLM
            response_text = await self._call_llm(messages, temperature)
            
            # Parse JSON response
            try:
                result_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                # Fallback: create structured response from raw text
                result_data = {
                    "answer": response_text,
                    "evidence": "LLM response could not be parsed as structured JSON",
                    "references": [],
                    "caveats": "Response format may be incomplete",
                    "confidence": 0.7,
                    "implications": "",
                    "related_topics": []
                }
            
            # Validate and clean result data
            result_data = self._validate_result_data(result_data)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create query result
            query_result = QueryResult(
                query=query,
                matched_chunks=[chunk for chunk, _ in relevant_chunks],
                confidence_score=result_data.get('confidence', 0.0),
                decision_rationale=result_data.get('evidence', ''),
                structured_response=result_data,
                processing_time=processing_time,
                query_type=query_type,
                domain=domain_enum
            )
            
            logger.info(f"Processed query in {processing_time:.2f}s with confidence {query_result.confidence_score}")
            return query_result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            # Return error result instead of raising exception
            return QueryResult(
                query=query,
                matched_chunks=[],
                confidence_score=0.0,
                decision_rationale=f"Error processing query: {str(e)}",
                structured_response={
                    "answer": "I apologize, but I encountered an error processing your query. Please try again.",
                    "evidence": str(e),
                    "references": [],
                    "caveats": "Error occurred during processing",
                    "confidence": 0.0
                },
                processing_time=0.0,
                query_type=QueryType.FACTUAL,
                domain=Domain.GENERAL
            )
    
    def _validate_result_data(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean result data"""
        # Ensure required fields exist
        required_fields = {
            'answer': 'No answer provided',
            'evidence': 'No evidence provided',
            'references': [],
            'caveats': '',
            'confidence': 0.5
        }
        
        for field, default_value in required_fields.items():
            if field not in result_data:
                result_data[field] = default_value
        
        # Validate confidence score
        confidence = result_data.get('confidence', 0.5)
        if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
            result_data['confidence'] = 0.5
        
        # Ensure references is a list
        if not isinstance(result_data.get('references'), list):
            result_data['references'] = []
        
        # Clean and validate string fields
        string_fields = ['answer', 'evidence', 'caveats']
        for field in string_fields:
            if not isinstance(result_data.get(field), str):
                result_data[field] = str(result_data.get(field, ''))
        
        return result_data
    
    async def process_bulk_queries(
        self,
        queries: List[str],
        relevant_chunks_list: List[List[Tuple[Dict[str, Any], float]]],
        domain: str,
        temperature: float = 0.1,
        max_concurrent: int = 3
    ) -> List[QueryResult]:
        """Process multiple queries concurrently"""
        try:
            # Create semaphore to limit concurrent API calls
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_single_query(query, chunks):
                async with semaphore:
                    return await self.process_query(query, chunks, domain, temperature)
            
            # Create tasks for all queries
            tasks = [
                process_single_query(query, chunks)
                for query, chunks in zip(queries, relevant_chunks_list)
            ]
            
            # Execute all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing query {i}: {str(result)}")
                    # Create error result
                    error_result = QueryResult(
                        query=queries[i],
                        matched_chunks=[],
                        confidence_score=0.0,
                        decision_rationale=f"Error: {str(result)}",
                        structured_response={"answer": "Error processing query", "confidence": 0.0},
                        processing_time=0.0
                    )
                    processed_results.append(error_result)
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in bulk query processing: {str(e)}")
            raise
    
    async def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent and extract key information"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert at analyzing user queries and extracting intent. 
                    Analyze the given query and provide insights about what the user is trying to accomplish."""
                },
                {
                    "role": "user",
                    "content": f"""
Analyze this query and provide a JSON response with the following structure:
{{
    "query_type": "factual|comparative|analytical|procedural|compliance",
    "domain": "insurance|legal|hr|compliance|general",
    "key_entities": ["list of important entities mentioned"],
    "intent": "what the user is trying to accomplish",
    "complexity": "low|medium|high",
    "requires_multiple_sources": true/false,
    "suggested_followup_questions": ["list of related questions"]
}}

Query: {query}
"""
                }
            ]
            
            response = await self._call_llm(messages, temperature=0.1)
            
            try:
                analysis = json.loads(response)
                return analysis
            except json.JSONDecodeError:
                logger.warning("Could not parse query analysis as JSON")
                return {
                    "query_type": "factual",
                    "domain": "general",
                    "key_entities": [],
                    "intent": "Unknown",
                    "complexity": "medium",
                    "requires_multiple_sources": False,
                    "suggested_followup_questions": []
                }
                
        except Exception as e:
            logger.error(f"Error analyzing query intent: {str(e)}")
            return {}
    
    async def generate_followup_questions(self, query: str, answer: str) -> List[str]:
        """Generate relevant follow-up questions based on the query and answer"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at generating relevant follow-up questions that users might want to ask after receiving an answer."
                },
                {
                    "role": "user",
                    "content": f"""
Based on this query and answer, generate 3-5 relevant follow-up questions that a user might want to ask.

Original Query: {query}

Answer: {answer}

Return a JSON array of follow-up questions:
["Question 1", "Question 2", "Question 3", ...]
"""
                }
            ]
            
            response = await self._call_llm(messages, temperature=0.3)
            
            try:
                questions = json.loads(response)
                return questions if isinstance(questions, list) else []
            except json.JSONDecodeError:
                return []
                
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {str(e)}")
            return []
    
    async def summarize_document_content(self, chunks: List[Dict[str, Any]], max_length: int = 500) -> str:
        """Generate a summary of document content from chunks"""
        try:
            # Combine chunks into context
            content = "\n\n".join([chunk.get('content', '') for chunk in chunks[:10]])  # Limit to first 10 chunks
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at summarizing document content concisely and accurately."
                },
                {
                    "role": "user",
                    "content": f"""
Summarize the following document content in {max_length} characters or less. 
Focus on the key points, main topics, and important details.

Content:
{content}
"""
                }
            ]
            
            response = await self._call_llm(messages, temperature=0.1)
            
            # Truncate if necessary
            if len(response) > max_length:
                response = response[:max_length-3] + "..."
            
            return response
            
        except Exception as e:
            logger.error(f"Error summarizing document content: {str(e)}")
            return "Unable to generate summary"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        return {
            "model_name": self.model_name,
            "temperature": self.model_config['temperature'],
            "max_tokens": self.model_config['max_tokens'],
            "has_openai": self.openai_client is not None,
            "has_anthropic": self.anthropic_client is not None,
            "supported_domains": list(self.domain_prompts.keys()),
            "supported_query_types": list(self.query_type_patterns.keys())
        }
