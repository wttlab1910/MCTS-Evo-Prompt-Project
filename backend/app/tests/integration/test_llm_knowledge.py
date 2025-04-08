"""
Tests for the Knowledge Integration with actual LLM interaction.

These tests require a running Ollama server and will make actual LLM API calls.
"""
import pytest
import asyncio
import time
from pathlib import Path
import tempfile
import os

from app.core.llm.interface import LLMInterface
from app.core.mdp.state import PromptState
from app.knowledge.knowledge_base import KnowledgeBase
from app.knowledge.domain.domain_knowledge import DomainKnowledgeManager
from app.knowledge.extraction.extractor import ConceptualKnowledgeExtractor
from app.knowledge.integration.integrator import PromptKnowledgeIntegrator
from app.config import LLM_CONFIG

# Skip all tests if SKIP_LLM_TESTS environment variable is set
pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_LLM_TESTS", "").lower() in ("true", "1", "yes"), 
    reason="Skipping tests that require LLM API calls"
)

class TestLLMIntegration:
    """Test cases for knowledge integration with actual LLM calls."""
    
    @pytest.fixture(scope="class")
    def llm_interface(self):
        """Create LLM interface for testing."""
        # Initialize LLM interface with default config
        llm = LLMInterface(config=LLM_CONFIG)
        return llm
    
    @pytest.fixture
    def knowledge_system(self):
        """Create knowledge system components."""
        # Create temporary directories for the knowledge base
        temp_dir = tempfile.TemporaryDirectory()
        domain_dir = Path(temp_dir.name) / "domain_knowledge"
        error_dir = Path(temp_dir.name) / "error_patterns"
        
        domain_dir.mkdir(exist_ok=True)
        error_dir.mkdir(exist_ok=True)
        
        # Initialize knowledge base with temp directories
        kb = KnowledgeBase(domain_dir, error_dir)
        
        # Initialize domain knowledge manager
        knowledge_manager = DomainKnowledgeManager(kb)
        
        # Initialize knowledge integrator
        integrator = PromptKnowledgeIntegrator()
        
        # Initialize extractor
        conceptual_extractor = ConceptualKnowledgeExtractor()
        
        yield {
            "kb": kb,
            "manager": knowledge_manager,
            "integrator": integrator,
            "extractor": conceptual_extractor,
            "temp_dir": temp_dir
        }
        
        # Cleanup
        temp_dir.cleanup()
    
    async def _generate_async(self, llm, prompt_text):
        """Helper to generate response asynchronously."""
        return await llm.generate_async(prompt_text)
    
    def test_basic_llm_call(self, llm_interface):
        """Test basic LLM call to ensure connection is working."""
        # Simple test prompt
        prompt_text = "What is the capital of France?"
        
        # Generate response
        response = llm_interface.generate(prompt_text)
        
        # Basic validation
        assert response is not None
        assert "text" in response
        assert len(response["text"]) > 0
        assert "Paris" in response["text"]
    
    def test_knowledge_integration_effect(self, llm_interface, knowledge_system):
        """Test how knowledge integration affects LLM responses."""
        # Setup components
        kb = knowledge_system["kb"]
        integrator = knowledge_system["integrator"]
        extractor = knowledge_system["extractor"]
        
        # Sample biomedical text for knowledge extraction
        conceptual_text = """
        The HER2 gene is defined as a proto-oncogene located on chromosome 17q21.
        It encodes a member of the epidermal growth factor receptor family of receptor tyrosine kinases.
        Amplification or overexpression of HER2 plays an important role in the development and progression
        of certain aggressive types of breast cancer.
        """
        
        # Extract knowledge
        knowledge_items = extractor.extract(conceptual_text, domain="biomedical")
        assert len(knowledge_items) > 0
        
        # Store knowledge
        stored_ids = []
        for item in knowledge_items:
            try:
                item_id = kb.add_knowledge(item)
                stored_ids.append(item_id)
            except ValueError:
                pass  # Skip duplicates
        
        # Create original prompt
        original_prompt = PromptState("""
        Task: Explain the role of HER2 in breast cancer.
        
        Instructions:
        - Provide a clear, concise explanation
        - Focus on the biological mechanism
        - Mention clinical significance if relevant
        """)
        
        # Create enhanced prompt with knowledge integration
        enhanced_prompt = original_prompt
        for item_id in stored_ids:
            knowledge = kb.get_knowledge(item_id)
            if knowledge:
                enhanced_prompt = integrator.integrate(enhanced_prompt, knowledge)
        
        # Generate responses for both prompts
        original_response = llm_interface.generate(original_prompt.text)
        enhanced_response = llm_interface.generate(enhanced_prompt.text)
        
        # Validation
        assert len(original_response["text"]) > 0
        assert len(enhanced_response["text"]) > 0
        
        # In an ideal test, we would analyze the responses for accuracy and quality
        # but for a simple test, we'll check for the presence of key terms
        assert "proto-oncogene" in enhanced_response["text"] or "chromosome 17" in enhanced_response["text"]
        
        # Print responses for manual inspection (during development)
        print("\n=== Original Prompt ===")
        print(original_prompt.text)
        print("\n=== Original Response ===")
        print(original_response["text"][:500] + "..." if len(original_response["text"]) > 500 else original_response["text"])
        
        print("\n=== Enhanced Prompt ===")
        print(enhanced_prompt.text)
        print("\n=== Enhanced Response ===")
        print(enhanced_response["text"][:500] + "..." if len(enhanced_response["text"]) > 500 else enhanced_response["text"])
    
    def test_domain_specific_knowledge(self, llm_interface, knowledge_system):
        """Test LLM responses with domain-specific knowledge integration."""
        # Setup components
        integrator = knowledge_system["integrator"]
        
        # Create domain-specific knowledge for sentiment analysis
        sentiment_knowledge = {
            "id": "k_sentiment",
            "type": "procedural_knowledge",
            "statement": "Process for accurate sentiment analysis",
            "procedure_steps": [
                "Identify subjective words and phrases",
                "Consider negations and intensifiers",
                "Pay attention to domain-specific terms",
                "Analyze context before determining sentiment",
                "Consider idiomatic expressions that may impact sentiment"
            ],
            "entities": ["sentiment analysis"],
            "metadata": {
                "source": "expert_knowledge",
                "domain": "nlp",
                "confidence": 0.9
            }
        }
        
        # Create original prompt for sentiment analysis
        original_prompt = PromptState("""
        Task: Analyze the sentiment of the following text:
        
        "The new phone has a great camera, but the battery life is terrible and the price is way too high."
        
        Instructions:
        - Determine if the sentiment is positive, negative, or mixed
        - Explain your reasoning
        """)
        
        # Create enhanced prompt with knowledge integration
        enhanced_prompt = integrator.integrate(original_prompt, sentiment_knowledge)
        
        # Generate responses for both prompts
        original_response = llm_interface.generate(original_prompt.text)
        enhanced_response = llm_interface.generate(enhanced_prompt.text)
        
        # Validation
        assert len(original_response["text"]) > 0
        assert len(enhanced_response["text"]) > 0
        
        # The enhanced response should reflect the procedural knowledge
        assert "mixed" in enhanced_response["text"].lower()
        assert "camera" in enhanced_response["text"] and "battery" in enhanced_response["text"]
        
        # Print responses for manual inspection
        print("\n=== Original Prompt (Sentiment) ===")
        print(original_prompt.text)
        print("\n=== Original Response (Sentiment) ===")
        print(original_response["text"][:500] + "..." if len(original_response["text"]) > 500 else original_response["text"])
        
        print("\n=== Enhanced Prompt (Sentiment) ===")
        print(enhanced_prompt.text)
        print("\n=== Enhanced Response (Sentiment) ===")
        print(enhanced_response["text"][:500] + "..." if len(enhanced_response["text"]) > 500 else enhanced_response["text"])
    
    @pytest.mark.asyncio
    async def test_async_llm_with_knowledge(self, llm_interface, knowledge_system):
        """Test async LLM calls with knowledge integration."""
        # Setup components
        integrator = knowledge_system["integrator"]
        
        # Create knowledge item
        coding_knowledge = {
            "id": "k_python",
            "type": "entity_classification",
            "statement": "Python is a high-level, interpreted programming language with dynamic semantics.",
            "entities": ["Python"],
            "relations": [
                {"subject": "Python", "predicate": "isA", "object": "programming language"}
            ],
            "metadata": {
                "source": "expert_knowledge",
                "domain": "programming",
                "confidence": 0.95
            }
        }
        
        # Create original prompt
        original_prompt = PromptState("""
        Task: Write a function in Python to find all prime numbers up to n.
        
        Instructions:
        - The function should be efficient
        - Include comments explaining the algorithm
        - Provide a simple example of usage
        """)
        
        # Create enhanced prompt with knowledge integration
        enhanced_prompt = integrator.integrate(original_prompt, coding_knowledge)
        
        # Generate responses asynchronously
        original_response = await self._generate_async(llm_interface, original_prompt.text)
        enhanced_response = await self._generate_async(llm_interface, enhanced_prompt.text)
        
        # Validation
        assert len(original_response["text"]) > 0
        assert len(enhanced_response["text"]) > 0
        
        # Both responses should include Python code
        assert "def" in original_response["text"] and "def" in enhanced_response["text"]
        assert "prime" in original_response["text"].lower() and "prime" in enhanced_response["text"].lower()
        
        # Print responses for manual inspection
        print("\n=== Original Response (Python) ===")
        print(original_response["text"][:500] + "..." if len(original_response["text"]) > 500 else original_response["text"])
        
        print("\n=== Enhanced Response (Python) ===")
        print(enhanced_response["text"][:500] + "..." if len(enhanced_response["text"]) > 500 else enhanced_response["text"])
    
    def test_multi_knowledge_integration(self, llm_interface, knowledge_system):
        """Test LLM response with multiple integrated knowledge items."""
        # Setup components
        integrator = knowledge_system["integrator"]
        
        # Create multiple knowledge items
        knowledge_items = [
            {
                "id": "k_sql_basics",
                "type": "conceptual_knowledge",
                "statement": "SQL (Structured Query Language) is a domain-specific language for managing data in relational databases.",
                "entities": ["SQL"],
                "relations": [
                    {"subject": "SQL", "predicate": "isA", "object": "domain-specific language"}
                ],
                "metadata": {
                    "source": "expert_knowledge",
                    "domain": "database",
                    "confidence": 0.9
                }
            },
            {
                "id": "k_sql_join",
                "type": "procedural_knowledge",
                "statement": "SQL JOIN combines rows from two or more tables based on a related column.",
                "procedure_steps": [
                    "Identify the tables to join",
                    "Determine the type of join (INNER, LEFT, RIGHT, FULL)",
                    "Specify the columns to match in the ON clause",
                    "Select the columns to include in the result set"
                ],
                "entities": ["SQL JOIN"],
                "metadata": {
                    "source": "expert_knowledge",
                    "domain": "database",
                    "confidence": 0.85
                }
            },
            {
                "id": "k_sql_performance",
                "type": "boundary_knowledge",
                "statement": "SQL query performance considerations",
                "boundary_cases": [
                    "Avoid using SELECT * to reduce data transfer",
                    "Use appropriate indexes for frequently queried columns",
                    "Consider query execution plan for complex queries",
                    "Be cautious with subqueries as they can impact performance"
                ],
                "entities": ["SQL performance"],
                "metadata": {
                    "source": "expert_knowledge",
                    "domain": "database",
                    "confidence": 0.8
                }
            }
        ]
        
        # Create original prompt
        original_prompt = PromptState("""
        Task: Write an SQL query to find the top 5 customers who have spent the most money.
        
        Database Schema:
        - customers(id, name, email)
        - orders(id, customer_id, order_date, total_amount)
        
        Instructions:
        - Write an efficient SQL query
        - Explain your approach
        """)
        
        # Create enhanced prompt with multiple knowledge items
        enhanced_prompt = original_prompt
        for item in knowledge_items:
            enhanced_prompt = integrator.integrate(enhanced_prompt, item)
        
        # Generate responses
        original_response = llm_interface.generate(original_prompt.text)
        enhanced_response = llm_interface.generate(enhanced_prompt.text)
        
        # Validation
        assert len(original_response["text"]) > 0
        assert len(enhanced_response["text"]) > 0
        
        # Enhanced response should include specific SQL best practices
        # Look for concepts from the knowledge items
        assert "SELECT" in enhanced_response["text"] and "JOIN" in enhanced_response["text"]
        assert "customers" in enhanced_response["text"] and "orders" in enhanced_response["text"]
        assert any(term in enhanced_response["text"].lower() for term in ["index", "performance", "efficient"])
        
        # Print prompts and responses for manual inspection
        print("\n=== Original Prompt (SQL) ===")
        print(original_prompt.text)
        print("\n=== Enhanced Prompt (SQL) ===")
        print(enhanced_prompt.text)
        print("\n=== Enhanced Response (SQL) ===")
        print(enhanced_response["text"][:500] + "..." if len(enhanced_response["text"]) > 500 else enhanced_response["text"])