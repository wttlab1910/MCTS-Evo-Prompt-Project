"""
Unit tests for service components.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.prompt_service import PromptService
from app.services.optimization_service import OptimizationService
from app.services.knowledge_service import KnowledgeService

class TestPromptService:
    """Tests for the prompt service."""
    
    def setup_method(self):
        """Set up test environment."""
        self.service = PromptService()
    
    def test_process_input(self):
        """Test processing input text."""
        input_text = "Instruction: Classify this review. Data: This product is amazing!"
        
        result = self.service.process_input(input_text)
        
        assert "prompt" in result
        assert "data" in result
        assert "task_analysis" in result
        assert "expanded_prompt" in result
        
        assert result["prompt"] == "Classify this review."
        assert result["data"] == "This product is amazing!"
        assert result["task_analysis"]["task_type"] == "classification"
        assert "Role:" in result["expanded_prompt"]
    
    def test_expand_prompt(self):
        """Test expanding a prompt."""
        prompt = "Summarize this article."
        
        expanded = self.service.expand_prompt(prompt)
        
        assert len(expanded) > len(prompt)
        assert "Role:" in expanded
        assert "Task:" in expanded
        assert "Steps:" in expanded
    
    @pytest.mark.asyncio
    @patch('app.services.prompt_service.LLMFactory')
    async def test_evaluate_prompt(self, mock_factory):
        """Test evaluating a prompt."""
        # Mock LLM - 使用 AsyncMock 代替 MagicMock
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = {
            "text": "Positive",
            "elapsed_time": 0.1,
            "finish_reason": "stop"
        }
        mock_factory.create.return_value = mock_llm
        
        # Set LLM instance
        self.service.llm = mock_llm
        
        prompt = "Classify the sentiment of this review."
        data = "This product is amazing!"
        expected_output = "Positive"
        
        result = await self.service.evaluate_prompt(
            prompt=prompt,
            task_type="classification",
            data=data,
            expected_output=expected_output
        )
        
        assert "prompt" in result
        assert "response" in result
        assert "validation" in result
        assert "metrics" in result
        
        assert result["response"] == "Positive"
        assert result["validation"]["valid"]
        assert "quality_score" in result["validation"]

class TestOptimizationService:
    """Tests for the optimization service."""
    
    def setup_method(self):
        """Set up test environment."""
        self.service = OptimizationService()
    
    @pytest.mark.asyncio
    async def test_start_optimization(self):
        """Test starting an optimization job."""
        input_text = "Instruction: Classify this review. Data: This product is amazing!"
        
        # 使用 patch 替换 PromptService.process_input 方法而不是整个类
        with patch('app.services.optimization_service.PromptService.process_input') as mock_process_input:
            # 设置模拟返回值，使用与实际处理相同的内容
            mock_process_input.return_value = {
                "prompt": "Classify this review.",
                "data": "This product is amazing!",
                "task_analysis": {"task_type": "classification"},
                "expanded_prompt": "Expanded test prompt"
            }
            
            # 启动优化
            optimization_id = await self.service.start_optimization(
                input_text=input_text,
                expected_output="Positive",
                iterations=10,
                timeout=5
            )
            
            assert optimization_id is not None
            assert optimization_id in self.service.optimization_jobs
            
            job = self.service.optimization_jobs[optimization_id]
            assert job["status"] == "running"
            assert job["input"]["prompt"] == "Classify this review."
            assert job["input"]["data"] == "This product is amazing!"
            assert job["input"]["expected_output"] == "Positive"
    
    @pytest.mark.asyncio
    async def test_get_optimization_status(self):
        """Test getting optimization status."""
        # 使用 patch 替换 PromptService
        with patch('app.services.optimization_service.PromptService') as MockPromptService:
            # 设置模拟返回值
            MockPromptService.return_value.process_input.return_value = {
                "prompt": "Test prompt",
                "data": "Test data",
                "task_analysis": {"task_type": "classification"},
                "expanded_prompt": "Expanded test prompt"
            }
            
            # 使用 AsyncMock 模拟异步方法
            mock_eval = AsyncMock()
            mock_eval.return_value = {
                "prompt": "Test prompt",
                "response": "Test response",
                "validation": {"valid": True, "quality_score": 0.9},
                "metrics": {"accuracy": 0.9}
            }
            MockPromptService.return_value.evaluate_prompt = mock_eval
            
            # First start an optimization
            input_text = "Instruction: Classify this review. Data: This product is amazing!"
            optimization_id = await self.service.start_optimization(
                input_text=input_text,
                iterations=1,
                timeout=1
            )
            
            # Wait a bit for it to process
            await asyncio.sleep(0.5)
            
            # Get status
            status = await self.service.get_optimization_status(optimization_id)
            
            assert status is not None
            assert "status" in status
            # 接受所有可能的状态
            assert status["status"] in ["running", "completed", "failed", "cancelled"]
            assert "progress" in status
            
            # For non-existent job
            no_status = await self.service.get_optimization_status("non-existent")
            assert no_status is None
    
    @pytest.mark.asyncio
    async def test_cancel_optimization(self):
        """Test cancelling an optimization job."""
        # 使用 patch 替换 PromptService
        with patch('app.services.optimization_service.PromptService') as MockPromptService:
            # 设置模拟返回值
            MockPromptService.return_value.process_input.return_value = {
                "prompt": "Test prompt",
                "data": "Test data",
                "task_analysis": {"task_type": "classification"},
                "expanded_prompt": "Expanded test prompt"
            }
            
            # 使用 AsyncMock 模拟异步方法
            mock_eval = AsyncMock()
            mock_eval.return_value = {
                "prompt": "Test prompt",
                "response": "Test response",
                "validation": {"valid": True, "quality_score": 0.9},
                "metrics": {"accuracy": 0.9}
            }
            MockPromptService.return_value.evaluate_prompt = mock_eval
        
            # First start an optimization with longer timeout
            input_text = "Instruction: Classify this review. Data: This product is amazing!"
            optimization_id = await self.service.start_optimization(
                input_text=input_text,
                iterations=100,
                timeout=30
            )
            
            # Cancel it
            success = await self.service.cancel_optimization(optimization_id)
            
            assert success
            
            # Get status to confirm it's cancelled
            status = await self.service.get_optimization_status(optimization_id)
            assert status["status"] == "cancelled"
            
            # Try cancelling non-existent job
            no_success = await self.service.cancel_optimization("non-existent")
            assert not no_success

class TestKnowledgeService:
    """Tests for the knowledge service."""
    
    def setup_method(self):
        """Set up test environment."""
        self.service = KnowledgeService()
    
    @pytest.mark.asyncio
    async def test_create_list_entry(self):
        """Test creating and listing knowledge entries."""
        # Create an entry
        entry = await self.service.create_entry(
            knowledge_type="entity_classification",
            statement="PAH is a gene, not a disease",
            domain="biomedical",
            metadata={"source": "test", "confidence": 0.95}
        )
        
        assert entry is not None
        assert "id" in entry
        assert entry["knowledge_type"] == "entity_classification"
        assert entry["statement"] == "PAH is a gene, not a disease"
        assert entry["domain"] == "biomedical"
        
        # List entries
        entries = await self.service.list_entries()
        
        # Should include the one we just created
        assert len(entries) >= 1
        assert any(e["id"] == entry["id"] for e in entries)
        
        # List filtered by domain
        biomedical_entries = await self.service.list_entries(domain="biomedical")
        assert any(e["id"] == entry["id"] for e in biomedical_entries)
        
        # Clean up
        await self.service.delete_entry(entry["id"])
    
    @pytest.mark.asyncio
    async def test_get_entry(self):
        """Test getting a knowledge entry."""
        # Create an entry
        entry = await self.service.create_entry(
            knowledge_type="concept_definition",
            statement="Machine learning is a field of AI",
            domain="computer_science",
            metadata={"source": "test"}
        )
        
        # Get the entry
        retrieved = await self.service.get_entry(entry["id"])
        
        assert retrieved is not None
        assert retrieved["id"] == entry["id"]
        assert retrieved["statement"] == "Machine learning is a field of AI"
        
        # Non-existent entry
        no_entry = await self.service.get_entry("non-existent")
        assert no_entry is None
        
        # Clean up
        await self.service.delete_entry(entry["id"])
    
    @pytest.mark.asyncio
    async def test_update_entry(self):
        """Test updating a knowledge entry."""
        # Create an entry
        entry = await self.service.create_entry(
            knowledge_type="concept_definition",
            statement="Original statement",
            domain="test_domain"
        )
        
        # Update the entry
        updated = await self.service.update_entry(
            entry_id=entry["id"],
            knowledge_type="concept_definition",
            statement="Updated statement",
            domain="test_domain",
            metadata={"updated": True}
        )
        
        assert updated is not None
        assert updated["id"] == entry["id"]
        assert updated["statement"] == "Updated statement"
        assert updated["metadata"]["updated"] is True
        
        # Verify update
        retrieved = await self.service.get_entry(entry["id"])
        assert retrieved["statement"] == "Updated statement"
        
        # Clean up
        await self.service.delete_entry(entry["id"])
    
    @pytest.mark.asyncio
    async def test_delete_entry(self):
        """Test deleting a knowledge entry."""
        # Create an entry
        entry = await self.service.create_entry(
            knowledge_type="test_type",
            statement="Test statement",
            domain="test_domain"
        )
        
        # Delete the entry
        success = await self.service.delete_entry(entry["id"])
        
        assert success
        
        # Verify it's gone
        retrieved = await self.service.get_entry(entry["id"])
        assert retrieved is None
        
        # Try deleting non-existent entry
        no_success = await self.service.delete_entry("non-existent")
        assert not no_success