"""
Tests for the MDP action definitions.
"""
import pytest
from app.core.mdp.state import PromptState
from app.core.mdp.action import (
    AddRoleAction, AddGoalAction, ModifyWorkflowAction, AddConstraintAction,
    AddExplanationAction, AddExampleAction, AdjustDetailAction,
    AddDomainKnowledgeAction, ClarifyTerminologyAction, AddRuleAction,
    SpecifyFormatAction, AddTemplateAction, StructureOutputAction,
    create_action
)

class TestMdpActions:
    """
    Tests for MDP actions.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.empty_state = PromptState("")
        
        # Create a state with some components
        self.basic_state = PromptState(
            """
            Role: Test Expert
            Task: Analyze the following content.
            
            Steps:
            - Read carefully
            - Identify key elements
            - Provide analysis
            
            Content: This is test content.
            """
        )
    
    def test_add_role_action(self):
        """Test adding or modifying a role."""
        # Add role to empty state
        action = AddRoleAction(parameters={"role_text": "Classification Expert"})
        new_state = action.apply(self.empty_state)
        
        assert "Role: Classification Expert" in new_state.text
        assert new_state.components["role"] == "Classification Expert"
        
        # Replace existing role
        action = AddRoleAction(parameters={"role_text": "Data Analyst"})
        new_state = action.apply(self.basic_state)
        
        assert "Role: Data Analyst" in new_state.text
        assert "Test Expert" not in new_state.text
        assert new_state.components["role"] == "Data Analyst"
        
        # Append to existing role
        action = AddRoleAction(parameters={"role_text": "with ML expertise", "replace": False})
        new_state = action.apply(self.basic_state)
        
        assert "Test Expert" in new_state.text
        assert "ML expertise" in new_state.text
        assert "Test Expert. with ML expertise" == new_state.components["role"]
    
    def test_add_goal_action(self):
        """Test adding or modifying a task goal."""
        # Add goal to empty state
        action = AddGoalAction(parameters={"goal_text": "Classify the sentiment of the text."})
        new_state = action.apply(self.empty_state)
        
        assert "Task: Classify the sentiment of the text." in new_state.text
        assert new_state.components["task"] == "Classify the sentiment of the text."
        
        # Replace existing goal
        action = AddGoalAction(parameters={"goal_text": "Extract the main entities."})
        new_state = action.apply(self.basic_state)
        
        assert "Task: Extract the main entities." in new_state.text
        assert "Analyze the following content" not in new_state.components["task"]
        assert new_state.components["task"] == "Extract the main entities."
    
    def test_modify_workflow_action(self):
        """Test modifying workflow steps."""
        # Add steps to empty state
        action = ModifyWorkflowAction(parameters={
            "steps": ["Read the text", "Identify entities", "Categorize entities"]
        })
        new_state = action.apply(self.empty_state)
        
        assert "Steps:" in new_state.text
        assert "- Read the text" in new_state.text
        assert "- Identify entities" in new_state.text
        assert "- Categorize entities" in new_state.text
        assert len(new_state.components["steps"]) == 3
        
        # Replace existing steps
        action = ModifyWorkflowAction(parameters={
            "steps": ["Process input", "Generate output"],
            "operation": "replace"
        })
        new_state = action.apply(self.basic_state)
        
        assert "- Process input" in new_state.text
        assert "- Generate output" in new_state.text
        assert "- Read carefully" not in new_state.text
        assert len(new_state.components["steps"]) == 2
        
        # Add a step
        action = ModifyWorkflowAction(parameters={
            "steps": ["Verify results"],
            "operation": "add"
        })
        new_state = action.apply(self.basic_state)
        
        assert "- Read carefully" in new_state.text
        assert "- Verify results" in new_state.text
        assert len(new_state.components["steps"]) == 4
    
    def test_add_constraint_action(self):
        """Test adding constraints."""
        # Add constraint to state
        action = AddConstraintAction(parameters={
            "constraint_text": "Ensure output is concise and focused."
        })
        new_state = action.apply(self.basic_state)
        
        assert "Constraint" in new_state.text
        assert "concise and focused" in new_state.text
        assert "constraints" in new_state.components
        assert "concise and focused" in str(new_state.components["constraints"])
    
    def test_add_explanation_action(self):
        """Test adding explanations."""
        # Add explanation for task
        action = AddExplanationAction(parameters={
            "explanation_text": "This helps understand the sentiment of the text.",
            "target": "task"
        })
        new_state = action.apply(self.basic_state)
        
        assert "This helps understand" in new_state.text
        assert "sentiment" in new_state.components["task"]
        
        # Add explanation for step
        action = AddExplanationAction(parameters={
            "explanation_text": "Look for specific terms and phrases.",
            "target": "step",
            "target_index": 1
        })
        new_state = action.apply(self.basic_state)
        
        assert "specific terms and phrases" in new_state.text
        assert any("specific terms" in step for step in new_state.components["steps"])
    
    def test_add_example_action(self):
        """Test adding examples."""
        # Add example
        action = AddExampleAction(parameters={
            "example_text": "Input: 'I love this product.' Output: Positive",
            "example_type": "input_output"
        })
        new_state = action.apply(self.basic_state)
        
        # Check that either "Example:" or "Examples:" appears in the text (accommodates both singular and plural forms)
        assert "Example:" in new_state.text or "Examples:" in new_state.text
        assert "Input: 'I love this product.'" in new_state.text
        assert "examples" in new_state.components
        assert any("love this product" in str(example) for example in new_state.components["examples"])
    
    def test_adjust_detail_action(self):
        """Test adjusting level of detail."""
        # Increase detail
        action = AdjustDetailAction(parameters={
            "direction": "increase",
            "target": "steps",
            "adjustment_text": "paying special attention to context"
        })
        new_state = action.apply(self.basic_state)
        
        assert "paying special attention to context" in new_state.text
        assert "special attention" in str(new_state.components["steps"])
        
        # Decrease detail
        basic_state_with_long_steps = PromptState(
            """
            Role: Test Expert
            Task: Analyze the following content.
            
            Steps:
            - Read carefully and make sure to understand the entire context. Take notes if necessary.
            - Identify key elements including entities, relationships, and main concepts in detail.
            - Provide analysis with thorough explanations.
            
            Content: This is test content.
            """
        )
        
        action = AdjustDetailAction(parameters={
            "direction": "decrease",
            "target": "steps"
        })
        new_state = action.apply(basic_state_with_long_steps)
        
        # Should be simplified
        assert len(new_state.text) < len(basic_state_with_long_steps.text)
        assert len(str(new_state.components["steps"])) < len(str(basic_state_with_long_steps.components["steps"]))
    
    def test_add_domain_knowledge_action(self):
        """Test adding domain knowledge."""
        # Add domain knowledge
        action = AddDomainKnowledgeAction(parameters={
            "knowledge_text": "Sentiment can be classified as positive, negative, or neutral.",
            "domain": "sentiment_analysis"
        })
        new_state = action.apply(self.basic_state)
        
        assert "Domain Knowledge" in new_state.text
        assert "positive, negative, or neutral" in new_state.text
        assert "domain_knowledge" in new_state.components
        # 验证domain_knowledge是列表且包含正确的text和domain
        assert isinstance(new_state.components["domain_knowledge"], list)
        domain_knowledge_item = new_state.components["domain_knowledge"][0]
        assert "text" in domain_knowledge_item
        assert "domain" in domain_knowledge_item
        assert domain_knowledge_item["domain"] == "sentiment_analysis"
        assert "positive, negative, or neutral" in domain_knowledge_item["text"]
    
    def test_clarify_terminology_action(self):
        """Test clarifying terminology."""
        # Add terminology clarification
        action = ClarifyTerminologyAction(parameters={
            "term": "NER",
            "definition": "Named Entity Recognition, a process to identify entities like people, organizations, etc."
        })
        
        # Create a state with the term
        state_with_term = PromptState(
            """
            Role: NLP Expert
            Task: Perform NER on the following text.
            
            Steps:
            - Process the text
            - Extract entities
            
            Content: This is sample text.
            """
        )
        
        new_state = action.apply(state_with_term)
        
        assert "NER" in new_state.text
        assert "Named Entity Recognition" in new_state.text
        assert "terminology" in new_state.components
        assert "NER" in new_state.components["terminology"]
    
    def test_add_rule_action(self):
        """Test adding rules."""
        # Add rule
        action = AddRuleAction(parameters={
            "rule_text": "Always consider context when determining sentiment.",
            "priority": "high"
        })
        new_state = action.apply(self.basic_state)
        
        assert "Rule" in new_state.text or "IMPORTANT" in new_state.text
        assert "context when determining sentiment" in new_state.text
        assert "rules" in new_state.components
        # 验证rules是列表且包含正确的text和priority
        assert isinstance(new_state.components["rules"], list)
        rule_item = new_state.components["rules"][0]
        assert "text" in rule_item
        assert "priority" in rule_item
        assert rule_item["priority"] == "high"
        assert "context when determining sentiment" in rule_item["text"]
    
    def test_specify_format_action(self):
        """Test specifying output format."""
        # Add output format
        action = SpecifyFormatAction(parameters={
            "format_text": "Provide the classification as 'Sentiment: [positive/negative/neutral]' followed by confidence score."
        })
        new_state = action.apply(self.basic_state)
        
        assert "Output Format:" in new_state.text
        assert "Sentiment: [positive/negative/neutral]" in new_state.text
        assert "confidence score" in new_state.text
        assert new_state.components["output_format"] and "Sentiment:" in new_state.components["output_format"]
    
    def test_add_template_action(self):
        """Test adding output templates."""
        # Add template
        action = AddTemplateAction(parameters={
            "template_text": "Sentiment: Positive\nConfidence: 0.92\nEvidence: 'love', 'great', 'excellent'",
            "template_type": "example"
        })
        new_state = action.apply(self.basic_state)
        
        assert "Output Example:" in new_state.text
        assert "Sentiment: Positive" in new_state.text
        assert "Confidence: 0.92" in new_state.text
        assert "templates" in new_state.components
        # 验证templates是列表且包含正确的text和type
        assert isinstance(new_state.components["templates"], list)
        template_item = new_state.components["templates"][0]
        assert "text" in template_item
        assert "type" in template_item
        assert template_item["type"] == "example"
        assert "Positive" in template_item["text"]
        assert "Confidence: 0.92" in template_item["text"]
    
    def test_structure_output_action(self):
        """Test structuring output."""
        # Add output structure
        action = StructureOutputAction(parameters={
            "structure_type": "sections",
            "elements": ["Sentiment", "Confidence", "Evidence", "Explanation"]
        })
        new_state = action.apply(self.basic_state)
        
        assert "structure" in new_state.text.lower()
        assert "Sentiment" in new_state.text
        assert "Confidence" in new_state.text
        assert "Evidence" in new_state.text
        assert "output_structure" in new_state.components
        # 验证output_structure是字典且包含正确的type和elements
        assert isinstance(new_state.components["output_structure"], dict)
        assert "type" in new_state.components["output_structure"]
        assert "elements" in new_state.components["output_structure"]
        assert new_state.components["output_structure"]["type"] == "sections"
        assert "Sentiment" in new_state.components["output_structure"]["elements"]
    
    def test_create_action_function(self):
        """Test the action creation utility function."""
        # Create role action
        action = create_action("add_role", parameters={"role_text": "Test Role"})
        assert isinstance(action, AddRoleAction)
        assert action.parameters["role_text"] == "Test Role"
        
        # Create goal action
        action = create_action("add_goal", parameters={"goal_text": "Test Goal"})
        assert isinstance(action, AddGoalAction)
        assert action.parameters["goal_text"] == "Test Goal"
        
        # Create invalid action
        with pytest.raises(ValueError):
            create_action("invalid_action")