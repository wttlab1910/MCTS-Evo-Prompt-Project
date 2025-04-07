"""
Test file for verifying Ollama models setup.
"""
import asyncio
import sys
import os
from pathlib import Path
import pytest

pytestmark = pytest.mark.asyncio
# Add the project root directory to Python path
sys.path.append(str(Path(__file__).resolve().parent))

from app.llm.interface import LLMFactory
from app.llm.providers.ollama import OllamaProvider

async def test_ollama_mistral():
    """Test Ollama Mistral model."""
    try:
        print("\n===== Testing Mistral model =====")
        
        # Create LLM instance
        llm = LLMFactory.create("ollama", model_id="mistral")
        
        # Generate text
        prompt = "Write a short poem about artificial intelligence."
        print(f"Sending prompt: '{prompt}'")
        
        response = await llm.generate(prompt)
        
        # Check response
        if not response.get("text"):
            print("❌ ERROR: Response text is empty")
            return False
            
        if response.get("model") != "mistral":
            print(f"❌ ERROR: Expected model 'mistral', got '{response.get('model')}'")
            return False
            
        if response.get("finish_reason") != "stop":
            print(f"❌ ERROR: Expected finish_reason 'stop', got '{response.get('finish_reason')}'")
            return False
        
        # Print results
        print("\n=== Mistral Response ===")
        print(f"Text: {response['text']}")
        print(f"Model: {response['model']}")
        print(f"Elapsed time: {response['elapsed_time']:.2f}s")
        print(f"Finish reason: {response['finish_reason']}")
        
        print("\n✅ Mistral test PASSED!")
        return True
    except Exception as e:
        print(f"❌ ERROR testing Mistral: {e}")
        return False

async def test_ollama_gemma3():
    """Test Ollama Gemma3 model."""
    try:
        print("\n===== Testing Gemma3 model =====")
        
        # Create LLM instance
        llm = LLMFactory.create("ollama", model_id="gemma3:12b")
        
        # Generate text
        prompt = "Write a short poem about artificial intelligence."
        print(f"Sending prompt: '{prompt}'")
        
        response = await llm.generate(prompt)
        
        # Check response
        if not response.get("text"):
            print("❌ ERROR: Response text is empty")
            return False
            
        if response.get("model") != "gemma3:12b":
            print(f"❌ ERROR: Expected model 'gemma3:12b', got '{response.get('model')}'")
            return False
            
        if response.get("finish_reason") != "stop":
            print(f"❌ ERROR: Expected finish_reason 'stop', got '{response.get('finish_reason')}'")
            return False
        
        # Print results
        print("\n=== Gemma3 Response ===")
        print(f"Text: {response['text']}")
        print(f"Model: {response['model']}")
        print(f"Elapsed time: {response['elapsed_time']:.2f}s")
        print(f"Finish reason: {response['finish_reason']}")
        
        print("\n✅ Gemma3 test PASSED!")
        return True
    except Exception as e:
        print(f"❌ ERROR testing Gemma3: {e}")
        return False

async def test_ollama_deepseek():
    """Test Ollama DeepSeek model."""
    try:
        print("\n===== Testing DeepSeek model =====")
        
        # Create LLM instance
        llm = LLMFactory.create("ollama", model_id="deepseek-r1:32b")
        
        # Generate text
        prompt = "Write a short poem about artificial intelligence."
        print(f"Sending prompt: '{prompt}'")
        
        response = await llm.generate(prompt)
        
        # Check response
        if not response.get("text"):
            print("❌ ERROR: Response text is empty")
            return False
            
        if response.get("model") != "deepseek-r1:32b":
            print(f"❌ ERROR: Expected model 'deepseek-r1:32b', got '{response.get('model')}'")
            return False
            
        if response.get("finish_reason") != "stop":
            print(f"❌ ERROR: Expected finish_reason 'stop', got '{response.get('finish_reason')}'")
            return False
        
        # Print results
        print("\n=== DeepSeek Response ===")
        print(f"Text: {response['text']}")
        print(f"Model: {response['model']}")
        print(f"Elapsed time: {response['elapsed_time']:.2f}s")
        print(f"Finish reason: {response['finish_reason']}")
        
        print("\n✅ DeepSeek test PASSED!")
        return True
    except Exception as e:
        print(f"❌ ERROR testing DeepSeek: {e}")
        return False

async def run_all_tests():
    """Run all tests and report results."""
    print("🧪 TESTING OLLAMA MODELS 🧪")
    print("Make sure Ollama server is running on http://localhost:11434")
    print("==================================================")
    
    # Register Ollama provider if not already registered
    if "ollama" not in LLMFactory._providers:
        LLMFactory.register_provider("ollama", OllamaProvider)
        print("Registered Ollama provider with LLMFactory")
    
    # Run tests
    mistral_result = await test_ollama_mistral()
    gemma3_result = await test_ollama_gemma3()
    deepseek_result = await test_ollama_deepseek()
    
    # Print summary
    print("\n🔍 TEST RESULTS SUMMARY 🔍")
    print("==================================================")
    print(f"Mistral: {'✅ PASSED' if mistral_result else '❌ FAILED'}")
    print(f"Gemma3: {'✅ PASSED' if gemma3_result else '❌ FAILED'}")
    print(f"DeepSeek: {'✅ PASSED' if deepseek_result else '❌ FAILED'}")
    
    if mistral_result and gemma3_result and deepseek_result:
        print("\n🎉 ALL TESTS PASSED! Your Ollama models are set up correctly.")
    else:
        print("\n⚠️ SOME TESTS FAILED. Please check the error messages above.")
        failed_models = []
        if not mistral_result:
            failed_models.append("mistral")
        if not gemma3_result:
            failed_models.append("gemma3:12b")
        if not deepseek_result:
            failed_models.append("deepseek-r1:32b")
            
        print(f"\nMake sure the following models are installed in Ollama:")
        for model in failed_models:
            print(f"  - {model}")
        print("\nYou can install them with these commands:")
        for model in failed_models:
            print(f"  > ollama pull {model}")

if __name__ == "__main__":
    asyncio.run(run_all_tests())