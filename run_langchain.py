import os
import json
import argparse
from typing import List, Dict, Any
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DataLoader:
    def __init__(self, data_root_test: str, data_root_train: str, data_root_vali: str = None):
        self.data_root_test = data_root_test
        self.data_root_train = data_root_train
        self.data_root_vali = data_root_vali
        
    def load_data(self) -> tuple[Dict, List, List]:
        """Load and process data from the specified paths."""
        if 'medqa' in self.data_root_test:
            return self._load_medqa_data()
        elif 'MATH' in self.data_root_test:
            return self._load_math_data()
        else:
            return self._load_generic_data()
    
    def _load_medqa_data(self) -> tuple[Dict, List, List]:
        problems_test = [json.loads(line) for line in open(self.data_root_test, 'r')]
        problems_train = [json.loads(line) for line in open(self.data_root_train, 'r')]
        problems = problems_test + problems_train
        
        test_pids = list(range(len(problems_test)))
        train_pids = list(range(len(problems_test), len(problems_test) + len(problems_train)))
        
        if self.data_root_vali:
            problems_vali = [json.loads(line) for line in open(self.data_root_vali, 'r')]
            problems.extend(problems_vali)
            
        return problems, test_pids, train_pids
    
    def _load_math_data(self) -> tuple[Dict, List, List]:
        problems_test = self._read_json_all(self.data_root_test)
        problems_train = self._read_json_all(self.data_root_train)
        problems = problems_test + problems_train
        
        test_pids = list(range(len(problems_test)))
        train_pids = list(range(len(problems_test), len(problems_test) + len(problems_train)))
        
        return problems, test_pids, train_pids
    
    def _load_generic_data(self) -> tuple[Dict, List, List]:
        problems_test = json.load(open(self.data_root_test))
        problems_train = json.load(open(self.data_root_train))
        problems = {**problems_test, **problems_train}
        
        test_pids = list(problems_test.keys())
        train_pids = list(problems_train.keys())
        
        return problems, test_pids, train_pids
    
    @staticmethod
    def _read_json_all(file_path: str) -> List:
        """Read all JSON objects from a file."""
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f]

class LangChainRunner:
    def __init__(self, 
                 model_name: str = "gpt-4",
                 temperature: float = 0.0,
                 max_tokens: int = 512,
                 api_key: str = None,
                 api_base: str = None):
        self.model = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=api_base,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
    def get_completion(self, prompt: str, user_prompt: str = None) -> str:
        """Get completion from the model."""
        messages = [
            SystemMessage(content=prompt)
        ]
        
        if user_prompt:
            messages.append(HumanMessage(content=user_prompt))
            
        response = self.model.invoke(messages)
        return response.content.strip()
    
    def run_inference(self, 
                     problems: Dict,
                     test_pids: List,
                     shot_number: int = 2,
                     prompt_format: str = "Q-A") -> Dict[str, Any]:
        """Run inference on the test set."""
        results = {}
        correct = 0
        total = len(test_pids)
        
        for pid in test_pids:
            # Get the problem
            problem = problems[pid]
            
            # Construct prompt with few-shot examples
            prompt = self._construct_prompt(problem, problems, shot_number, prompt_format)
            
            # Get model's response
            response = self.get_completion(prompt)
            
            # Process and evaluate response
            prediction = self._extract_prediction(response)
            is_correct = self._evaluate_prediction(prediction, problem)
            
            if is_correct:
                correct += 1
                
            results[pid] = {
                "prediction": prediction,
                "correct": is_correct,
                "response": response
            }
            
        accuracy = correct / total if total > 0 else 0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }
    
    def _construct_prompt(self, 
                         problem: Dict,
                         problems: Dict,
                         shot_number: int,
                         prompt_format: str) -> str:
        """Construct the prompt with few-shot examples."""
        # Implementation depends on the specific prompt format
        # This is a placeholder implementation
        prompt = f"Question: {problem['question']}\n"
        if 'options' in problem:
            prompt += "Options:\n"
            for opt in problem['options']:
                prompt += f"{opt}\n"
        return prompt
    
    def _extract_prediction(self, response: str) -> str:
        """Extract the prediction from the model's response."""
        # Implementation depends on the expected response format
        return response.strip()
    
    def _evaluate_prediction(self, prediction: str, problem: Dict) -> bool:
        """Evaluate if the prediction is correct."""
        # Implementation depends on the evaluation criteria
        return prediction == problem.get('answer', '')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_train', type=str, required=True)
    parser.add_argument('--data_root_test', type=str, required=True)
    parser.add_argument('--data_root_vali', type=str, default=None)
    parser.add_argument('--output_root', type=str, default='results')
    parser.add_argument('--model_name', type=str, default='gpt-4')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--shot_number', type=int, default=2)
    parser.add_argument('--prompt_format', type=str, default='Q-A')
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--api_base', type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_root, exist_ok=True)
    
    # Initialize data loader
    data_loader = DataLoader(
        data_root_test=args.data_root_test,
        data_root_train=args.data_root_train,
        data_root_vali=args.data_root_vali
    )
    
    # Load data
    problems, test_pids, train_pids = data_loader.load_data()
    
    # Initialize LangChain runner
    runner = LangChainRunner(
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_key=args.api_key,
        api_base=args.api_base
    )
    
    # Run inference
    results = runner.run_inference(
        problems=problems,
        test_pids=test_pids,
        shot_number=args.shot_number,
        prompt_format=args.prompt_format
    )
    
    # Save results
    output_file = os.path.join(args.output_root, f"results_{args.model_name}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main() 