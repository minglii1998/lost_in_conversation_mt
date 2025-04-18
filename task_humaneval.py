from typing import List, Dict, Any
import json
import copy
import random
import os
from task_humaneval_eval import evaluate_functional_correctness_from_samples
from task_base import Task
import re
import ast


class TaskHumanEval(Task):
    def __init__(self, version="0.2"):
        self.version = version
        with open("prompts/humaneval/humaneval_full_prompt.txt", "r") as f:
            self.fully_specified_prompt = f.read()
        with open("prompts/humaneval/humaneval_system_prompt.txt", "r") as f:
            self.system_prompt = f.read()
        self.seed = 42
        random.seed(self.seed)

        self.answer_extraction_strategy = "task_specific"

    def get_dataset_file(self) -> str:
        if os.path.exists(f"data/lazy_humaneval_{self.version}.json"):
            return f"data/lazy_humaneval_{self.version}.json"
        else:
            raise ValueError(f"Version {self.version} not supported")

    def get_samples(self, filter="accepted") -> List[Dict[str, Any]]:
        with open(self.get_dataset_file(), "r") as f:
            data = json.load(f)
        if filter == "mini":
            return data[:10]
        elif filter == "accepted":
            return [d for d in data if d.get("annotation_status") == "accepted"]
        elif filter == "auto_verified":
            return [d for d in data if d.get("annotation_status") == "accepted" and d.get("is_auto_verified")]
        elif filter == "perfect":
            return [d for d in data if d.get("annotation_status") == "accepted" and d.get("is_auto_verified") and d.get("is_perfect")]
        elif filter == "imperfect":
            return [d for d in data if d.get("annotation_status") == "accepted" and d.get("is_auto_verified") and not d.get("is_perfect")]
        elif filter == "full":
            return data

    def get_task_name(self):
        return "humaneval"
    
    def get_answer_description(self) -> str:
        return ("A final answer must be a valid Python function that is defined (def ...) and returns "
                "something. If the answer is just a natural language explanation, it is not a valid "
                "answer. Special symbols like newlines and quotes must be escaped.")
    
    def generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        return self.system_prompt

    def evaluator_function(self, extracted_answer: str, sample: Dict[str, Any]) -> bool:
        pred_python_func = extracted_answer.replace("```python", "").replace("```", "")

        # Extract imports from sample["prompt"] -- this affects full
        prompt_ast = ast.parse(sample["prompt"])
        imports = []
        for node in prompt_ast.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.unparse(node))
        
        # Prepend imports to pred_python_func
        if imports:
            pred_python_func = "\n".join(imports) + "\n\n" + pred_python_func

        true_function_name = sample["entry_point"]
        # replace the function name with the true function name
        if "def " not in pred_python_func:
            return {"is_correct": False, "pass@1": 0}
        old_func_name = pred_python_func.split("def ")[1].split("(")[0].strip()
        pred_python_func = pred_python_func.replace(old_func_name, true_function_name, 1)  

        # execute the code
        sample_to_be_executed = copy.deepcopy(sample)
        sample_to_be_executed["completion"] = pred_python_func
        samples = [sample_to_be_executed]
        # Setting no_prompt to True because we are asking for the full function from the model
        # Can submit multiple samples for the same task_id, for now just one
        _, score_dict = evaluate_functional_correctness_from_samples(samples, no_prompt=True)

        return {"is_correct": int(score_dict["pass@1"]) == 1, "pass@1": score_dict["pass@1"]}

    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        user_query = sample["prompt"]
        # wrap with python code block
        return self.fully_specified_prompt.replace("[[INSTRUCTION]]", f"Complete the following incomplete function signature:\n```python\n{user_query}\n```")

    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        query = sample["initial_query"] + "\n"
        for hint in sample["hints"]:
            query += f"- {hint['hint']}\n"
        return self.fully_specified_prompt.replace("[[INSTRUCTION]]", query)

    def extract_fully_specific_response(self, response: str, sample: Dict[str, Any]) -> str:
        # FIXME(hiro): "completion" is not the best name for the field because we ask for a full function
        return response["completion"]

    def process_original_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process HumanEval sample for annotation UI display"""
        return {
            "task_id": sample["task_id"],
            "prompt": sample["prompt"],
            "entry_point": sample["entry_point"],
        }
    
    def extract_answer(self, text: str) -> str:
        """Extract the last Python function definition from text, handling both raw Python and markdown.
        
        Args:
            text: String that may contain Python code, possibly within markdown blocks
            
        Returns:
            The complete function definition as a string, or empty string if no function found
        """
        # First try to extract Python code blocks from markdown
        code_blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', text, re.DOTALL)
        
        # If we found code blocks, try each one from last to first
        if code_blocks:
            for block in reversed(code_blocks):
                result = self._extract_function_from_code(block)
                if result:
                    return result
            return ""  # No function found in any code block
            
        # If no code blocks found, try to find the last function definition in the text
        text = text.strip()
        if text.startswith("```") or text.startswith("`"):
            text = text[text.find("\n"):].strip()
        import_idx = text.rfind("import")
        def_idx = text.rfind("def")
        start_idx = import_idx if import_idx >= 0 else def_idx
        if start_idx >= 0:
            text = text[start_idx:]
        return self._extract_function_from_code(text)
    
    def _add_parent_info(self, node, parent=None):
        """Add parent information to all nodes in the AST."""
        node.parent = parent
        for child in ast.iter_child_nodes(node):
            self._add_parent_info(child, node)

    def _extract_function_from_code(self, code: str) -> str:
        """Helper method to extract function from pure Python code using AST."""
        try:
            # Parse the code
            tree = ast.parse(code)
            
            # Add parent information to all nodes
            self._add_parent_info(tree)
            
            # Find all import statements
            import_nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            imports = []
            for node in import_nodes:
                imports.append(ast.unparse(node))
            
            # Find all function definitions
            function_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            if not function_nodes:
                return ""  # No functions found
            
            # Get the last function definition that is at the top level
            last_function = None
            for node in reversed(function_nodes):
                # Check if the parent is the module (top level)
                if isinstance(node.parent, ast.Module):
                    last_function = node
                    break
            
            if not last_function:
                return ""  # No top-level functions found
            
            # Get thec source lines from the original text
            source_lines = code.splitlines()
            
            # Function spans from its first line to the last line of its body
            # If there are decorators, start from the first decorator
            start_line = (last_function.decorator_list[0].lineno - 1 
                         if last_function.decorator_list 
                         else last_function.lineno - 1)  # ast line numbers are 1-based
            end_line = last_function.end_lineno
            
            # Extract the complete function text
            function_text = '\n'.join(source_lines[start_line:end_line])
            
            # Prepend imports if any were found
            if imports:
                return '\n'.join(imports) + '\n\n' + function_text
            return function_text
        except Exception:
            return ""
        


if __name__ == "__main__":
    with open("data/lazy_humaneval_0.2.json", "r") as f:
        samples = json.load(f)

    samples = [s for s in samples if s.get("annotation_status") == "accepted"]
    task = TaskHumanEval()

    pred_python_func = "def some_random_function_name(numbers, threshold):\n    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n"

    pred_python_func = """
def process_numbers(numbers):
    valid_numbers = [num for num in numbers if 1 <= num <= 9]
    sorted_numbers = sorted(valid_numbers, reverse=True)
    number_words = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    return [number_words[num - 1] for num in sorted_numbers]""".strip()

    sample = samples[0]
    # print(sample)

    print(pred_python_func)
    # print(sample)
    print(task.evaluator_function(pred_python_func, sample))

