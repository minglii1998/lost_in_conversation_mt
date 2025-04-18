from typing import List, Dict, Any, Tuple
import json
import random
import os
import re
import pickle
import zlib
import base64
import ast

import numpy as np

from task_base import Task
from task_lcb_eval import check_correctness

class TaskLCB(Task):
    def __init__(self, version="0.1"):
        self.version = version
        with open("prompts/lcb/lcb_full_prompt.txt", "r") as f:
            self.fully_specified_prompt = f.read()
        with open("prompts/lcb/lcb_system_prompt.txt", "r") as f:
            self.system_prompt = f.read()
        self.seed = 42
        random.seed(self.seed)

        self.answer_extraction_strategy = "task_specific"

    def get_dataset_file(self) -> str:
        if os.path.exists(f"data/lazy_lcb_{self.version}.json"):
            return f"data/lazy_lcb_{self.version}.json"
        else:
            raise ValueError(f"Version {self.version} not supported")

    def get_samples(self, filter="accepted") -> List[Dict[str, Any]]:
        with open(self.get_dataset_file(), "r") as f:
            data = json.load(f)
        if filter == "mini":
            return data[:5]
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
        return "lcb"

    def get_answer_description(self) -> str:
        return ("A final answer must be a valid Python function that reads input from stdin and writes to stdout. "
                "The function should handle multiple test cases according to the problem specification. "
                "If the answer is just a natural language explanation, it is not a valid answer. "
                "Special symbols like newlines and quotes must be escaped.")

    def generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        return self.system_prompt

    def load_test_cases(self, sample):
        public_test_cases = json.loads(sample["public_test_cases"])  # type: ignore
        try:
            private_test_cases = json.loads(sample["private_test_cases"])  # type: ignore
        except:
            private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(sample["private_test_cases"].encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore

        return json.dumps(
            {
                "inputs": [
                    t["input"]
                    for t in public_test_cases + private_test_cases
                ],
                "outputs": [
                    t["output"]
                    for t in public_test_cases + private_test_cases
                ],
                "fn_name": sample["metadata"].get("func_name", None),
            }
        )

    def evaluator_function(self, extracted_answer: str, sample: Dict[str, Any]) -> bool:
        is_lazy = sample.get("is_lazy", False)

        pred_python_code = extracted_answer.replace("```python", "").replace("```", "")

        # only modify the function name if simulating lazy and the function name is provided
        # Full and concat runs will have access to the "starter code" in the prompt
        if is_lazy and sample.get("metadata", {}).get("func_name", None) is not None:
            old_func_name = pred_python_code.split("def ")[1].split("(")[0].strip()
            pred_python_code = pred_python_code.replace(old_func_name, sample["metadata"]["func_name"])

        # Extract function body for stdio problems
        if sample.get("problem_type", "") == "stdio":
            pred_python_code = self.extract_function_body(pred_python_code)

        testcases = self.load_test_cases(sample)

        output, metadata = check_correctness(sample, pred_python_code, testcases, timeout=6)
        all_test_cases_passed = all(o is True for o in output)

        return {"is_correct": all_test_cases_passed, "pass@1": 1 if all_test_cases_passed else 0, "metadata": metadata}

    def get_formatting_preamble(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """Outputs the formatting preamble and the starter code block"""
        no_starter = sample.get("no_starter", False)

        # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py#L33
        call_preamble = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
        stdio_preamble = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). "
                          "Enclose your code within delimiters as follows. "
                          "Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.")

        if no_starter or sample.get("starter_code", "") == "":
            return stdio_preamble, "```python\n# YOUR CODE HERE\n```"
        else:
            return call_preamble, f"```python\n{sample['starter_code'] if sample['starter_code'] else '# YOUR CODE HERE'}\n```"

    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        query = sample["question_content"]
        formatting_preamble, starter_code = self.get_formatting_preamble(sample)

        return self.fully_specified_prompt.replace("[[QUESTION]]", query) \
            .replace("[[FORMATTING_PREAMBLE]]", formatting_preamble) \
            .replace("[[FORMATTING]]", starter_code)

    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        query = sample["initial_query"] + "\n"
        for hint in sample["hints"]:
            query += f"- {hint['hint']}\n"

        formatting_preamble, starter_code = self.get_formatting_preamble(sample)

        return self.fully_specified_prompt.replace("[[QUESTION]]", query) \
            .replace("[[FORMATTING_PREAMBLE]]", formatting_preamble) \
            .replace("[[FORMATTING]]", starter_code)

    def extract_fully_specific_response(self, response: str, sample: Dict[str, Any]) -> str:
        return response["completion"]

    def process_original_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process LiveCodeBench sample for annotation UI display"""
        return {
            "task_id": sample["task_id"],
            "question_content": sample["question_content"],
            "starter_code": sample["starter_code"],
        }

    def extract_answer_org(self, text: str) -> str:
        """Extract the Python code from text, handling both raw Python and markdown.

        Args:
            text: String that may contain Python code, possibly within markdown blocks

        Returns:
            The complete code as a string, or empty string if no code found
        """
        # First try to extract Python code blocks from markdown
        code_blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', text, re.DOTALL)

        # If we found code blocks, use the last one
        if code_blocks:
            return code_blocks[-1].strip()

        # If no code blocks found, try to find Python code in the text
        text = text.strip()
        if text.startswith("```") or text.startswith("`"):
            text = text[text.find("\n"):].strip()
        return text

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

    def extract_function_body(self, code: str) -> str:
        """Extract the body of a function and convert it to top-level code.

        Args:
            code: String containing a Python function definition

        Returns:
            The function body as top-level code, or empty string if no function found
        """
        try:
            # Parse the code
            tree = ast.parse(code)

            # Add parent information to all nodes
            self._add_parent_info(tree)

            # Find all function definitions
            function_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

            if not function_nodes:
                print("No functions found")
                return ""  # No functions found

            # Get the last function definition that is at the top level
            last_function = None
            for node in reversed(function_nodes):
                # Check if the parent is the module (top level)
                if isinstance(node.parent, ast.Module):
                    last_function = node
                    break

            if not last_function:
                print("No top-level functions found")
                return ""  # No top-level functions found

            # Get the source lines from the original text
            source_lines = code.splitlines()

            # Function body starts after the function definition line
            start_line = last_function.lineno  # ast line numbers are 1-based
            end_line = last_function.end_lineno

            # Extract the function body text
            body_lines = source_lines[start_line:end_line]

            # Find the minimum indentation level in the body
            min_indent = float('inf')
            for line in body_lines:
                if line.strip():  # Skip empty lines
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)

            # Remove only the initial indentation level from each line
            body_lines = [line[min_indent:] if line.strip() else line for line in body_lines]

            return '\n'.join(body_lines)
        except Exception as e:
            print("Error extracting function body")
            print(e)
            return ""


if __name__ == "__main__":
    with open("data/lazy_lcb_0.1.json", "r") as f:
        samples = json.load(f)

    samples = [s for s in samples if s.get("annotation_status") == "accepted"]
    task = TaskLCB()

    pred_python_code = """
def solve():
    s = input().strip()
    target = "abc"
    if s == target:
        return "YES"
    if len(s) != 3:
        return "NO"

    # Try all possible swaps
    for i in range(3):
        for j in range(i+1, 3):
            # Make a copy and swap
            chars = list(s)
            chars[i], chars[j] = chars[j], chars[i]
            if "".join(chars) == target:
                return "YES"
    return "NO"

t = int(input())
for _ in range(t):
    print(solve())
""".strip()

    sample = samples[0]
    print(pred_python_code)
    print(task.evaluator_function(pred_python_code, sample))