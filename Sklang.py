import re
import sys

# ----------------------------------------------------
# 1. CORE COMPONENTS: Token and AST Node Structures
# ----------------------------------------------------
class Token:
    """Represents a lexical unit."""
    def __init__(self, type, value=None):
        self.type = type
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {repr(self.value) if isinstance(self.value, str) else self.value or ''})"

class ASTNode:
    """Base class for Abstract Syntax Tree nodes."""
    pass

class VarDeclNode(ASTNode):
    def __init__(self, var_name, value_expr):
        self.var_name = var_name
        self.value_expr = value_expr

class AssignmentNode(ASTNode):
    def __init__(self, var_name, value_expr):
        self.var_name = var_name
        self.value_expr = value_expr

class IfStatementNode(ASTNode):
    def __init__(self, condition, then_body):
        self.condition = condition
        self.then_body = then_body

class BinOpNode(ASTNode):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class NumberNode(ASTNode):
    def __init__(self, value):
        self.value = value
        
class IdentifierNode(ASTNode):
    def __init__(self, name):
        self.name = name

class StringNode(ASTNode):
    def __init__(self, value):
        self.value = value

# ----------------------------------------------------
# 2. LEXER (TOKENIZER)
# ----------------------------------------------------

TOKEN_SPECIFICATIONS = [
    ('NUMBER',              r'\d+\.\d+|\d+'), # Correct float/int regex
    ('ID',                  r'[a-zA-Z_][a-zA-Z0-9_]*'), 
    ('STRING',              r'".*?"'), 
    ('EQEQ',                r'=='),
    ('NEQ',                 r'!='),      # NEW: Not Equal
    ('GTE',                 r'>='),      # NEW: Greater Than or Equal
    ('LTE',                 r'<='),      # NEW: Less Than or Equal
    ('GT',                  r'>'),
    ('LT',                  r'<'),       # NEW: Less Than
    ('ASSIGN',              r'='),       
    ('PLUS',                r'\+'),      
    ('MINUS',               r'-'),       
    ('MUL',                 r'\*'),      
    ('DIV',                 r'/'),       
    ('COMMENT',             r'//[^\n]*'), 
    ('SEMI',                r';'),       
    ('LPAREN',              r'\('),      
    ('RPAREN',              r'\)'),      
    ('WHITESPACE',          r'[ \t\n\r]+'), 
    ('MISMATCH',            r'.'), 
]

KEYWORDS = {
    'SKIBIDI': 'SKIBIDI', 
    'ECHO': 'ECHO',       
    'IF': 'IF',           
    'THEN': 'THEN',       
    'ENDIF': 'ENDIF',
    'AND': 'AND',         # NEW: Boolean AND
    'OR': 'OR',           # NEW: Boolean OR
}

class Lexer:
    def __init__(self, text):
        self.text = text
        self.position = 0
        self.token_regex = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKEN_SPECIFICATIONS), re.DOTALL)
        
    def get_next_token(self):
        if self.position >= len(self.text):
            return Token('EOF')
        
        match = self.token_regex.match(self.text, self.position)
        
        if match:
            token_type = match.lastgroup
            token_value = match.group(token_type)
            self.position = match.end()

            if token_type in ('WHITESPACE', 'COMMENT'):
                return self.get_next_token()
            
            if token_type == 'ID':
                token_type = KEYWORDS.get(token_value, 'ID')
            
            if token_type == 'NUMBER':
                if '.' in token_value:
                    token_value = float(token_value)
                else:
                    token_value = int(token_value)
            
            if token_type == 'STRING': 
                token_value = token_value[1:-1]
            
            if token_type == 'MISMATCH':
                raise Exception(f"Lexical Error: Invalid character '{token_value}' at position {self.position - 1}")

            return Token(token_type, token_value)
        
        raise Exception("Lexical Error: Cannot tokenize remaining input")

# ----------------------------------------------------
# 3. PARSER (SYNTAX CHECKER & AST BUILDER)
# ----------------------------------------------------
class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.tokens = []
        self.token_index = 0
        
        current_token = self.lexer.get_next_token()
        while current_token.type != 'EOF':
             self.tokens.append(current_token)
             current_token = self.lexer.get_next_token()

    def consume(self, expected_type):
        if self.token_index < len(self.tokens) and self.tokens[self.token_index].type == expected_type:
            token = self.tokens[self.token_index]
            self.token_index += 1
            return token
        else:
            current_type = self.tokens[self.token_index].type if self.token_index < len(self.tokens) else 'EOF'
            raise Exception(f"Syntax Error: Expected {expected_type} but got {current_type}")

    def peek(self, offset=0):
        idx = self.token_index + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return Token('EOF')

    # --- NEW GRAMMAR FOR BOOLEAN LOGIC ---
    # expr -> bool_term (OR bool_term)*
    # bool_term -> comparison (AND comparison)*
    # comparison -> arith_expr (COMPARISON_OP arith_expr)?
    # ... (rest of arithmetic grammar) ...

    def factor(self):
        token = self.peek()
        if token.type == 'NUMBER':
            self.consume('NUMBER')
            return NumberNode(token.value)
        elif token.type == 'STRING':
            self.consume('STRING')
            return StringNode(token.value)
        elif token.type == 'ID':
            self.consume('ID')
            return IdentifierNode(token.value)
        elif token.type == 'LPAREN':
            self.consume('LPAREN')
            node = self.expr() # Start from the lowest precedence (expr)
            self.consume('RPAREN')
            return node
        else:
            raise Exception(f"Syntax Error: Expected literal, ID, or (expression) but got {token.type}")

    def term(self):
        node = self.factor()
        while self.peek().type in ('MUL', 'DIV'):
            op_token = self.peek()
            self.consume(op_token.type)
            node = BinOpNode(left=node, op=op_token.type, right=self.factor())
        return node

    def arith_expr(self):
        node = self.term()
        while self.peek().type in ('PLUS', 'MINUS'):
            op_token = self.peek()
            self.consume(op_token.type)
            node = BinOpNode(left=node, op=op_token.type, right=self.term())
        return node
    
    def comparison(self):
        node = self.arith_expr()
        # NEW: All comparison ops
        comparison_ops = ('EQEQ', 'GT', 'LT', 'GTE', 'LTE', 'NEQ')
        
        if self.peek().type in comparison_ops:
            op_token = self.peek()
            self.consume(op_token.type)
            node = BinOpNode(left=node, op=op_token.type, right=self.arith_expr())
        return node

    def bool_term(self): # NEW: Handles AND
        node = self.comparison()
        while self.peek().type == 'AND':
            op_token = self.peek()
            self.consume('AND')
            node = BinOpNode(left=node, op=op_token.type, right=self.comparison())
        return node

    def expr(self): # NEW: Handles OR (lowest precedence)
        node = self.bool_term()
        while self.peek().type == 'OR':
            op_token = self.peek()
            self.consume('OR')
            node = BinOpNode(left=node, op=op_token.type, right=self.bool_term())
        return node
        
    # --- Statement Rules ---

    def var_declaration(self):
        self.consume('SKIBIDI')
        var_name_token = self.peek()
        self.consume('ID')
        self.consume('ASSIGN')
        value_expr = self.expr() # Use new expr
        self.consume('SEMI')
        return VarDeclNode(var_name=var_name_token.value, value_expr=value_expr)

    def assignment_statement(self):
        var_name_token = self.peek()
        self.consume('ID')
        self.consume('ASSIGN')
        value_expr = self.expr() # Use new expr
        self.consume('SEMI')
        return AssignmentNode(var_name=var_name_token.value, value_expr=value_expr)
    
    def echo_statement(self):
        self.consume('ECHO')
        expr_node = self.expr() # Use new expr
        self.consume('SEMI')
        return ('ECHO', expr_node) 
    
    def if_statement(self):
        self.consume('IF')
        # Parentheses are now recommended for clarity, but expr() will handle it
        condition_expr = self.expr() 
        self.consume('THEN')
        
        then_body = []
        while self.peek().type != 'ENDIF':
            if self.token_index >= len(self.tokens):
                 raise Exception("Syntax Error: Expected ENDIF before EOF")
            
            statement = self.parse_single_statement()
            if statement:
                then_body.append(statement)
        
        self.consume('ENDIF')
        self.consume('SEMI')
        return IfStatementNode(condition=condition_expr, then_body=then_body)
    
    def parse_single_statement(self):
        next_type = self.peek().type
        next_next_type = self.peek(1).type
        
        if next_type == 'SKIBIDI':
            return self.var_declaration()
        elif next_type == 'ECHO':
            return self.echo_statement()
        elif next_type == 'IF':
            return self.if_statement()
        elif next_type == 'ID' and next_next_type == 'ASSIGN':
            return self.assignment_statement()
        else:
             raise Exception(f"Syntax Error: Unexpected statement start with {next_type}")

    def parse_program(self):
        statements = []
        while self.token_index < len(self.tokens):
            statements.append(self.parse_single_statement())
        return statements

# ----------------------------------------------------
# 4. INTERPRETER (EVALUATOR)
# ----------------------------------------------------
class Interpreter:
    def __init__(self, program_ast):
        self.program_ast = program_ast
        self.symbol_table = {}

    def visit(self, node):
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    def visit_NumberNode(self, node):
        return node.value

    def visit_StringNode(self, node):
        return node.value
        
    def visit_IdentifierNode(self, node):
        if node.name not in self.symbol_table:
            raise Exception(f"Runtime Error: Variable '{node.name}' not declared (SKIBIDI)")
        return self.symbol_table[node.name]

    def visit_BinOpNode(self, node):
        # NEW: Short-circuiting for AND/OR
        if node.op == 'AND':
            left_val = self.visit(node.left)
            if not left_val:
                return False
            return bool(self.visit(node.right))
        elif node.op == 'OR':
            left_val = self.visit(node.left)
            if left_val:
                return True
            return bool(self.visit(node.right))

        left_val = self.visit(node.left)
        right_val = self.visit(node.right)

        if node.op == 'PLUS':
            if isinstance(left_val, str) or isinstance(right_val, str):
                return str(left_val) + str(right_val)
            return left_val + right_val
            
        elif node.op == 'MINUS':
            return left_val - right_val
            
        elif node.op == 'MUL':
            return left_val * right_val
            
        elif node.op == 'DIV':
            if right_val == 0:
                raise Exception("Runtime Error: Division by zero")
            if isinstance(left_val, float) or isinstance(right_val, float):
                return left_val / right_val
            return int(left_val / right_val) 
        
        # NEW: All comparison ops
        elif node.op == 'EQEQ':
            return left_val == right_val
        elif node.op == 'NEQ':
            return left_val != right_val
        elif node.op == 'GT':
            return left_val > right_val
        elif node.op == 'GTE':
            return left_val >= right_val
        elif node.op == 'LT':
            return left_val < right_val
        elif node.op == 'LTE':
            return left_val <= right_val
        
        raise Exception(f"Runtime Error: Unknown operator {node.op}")

    def visit_VarDeclNode(self, node):
        if node.var_name in self.symbol_table:
             raise Exception(f"Runtime Error: Variable '{node.var_name}' already declared.")
        value = self.visit(node.value_expr)
        self.symbol_table[node.var_name] = value
        
    def visit_AssignmentNode(self, node):
        if node.var_name not in self.symbol_table:
            raise Exception(f"Runtime Error: Cannot assign to undeclared variable '{node.var_name}'. Use SKIBIDI first.")
        value = self.visit(node.value_expr)
        self.symbol_table[node.var_name] = value

    def visit_IfStatementNode(self, node):
        condition_result = self.visit(node.condition)
        
        # Interpreter evaluation of AND/OR/Comparisons will return True/False
        if condition_result is True:
            for statement in self.then_body:
                self.run_statement(statement)

    def visit_Echo(self, node):
        _, expr_node = node
        result = self.visit(expr_node)
        
        # Output formatting for bools, floats, and ints
        if isinstance(result, bool):
            print(f"Sklang Output: {str(result).upper()}")
        elif isinstance(result, float):
            if result == int(result):
                print(f"Sklang Output: {int(result)}")
            else:
                print(f"Sklang Output: {round(result, 4)}") # More precision
        else:
            print(f"Sklang Output: {result}")

    def run_statement(self, statement):
        if isinstance(statement, tuple) and statement[0] == 'ECHO':
            self.visit_Echo(statement)
        elif isinstance(statement, ASTNode):
            self.visit(statement)

    def run(self):
        for statement in self.program_ast:
            self.run_statement(statement)

# ----------------------------------------------------
# 5. EXECUTION LOGIC (COMMAND-LINE)
# ----------------------------------------------------

def run_sklang_file(filepath):
    """Reads the Sklang code from a file and runs the interpreter pipeline."""
    try:
        # Use explicit encoding for safety
        with open(filepath, 'r', encoding='utf-8') as f:
            sklang_code = f.read()
            
        print(f"--- Running Sklang File: {filepath} ---")

        # 1. Lexing & Parsing
        lexer = Lexer(sklang_code)
        parser = Parser(lexer)
        program_ast = parser.parse_program()
        print("--- AST Successfully Generated ---")

        # 2. Interpretation (Execution)
        interpreter = Interpreter(program_ast)
        print("\n--- Sklang Execution Start ---")
        interpreter.run()
        print("--- Sklang Execution End ---")

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        print(f"\n--- Sklang ERROR --- \n{e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Sklang.py <filepath.skb>") # Assumed filename Sklang.py
        
        print("\n--- Running Self-Test Example (Full Features) ---")
        
        # NEW: Updated self-test to use floats and new syntax
        test_code_lines = [
            "SKIBIDI a = 10.5;",
            "SKIBIDI b = 5 + 3 * 2;",
            "ECHO a + b;",   
            "SKIBIDI greeting = \"Skibidi \";",
            "SKIBIDI subject = \"Language\";",
            "greeting = greeting + subject + 5; // Output: Skibidi Language5", # <-- FIX IS HERE TOO
            "ECHO greeting;",
            "IF (a > b) AND (b == 11) THEN", 
            "    ECHO \"This should not print\";",
            "ENDIF;",
            "IF (a < 100) OR (b != 11) THEN", 
            "    ECHO \"This should print\";",
            "ENDIF;"
        ]
        test_code = "\n".join(test_code_lines)
        
        try:
            lexer = Lexer(test_code)
            parser = Parser(lexer)
            program_ast = parser.parse_program()
            interpreter = Interpreter(program_ast)
            
            print("\n--- Sklang Execution Start ---")
            interpreter.run()
            print("--- Sklang Execution End ---")
        except Exception as e:
            print(f"Self-Test Failed: {e}")
            
    else:
        file_path = sys.argv[1]
        run_sklang_file(file_path)