# ----------------------------------------------------
# 1. CORE COMPONENTS: Token and AST Node Structures
# ----------------------------------------------------
class Token:
    """Represents a lexical unit."""
    def __init__(self, type, value=None):
        self.type = type
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {self.value or ''})"

class ASTNode:
    """Base class for Abstract Syntax Tree nodes."""
    pass

class VarDeclNode(ASTNode):
    """AST node for 'SKIBIDI var = expr;'"""
    def __init__(self, var_name, value_expr):
        self.var_name = var_name
        self.value_expr = value_expr

class BinOpNode(ASTNode):
    """AST node for expressions like 'a + b'"""
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class NumberNode(ASTNode):
    """AST node for a simple number literal."""
    def __init__(self, value):
        self.value = value
        
class IdentifierNode(ASTNode):
    """AST node for a variable name."""
    def __init__(self, name):
        self.name = name

# ----------------------------------------------------
# 2. LEXER (TOKENIZER)
# ----------------------------------------------------
import re

# Define all possible token types and their regular expressions
TOKEN_SPECIFICATIONS = [
    ('NUMBER',          r'\d+'),        # 123
    ('ID',              r'[a-zA-Z_][a-zA-Z0-9_]*'), # Variable names/Keywords
    ('ASSIGN',          r'='),          # =
    ('PLUS',            r'\+'),         # +
    ('MINUS',           r'-'),          # -
    ('SEMI',            r';'),          # ;
    ('LPAREN',          r'\('),         # (
    ('RPAREN',          r'\)'),         # )
    ('WHITESPACE',      r'[ \t\n]+'),   # Ignore spaces, tabs, newlines
    ('MISMATCH',        r'.'),          # Any other character (error)
]

# Map keyword strings to a TOKEN type
KEYWORDS = {
    'SKIBIDI': 'SKIBIDI', # Variable declaration
    'ECHO': 'ECHO',       # Print command (I/O)
}

class Lexer:
    def __init__(self, text):
        self.text = text
        self.position = 0
        self.token_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKEN_SPECIFICATIONS)
        
    def get_next_token(self):
        # Stop if we've consumed all text
        if self.position >= len(self.text):
            return Token('EOF')
        
        # Search for a match from the current position
        match = re.match(self.token_regex, self.text[self.position:])
        
        if match:
            token_type = match.lastgroup
            token_value = match.group(token_type)
            self.position += match.end()

            if token_type == 'WHITESPACE':
                # Skip whitespace and get the next token
                return self.get_next_token()
            
            if token_type == 'ID':
                # Check if the ID is actually a Keyword
                token_type = KEYWORDS.get(token_value, 'ID')
            
            if token_type == 'NUMBER':
                # Convert number strings to integers
                token_value = int(token_value)
                
            if token_type == 'MISMATCH':
                raise Exception(f"Lexical Error: Invalid character at position {self.position - 1}")

            return Token(token_type, token_value)
        
        # Should not be reached if TOKEN_SPECIFICATIONS includes 'MISMATCH'
        raise Exception("Lexical Error: Cannot tokenize remaining input")

# ----------------------------------------------------
# 3. PARSER (SYNTAX CHECKER & AST BUILDER)
# ----------------------------------------------------
class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()
        self.tokens = []
        while self.current_token.type != 'EOF':
             self.tokens.append(self.current_token)
             self.current_token = self.lexer.get_next_token()
        self.token_index = 0

    def consume(self, expected_type):
        """Advances the token stream and verifies the current token."""
        if self.token_index < len(self.tokens) and self.tokens[self.token_index].type == expected_type:
            self.token_index += 1
        else:
            current_type = self.tokens[self.token_index].type if self.token_index < len(self.tokens) else 'EOF'
            raise Exception(f"Syntax Error: Expected {expected_type} but got {current_type}")

    def peek(self, offset=0):
        """Look at the token ahead without consuming it."""
        idx = self.token_index + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return Token('EOF')

    # Basic expression parser for numbers, IDs, and binary operations (+, -)
    def factor(self):
        """Handles single units: numbers or (expressions)"""
        token = self.peek()
        if token.type == 'NUMBER':
            self.consume('NUMBER')
            return NumberNode(token.value)
        elif token.type == 'ID':
            self.consume('ID')
            return IdentifierNode(token.value)
        elif token.type == 'LPAREN':
            self.consume('LPAREN')
            node = self.expr()
            self.consume('RPAREN')
            return node
        else:
            raise Exception(f"Syntax Error: Expected number, ID, or (expression) but got {token.type}")

    def term(self):
        """Handles Multiplication and Division (not implemented in Lexer yet, so identical to expr for now)"""
        return self.expr() 

    def expr(self):
        """Handles Addition and Subtraction: factor (+|-) factor ..."""
        node = self.factor()
        
        while self.peek().type in ('PLUS', 'MINUS'):
            op_token = self.peek()
            self.consume(op_token.type)
            node = BinOpNode(left=node, op=op_token.type, right=self.factor())
            
        return node
        
    def var_declaration(self):
        """Handles 'SKIBIDI var = expr;'"""
        self.consume('SKIBIDI')
        var_name_token = self.peek()
        self.consume('ID')
        self.consume('ASSIGN')
        value_expr = self.expr()
        self.consume('SEMI')
        return VarDeclNode(var_name=var_name_token.value, value_expr=value_expr)
    
    def echo_statement(self):
        """Handles 'ECHO expr;'"""
        self.consume('ECHO')
        expr_node = self.expr()
        self.consume('SEMI')
        return ('ECHO', expr_node) # Simple tuple for the interpreter

    def parse_program(self):
        """The main parsing loop."""
        statements = []
        while self.token_index < len(self.tokens):
            if self.peek().type == 'SKIBIDI':
                statements.append(self.var_declaration())
            elif self.peek().type == 'ECHO':
                 statements.append(self.echo_statement())
            else:
                 # Minimal implementation: just skip unexpected tokens or handle other types here
                 raise Exception(f"Syntax Error: Unexpected statement start with {self.peek().type}")

        return statements

# ----------------------------------------------------
# 4. INTERPRETER (EVALUATOR)
# ----------------------------------------------------
class Interpreter:
    def __init__(self, program_ast):
        self.program_ast = program_ast
        # Symbol Table: Stores variable names and their values
        self.symbol_table = {}

    def visit(self, node):
        """Generic method to dispatch evaluation based on node type."""
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    def visit_NumberNode(self, node):
        return node.value

    def visit_IdentifierNode(self, node):
        if node.name not in self.symbol_table:
            raise Exception(f"Runtime Error: Variable '{node.name}' not declared (SKIBIDI)")
        return self.symbol_table[node.name]

    def visit_BinOpNode(self, node):
        left_val = self.visit(node.left)
        right_val = self.visit(node.right)

        if node.op == 'PLUS':
            return left_val + right_val
        elif node.op == 'MINUS':
            return left_val - right_val
        # Add logic for *, /, etc. here
        
        raise Exception(f"Runtime Error: Unknown operator {node.op}")

    def visit_VarDeclNode(self, node):
        # Evaluate the expression on the right side of the assignment
        value = self.visit(node.value_expr)
        # Store the variable in the symbol table
        self.symbol_table[node.var_name] = value

    def visit_Echo(self, node):
        """Handles the ECHO statement (Output)"""
        # Node is a tuple ('ECHO', expr_node)
        _, expr_node = node
        result = self.visit(expr_node)
        print(f"Sklang Output: {result}")

    def run(self):
        """Execute the entire program AST."""
        for statement in self.program_ast:
            # The structure for simple statements is either an ASTNode or a special tuple
            if isinstance(statement, tuple) and statement[0] == 'ECHO':
                self.visit_Echo(statement)
            elif isinstance(statement, ASTNode):
                self.visit(statement)
            else:
                raise Exception(f"Unknown statement type: {statement}")

# ----------------------------------------------------
# 5. EXECUTION EXAMPLE
# ----------------------------------------------------

# Sklang Code to interpret (Main Feature: Variables and Expressions)
sklang_code = """
SKIBIDI a = 10;
SKIBIDI b = 5 + 3;
SKIBIDI c = a - 2;
ECHO b + c;
"""

try:
    # 1. Lexing
    lexer = Lexer(sklang_code)
    # The Lexer runs during Parser initialization

    # 2. Parsing
    parser = Parser(lexer)
    program_ast = parser.parse_program()
    print("--- AST Successfully Generated ---")
    # print(program_ast) # Uncomment to see the raw AST nodes

    # 3. Interpretation (Execution)
    interpreter = Interpreter(program_ast)
    print("\n--- Sklang Execution Start ---")
    interpreter.run()
    print("--- Sklang Execution End ---")

except Exception as e:
    print(f"\n--- Sklang ERROR --- \n{e}")