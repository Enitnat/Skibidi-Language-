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

# --- Statement Nodes ---
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

class WhileLoopNode(ASTNode): # NEW: For Looping
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class FuncDeclNode(ASTNode): # NEW: For Functions
    def __init__(self, name, arg_names, body):
        self.name = name
        self.arg_names = arg_names
        self.body = body

class ReturnNode(ASTNode): # NEW: For Functions
    def __init__(self, value_expr):
        self.value_expr = value_expr

# --- Expression Nodes ---
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

class InputNode(ASTNode): # NEW: For Input
    pass

class CallNode(ASTNode): # NEW: For Functions
    def __init__(self, callee_name, arg_exprs):
        self.callee_name = callee_name
        self.arg_exprs = arg_exprs

# ----------------------------------------------------
# 2. LEXER (TOKENIZER)
# ----------------------------------------------------

TOKEN_SPECIFICATIONS = [
    ('NUMBER',    r'\d+\.\d+|\d+'),
    ('ID',        r'[a-zA-Z_][a-zA-Z0-9_]*'), 
    ('STRING',    r'".*?"'), 
    ('EQEQ',      r'=='),
    ('NEQ',       r'!='), 
    ('GTE',       r'>='), 
    ('LTE',       r'<='), 
    ('GT',        r'>'),
    ('LT',        r'<'), 
    ('ASSIGN',    r'='),       
    ('PLUS',      r'\+'),      
    ('MINUS',     r'-'),       
    ('MUL',       r'\*'),      
    ('COMMENT',   r'//[^\n]*'), # Must be before DIV
    ('DIV',       r'/'),       
    ('SEMI',      r';'),       
    ('LPAREN',    r'\('),      
    ('RPAREN',    r'\)'),      
    ('COMMA',     r','),       # NEW: For function args
    ('WHITESPACE',r'[ \t\n\r]+'), 
    ('MISMATCH',  r'.'), 
]

# --- 🧠 Brainrot Edition Keywords ---

KEYWORDS = {
    'SKIBIDI': 'SKIBIDI',  # Variable Declaration
    'YAP':     'ECHO',     # Output (as requested)
    'SUS':     'IF',       # Conditional
    'FR':      'THEN',     # "For Real"
    'NAH':     'ENDIF',    # End Conditional
    'AND':     'AND',      # Kept for logic
    'OR':      'OR',       # Kept for logic
    'COOK':   'WHILE',    # Loop
    'GYATT':    'DO',       # Do loop body
    'STOPCOOKIN': 'ENDWHILE',# End Loop
    'GIMME':   'INPUT',    # Get user input
    'SPAWN':   'DEF',      # Define Function
    'DESPAWN': 'ENDDEF',   # End Function
    'YEET':    'RETURN',   # Return value
}

class Lexer:
    def __init__(self, text):
        self.text = text
        self.position = 0
        self.token_regex = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKEN_SPECIFICATIONS), re.DOTALL)
        
    def get_next_token(self):
        if self.position >= len(self.text):
            return Token('EOF', None)
        
        match = self.token_regex.match(self.text, self.position)
        
        if not match:
            raise Exception(f"Lexical Error: Cannot tokenize at position {self.position}")

        token_type = match.lastgroup
        token_value = match.group(token_type)
        self.position = match.end()

        if token_type in ('WHITESPACE', 'COMMENT'):
            return self.get_next_token()
        
        if token_type == 'ID':
            token_type = KEYWORDS.get(token_value, 'ID')
        
        if token_type == 'NUMBER':
            token_value = float(token_value) if '.' in token_value else int(token_value)
        
        if token_type == 'STRING': 
            token_value = token_value[1:-1]
        
        if token_type == 'MISMATCH':
            raise Exception(f"Lexical Error: Invalid character '{token_value}' at position {self.position - 1}")

        return Token(token_type, token_value)

# ----------------------------------------------------
# 3. PARSER (SYNTAX CHECKER & AST BUILDER)
# ----------------------------------------------------
class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.tokens = []
        self.token_index = 0
        
        t = self.lexer.get_next_token()
        while t.type != 'EOF':
            self.tokens.append(t)
            t = self.lexer.get_next_token()

    def consume(self, expected_type):
        if self.token_index < len(self.tokens) and self.tokens[self.token_index].type == expected_type:
            tok = self.tokens[self.token_index]
            self.token_index += 1
            return tok
        current_type = self.peek().type
        raise Exception(f"Syntax Error: Expected {expected_type} but got {current_type}")

    def peek(self, offset=0):
        idx = self.token_index + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return Token('EOF', None)

    # --- Expression Grammar ---

    def factor(self):
        token = self.peek()
        if token.type == 'NUMBER':
            self.consume('NUMBER')
            return NumberNode(token.value)
        if token.type == 'STRING':
            self.consume('STRING')
            return StringNode(token.value)
        
        if token.type == 'INPUT': # NEW: Handle INPUT()
            self.consume('INPUT')
            self.consume('LPAREN')
            self.consume('RPAREN')
            return InputNode()

        if token.type == 'ID':
            if self.peek(1).type == 'LPAREN': # NEW: Handle Function Call
                return self.call_expression()
            else:
                self.consume('ID')
                return IdentifierNode(token.value)
                
        if token.type == 'LPAREN':
            self.consume('LPAREN')
            node = self.expr() 
            self.consume('RPAREN')
            return node
            
        raise Exception(f"Syntax Error: Expected literal, ID, or (expression) but got {token.type}")

    def call_expression(self): # NEW: For Function Calls
        name_token = self.consume('ID')
        self.consume('LPAREN')
        arg_exprs = []
        if self.peek().type != 'RPAREN':
            arg_exprs.append(self.expr())
            while self.peek().type == 'COMMA':
                self.consume('COMMA')
                arg_exprs.append(self.expr())
        self.consume('RPAREN')
        return CallNode(name_token.value, arg_exprs)

    def term(self):
        node = self.factor()
        while self.peek().type in ('MUL', 'DIV'):
            op = self.consume(self.peek().type)
            node = BinOpNode(node, op.type, self.factor())
        return node

    def arith_expr(self):
        node = self.term()
        while self.peek().type in ('PLUS', 'MINUS'):
            op = self.consume(self.peek().type)
            node = BinOpNode(node, op.type, self.term())
        return node
    
    def comparison(self):
        node = self.arith_expr()
        comparison_ops = ('EQEQ', 'NEQ', 'GT', 'GTE', 'LT', 'LTE')
        if self.peek().type in comparison_ops:
            op = self.consume(self.peek().type)
            node = BinOpNode(node, op.type, self.arith_expr())
        return node

    def bool_term(self):
        node = self.comparison()
        while self.peek().type == 'AND':
            op = self.consume('AND')
            node = BinOpNode(node, op.type, self.comparison())
        return node

    def expr(self):
        node = self.bool_term()
        while self.peek().type == 'OR':
            op = self.consume('OR')
            node = BinOpNode(node, op.type, self.bool_term())
        return node
        
    # --- Statement Rules ---

    def var_declaration(self):
        self.consume('SKIBIDI')
        name = self.consume('ID').value
        self.consume('ASSIGN')
        expr = self.expr()
        self.consume('SEMI')
        return VarDeclNode(name, expr)

    def assignment_statement(self):
        name = self.consume('ID').value
        self.consume('ASSIGN')
        expr = self.expr()
        self.consume('SEMI')
        return AssignmentNode(name, expr)
    
    def echo_statement(self):
        self.consume('ECHO')
        expr = self.expr()
        self.consume('SEMI')
        return ('ECHO', expr) 
    
    def if_statement(self):
        self.consume('IF')
        condition = self.expr() 
        self.consume('THEN')
        body = []
        while self.peek().type != 'ENDIF':
            if self.peek().type == 'EOF':
                raise Exception("Syntax Error: Expected ENDIF but reached EOF")
            body.append(self.parse_single_statement())
        self.consume('ENDIF')
        self.consume('SEMI')
        return IfStatementNode(condition, body)

    def while_statement(self): # NEW: For Loops
        self.consume('WHILE')
        condition = self.expr()
        self.consume('DO')
        body = []
        while self.peek().type != 'ENDWHILE':
            if self.peek().type == 'EOF':
                raise Exception("Syntax Error: Expected ENDWHILE but reached EOF")
            body.append(self.parse_single_statement())
        self.consume('ENDWHILE')
        self.consume('SEMI')
        return WhileLoopNode(condition, body)

    def func_declaration(self): # NEW: For Functions
        self.consume('DEF')
        name = self.consume('ID').value
        self.consume('LPAREN')
        arg_names = []
        if self.peek().type != 'RPAREN':
            arg_names.append(self.consume('ID').value)
            while self.peek().type == 'COMMA':
                self.consume('COMMA')
                arg_names.append(self.consume('ID').value)
        self.consume('RPAREN')
        self.consume('THEN')
        body = []
        while self.peek().type != 'ENDDEF' and self.peek().type != 'RETURN':
            if self.peek().type == 'EOF':
                raise Exception("Syntax Error: Expected ENDDEF or RETURN but reached EOF")
            body.append(self.parse_single_statement())
        
        # Handle optional return
        if self.peek().type == 'RETURN':
            body.append(self.return_statement())
            
        self.consume('ENDDEF')
        self.consume('SEMI')
        return FuncDeclNode(name, arg_names, body)

    def return_statement(self): # NEW: For Functions
        self.consume('RETURN')
        value_expr = self.expr()
        self.consume('SEMI')
        return ReturnNode(value_expr)
    
    def parse_single_statement(self):
        t = self.peek().type
        t2 = self.peek(1).type

        if t == 'SKIBIDI':
            return self.var_declaration()
        if t == 'ECHO':
            return self.echo_statement()
        if t == 'IF':
            return self.if_statement()
        if t == 'WHILE': # NEW
            return self.while_statement()
        if t == 'DEF': # NEW
            return self.func_declaration()
        if t == 'RETURN': # NEW
            return self.return_statement()
        if t == 'ID' and t2 == 'ASSIGN':
            return self.assignment_statement()
        # Handle function calls as statements
        if t == 'ID' and t2 == 'LPAREN':
            stmt = self.call_expression()
            self.consume('SEMI')
            return stmt

        raise Exception(f"Syntax Error: Unexpected statement start with {t}")

    def parse_program(self):
        program = []
        while self.token_index < len(self.tokens):
            program.append(self.parse_single_statement())
        return program

# ----------------------------------------------------
# 4. INTERPRETER (EVALUATOR)
# ----------------------------------------------------

class ReturnValue(Exception): # NEW: Custom exception for RETURN logic
    """Exception to handle non-local jumps for RETURN statements."""
    def __init__(self, value):
        self.value = value

class Interpreter:
    def __init__(self, program_ast):
        self.program_ast = program_ast
        # NEW: Scope stack for global/local scopes. Start with one global scope.
        self.scope_stack = [{}]
        # NEW: Separate table to store function definitions (FuncDeclNodes)
        self.function_table = {}

    # --- Scope Management Helpers ---
    def get_var(self, name):
        """Search for a variable from the top scope down."""
        for scope in reversed(self.scope_stack):
            if name in scope:
                return scope[name]
        raise Exception(f"Runtime Error: Variable '{name}' not declared")

    def set_var(self, name, value):
        """Set a variable's value in the scope it was declared in."""
        for scope in reversed(self.scope_stack):
            if name in scope:
                scope[name] = value
                return
        raise Exception(f"Runtime Error: Variable '{name}' not declared")

    def declare_var(self, name, value):
        """Declare a variable in the current (top-most) scope."""
        current_scope = self.scope_stack[-1]
        if name in current_scope:
            raise Exception(f"Runtime Error: Variable '{name}' already declared in this scope.")
        current_scope[name] = value

    # --- Visitor Methods ---
    
    def visit(self, node):
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f'Runtime Error: No visit_{type(node).__name__} method defined')

    def visit_NumberNode(self, node):
        return node.value

    def visit_StringNode(self, node):
        return node.value
        
    def visit_IdentifierNode(self, node):
        return self.get_var(node.name) # Use scope helper

    def visit_InputNode(self, node): # NEW: Handle Input
        val = input("Sklang Input: ")
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val # Default to string

    def visit_BinOpNode(self, node):
        # Short-circuiting for AND/OR
        if node.op == 'AND':
            return self.visit(node.left) and self.visit(node.right)
        elif node.op == 'OR':
            return self.visit(node.left) or self.visit(node.right)

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
        
        elif node.op == 'EQEQ': return left_val == right_val
        elif node.op == 'NEQ':  return left_val != right_val
        elif node.op == 'GT':   return left_val > right_val
        elif node.op == 'GTE':  return left_val >= right_val
        elif node.op == 'LT':   return left_val < right_val
        elif node.op == 'LTE':  return left_val <= right_val
        
        raise Exception(f"Runtime Error: Unknown operator {node.op}")

    def visit_VarDeclNode(self, node):
        value = self.visit(node.value_expr)
        self.declare_var(node.var_name, value) # Use scope helper
        
    def visit_AssignmentNode(self, node):
        value = self.visit(node.value_expr)
        self.set_var(node.var_name, value) # Use scope helper

    def visit_IfStatementNode(self, node):
        condition_result = self.visit(node.condition)
        if condition_result: # Use truthiness
            for statement in node.then_body: # Use node.then_body
                self.run_statement(statement)

    def visit_WhileLoopNode(self, node): # NEW: Handle Loops
        while self.visit(node.condition):
            for statement in node.body:
                self.run_statement(statement)

    def visit_FuncDeclNode(self, node): # NEW: Handle Func Def
        self.function_table[node.name] = node

    def visit_ReturnNode(self, node): # NEW: Handle Return
        value = self.visit(node.value_expr)
        raise ReturnValue(value)

    def visit_CallNode(self, node): # NEW: Handle Func Call
        if node.callee_name not in self.function_table:
            raise Exception(f"Runtime Error: Function '{node.callee_name}' not defined.")
            
        func = self.function_table[node.callee_name]
        
        if len(node.arg_exprs) != len(func.arg_names):
            raise Exception(f"Runtime Error: Function '{func.name}' expected {len(func.arg_names)} args, but got {len(node.arg_exprs)}")
        
        # 1. Create new scope
        new_scope = {}
        # 2. Evaluate args in *current* scope and place in *new* scope
        for name, expr in zip(func.arg_names, node.arg_exprs):
            new_scope[name] = self.visit(expr)
            
        # 3. Push new scope
        self.scope_stack.append(new_scope)
        
        return_value = None
        try:
            # 4. Execute function body
            for statement in func.body:
                self.run_statement(statement)
        except ReturnValue as e:
            # 5. Catch return value
            return_value = e.value
            
        # 6. Pop scope
        self.scope_stack.pop()
        
        return return_value # Will be None if no RETURN statement was hit

    def visit_Echo(self, node):
        _, expr_node = node
        result = self.visit(expr_node)
        
        if isinstance(result, bool):
            print(f"Sklang Output: {str(result).upper()}")
        elif isinstance(result, float):
            if result == int(result):
                print(f"Sklang Output: {int(result)}")
            else:
                print(f"Sklang Output: {round(result, 4)}")
        else:
            print(f"Sklang Output: {result}")

    def run_statement(self, statement):
        """Helper to run a single statement."""
        if isinstance(statement, tuple) and statement[0] == 'ECHO':
            self.visit_Echo(statement)
        elif isinstance(statement, ASTNode):
            self.visit(statement)

    def execute(self):
        """Execute the entire program AST."""
        for statement in self.program_ast:
            self.run_statement(statement)

# ----------------------------------------------------
# 5. EXECUTION LOGIC (COMMAND-LINE)
# ----------------------------------------------------

def run_sklang_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            sklang_code = f.read()
            
        print(f"--- Running Sklang File: {filepath} ---")
        
        lexer = Lexer(sklang_code)
        parser = Parser(lexer)
        program_ast = parser.parse_program()
        print("--- AST Successfully Generated ---")

        interpreter = Interpreter(program_ast)
        print("\n--- Sklang Execution Start ---")
        interpreter.execute()
        print("--- Sklang Execution End ---")

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        print(f"\n--- Sklang ERROR --- \n{e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Sklang.py <filepath.skb>") 
        
        print("\n--- Running Self-Test Example ---")
        
        test_code_lines = [
            "SKIBIDI a = 1;",
            "COOK (a < 3) GYATT",
            "    YAP a;",
            "    a = a + 1;",
            "STOPCOOKIN;",
            "YAP \"Done with loop\";"
]
        test_code = "\n".join(test_code_lines)
        
        try:
            lexer = Lexer(test_code)
            parser = Parser(lexer)
            program_ast = parser.parse_program()
            interpreter = Interpreter(program_ast)
            
            print("\n--- Sklang Execution Start ---")
            interpreter.execute()
            print("--- Sklang Execution End ---")
        except Exception as e:
            print(f"Self-Test Failed: {e}")
            
    else:
        file_path = sys.argv[1]
        run_sklang_file(file_path)