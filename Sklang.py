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
        return f"Token({self.type}, {repr(self.value)})"

class ASTNode:
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
    ('NUMBER',   r'\d+\.\d+|\d+'),
    ('ID',       r'[a-zA-Z_][a-zA-Z0-9_]*'),
    ('STRING',   r'".*?"'),
    ('EQEQ',     r'=='),
    ('NEQ',      r'!='), 
    ('GTE',      r'>='), 
    ('LTE',      r'<='), 
    ('GT',       r'>'),
    ('LT',       r'<'),
    ('ASSIGN',   r'='),       
    ('PLUS',     r'\+'),      
    ('MINUS',    r'-'),       
    ('MUL',      r'\*'),      
    ('COMMENT',  r'//[^\n]*'),   # COMMENT before DIV so // is recognized
    ('DIV',      r'/'),       
    ('SEMI',     r';'),       
    ('LPAREN',   r'\('),      
    ('RPAREN',   r'\)'),      
    ('WHITESPACE', r'[ \t\n\r]+'),
    ('MISMATCH', r'.'),
]

KEYWORDS = {
    'SKIBIDI': 'SKIBIDI',
    'ECHO': 'ECHO',
    'IF': 'IF',
    'THEN': 'THEN',
    'ENDIF': 'ENDIF',
    'AND': 'AND',
    'OR': 'OR',
}

class Lexer:
    def __init__(self, text):
        self.text = text
        self.position = 0
        self.token_regex = re.compile(
            '|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKEN_SPECIFICATIONS),
            re.DOTALL
        )

    def get_next_token(self):
        if self.position >= len(self.text):
            return Token('EOF', None)
        match = self.token_regex.match(self.text, self.position)
        if not match:
            raise Exception(f"Lexical Error: Cannot tokenize remaining input at position {self.position}")
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

        tok = self.lexer.get_next_token()
        while tok.type != 'EOF':
            self.tokens.append(tok)
            tok = self.lexer.get_next_token()

    def consume(self, expected_type):
        if self.token_index < len(self.tokens) and self.tokens[self.token_index].type == expected_type:
            tok = self.tokens[self.token_index]
            self.token_index += 1
            return tok
        else:
            current_type = self.tokens[self.token_index].type if self.token_index < len(self.tokens) else 'EOF'
            raise Exception(f"Syntax Error: Expected {expected_type} but got {current_type}")

    def peek(self, offset=0):
        idx = self.token_index + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return Token('EOF', None)   # <-- fixed: explicit None value

    # Expression grammar
    def factor(self):
        token = self.peek()
        if token.type == 'NUMBER':
            self.consume('NUMBER')
            return NumberNode(token.value)
        if token.type == 'STRING':
            self.consume('STRING')
            return StringNode(token.value)
        if token.type == 'ID':
            self.consume('ID')
            return IdentifierNode(token.value)
        if token.type == 'LPAREN':
            self.consume('LPAREN')
            node = self.expr()
            self.consume('RPAREN')
            return node
        raise Exception(f"Syntax Error: Expected literal, ID, or (expression) but got {token.type}")

    def term(self):
        node = self.factor()
        while self.peek().type in ('MUL', 'DIV'):
            op = self.peek()
            self.consume(op.type)
            node = BinOpNode(left=node, op=op.type, right=self.factor())
        return node

    def arith_expr(self):
        node = self.term()
        while self.peek().type in ('PLUS', 'MINUS'):
            op = self.peek()
            self.consume(op.type)
            node = BinOpNode(left=node, op=op.type, right=self.term())
        return node

    def comparison(self):
        node = self.arith_expr()
        if self.peek().type in ('EQEQ', 'NEQ', 'GT', 'GTE', 'LT', 'LTE'):
            op = self.peek()
            self.consume(op.type)
            node = BinOpNode(left=node, op=op.type, right=self.arith_expr())
        return node

    def bool_term(self):
        node = self.comparison()
        while self.peek().type == 'AND':
            op = self.peek()
            self.consume('AND')
            node = BinOpNode(left=node, op=op.type, right=self.comparison())
        return node

    def expr(self):
        node = self.bool_term()
        while self.peek().type == 'OR':
            op = self.peek()
            self.consume('OR')
            node = BinOpNode(left=node, op=op.type, right=self.bool_term())
        return node

    # Statements
    def var_declaration(self):
        self.consume('SKIBIDI')
        var_name_token = self.peek()
        self.consume('ID')
        self.consume('ASSIGN')
        value_expr = self.expr()
        self.consume('SEMI')
        return VarDeclNode(var_name=var_name_token.value, value_expr=value_expr)

    def assignment_statement(self):
        var_name_token = self.peek()
        self.consume('ID')
        self.consume('ASSIGN')
        value_expr = self.expr()
        self.consume('SEMI')
        return AssignmentNode(var_name=var_name_token.value, value_expr=value_expr)

    def echo_statement(self):
        self.consume('ECHO')
        expr_node = self.expr()
        self.consume('SEMI')
        return ('ECHO', expr_node)

    def if_statement(self):
        self.consume('IF')
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
        nxt = self.peek().type
        nxt2 = self.peek(1).type
        if nxt == 'SKIBIDI':
            return self.var_declaration()
        if nxt == 'ECHO':
            return self.echo_statement()
        if nxt == 'IF':
            return self.if_statement()
        if nxt == 'ID' and nxt2 == 'ASSIGN':
            return self.assignment_statement()
        raise Exception(f"Syntax Error: Unexpected statement start with {nxt}")

    def parse_program(self):
        stmts = []
        while self.token_index < len(self.tokens):
            stmts.append(self.parse_single_statement())
        return stmts

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
        # Short-circuit AND/OR
        if node.op == 'AND':
            left = self.visit(node.left)
            if not left:
                return False
            return bool(self.visit(node.right))
        if node.op == 'OR':
            left = self.visit(node.left)
            if left:
                return True
            return bool(self.visit(node.right))

        left = self.visit(node.left)
        right = self.visit(node.right)

        if node.op == 'PLUS':
            if isinstance(left, str) or isinstance(right, str):
                return str(left) + str(right)
            return left + right
        if node.op == 'MINUS':
            return left - right
        if node.op == 'MUL':
            return left * right
        if node.op == 'DIV':
            if right == 0:
                raise Exception("Runtime Error: Division by zero")
            if isinstance(left, float) or isinstance(right, float):
                return left / right
            return int(left / right)
        if node.op == 'EQEQ':
            return left == right
        if node.op == 'NEQ':
            return left != right
        if node.op == 'GT':
            return left > right
        if node.op == 'GTE':
            return left >= right
        if node.op == 'LT':
            return left < right
        if node.op == 'LTE':
            return left <= right
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
        # <-- FIX: use truthiness instead of `is True`
        if condition_result:
            for statement in node.then_body:
                self.run_statement(statement)

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
        if isinstance(statement, tuple) and statement[0] == 'ECHO':
            self.visit_Echo(statement)
        elif isinstance(statement, ASTNode):
            self.visit(statement)
        else:
            raise Exception(f"Runtime Error: Unknown statement type {type(statement)}")

    def run(self):
        for statement in self.program_ast:
            self.run_statement(statement)

# ----------------------------------------------------
# 5. EXECUTION LOGIC (COMMAND-LINE)
# ----------------------------------------------------
def run_sklang_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            sklang_code = f.read()
        lexer = Lexer(sklang_code)
        parser = Parser(lexer)
        program_ast = parser.parse_program()
        interpreter = Interpreter(program_ast)
        interpreter.run()
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        print(f"\n--- Sklang ERROR --- \n{e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        # Self-test
        test_code_lines = [
            "SKIBIDI a = 10.5;",
            "SKIBIDI b = 5 + 3 * 2; // b = 11",
            "ECHO a + b;",
            "SKIBIDI greeting = \"Skibidi \";",
            "SKIBIDI subject = \"Language\";",
            "greeting = greeting + subject + 5;",
            "ECHO greeting;",
            "IF (a > b) AND (b == 11) THEN",
            "    ECHO \"This should not print\";",
            "ENDIF;",
            "IF (a < 100) OR (b != 11) THEN",
            "    ECHO \"This should print\";",
            "ENDIF;"
        ]
        test_code = "\\n".join(test_code_lines)
        lexer = Lexer(test_code)
        parser = Parser(lexer)
        program_ast = parser.parse_program()
        interpreter = Interpreter(program_ast)
        print("--- Sklang Execution Start (Self-Test) ---")
        interpreter.run()
        print("--- Sklang Execution End ---")
    else:
        run_sklang_file(sys.argv[1])
