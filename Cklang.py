import re, sys

class Token:
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

class StringNode(ASTNode):
    def __init__(self, value):
        self.value = value

class IdentifierNode(ASTNode):
    def __init__(self, name):
        self.name = name


TOKEN_SPECIFICATIONS = [
    ('NUMBER', r'\d+\.\d+|\d+'),
    ('ID', r'[a-zA-Z_][a-zA-Z0-9_]*'),
    ('STRING', r'\".*?\"'),
    ('EQEQ', r'=='),
    ('NEQ', r'!='),
    ('GTE', r'>='),
    ('LTE', r'<='),
    ('GT', r'>'),
    ('LT', r'<'),
    ('ASSIGN', r'='),
    ('PLUS', r'\+'),
    ('MINUS', r'-'),
    ('MUL', r'\*'),

    # IMPORTANT: COMMENT MUST COME BEFORE DIV
    ('COMMENT', r'//[^\n]*'),

    ('DIV', r'/'),
    ('SEMI', r';'),
    ('LPAREN', r'\('),
    ('RPAREN', r'\)'),
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
            raise Exception(f"Lexical Error at position {self.position}")

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
            raise Exception(f"Invalid character '{token_value}' at position {self.position - 1}")

        return Token(token_type, token_value)


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
        raise Exception(f"Syntax Error: expected {expected_type}")

    def peek(self, offset=0):
        idx = self.token_index + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return Token('EOF', None)

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
        raise Exception("Expected number, string, or identifier")

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
        if self.peek().type in ('EQEQ', 'NEQ', 'GT', 'LT', 'GTE', 'LTE'):
            op = self.consume(self.peek().type)
            node = BinOpNode(node, op.type, self.arith_expr())
        return node

    def bool_term(self):
        node = self.comparison()
        while self.peek().type == 'AND':
            self.consume('AND')
            node = BinOpNode(node, 'AND', self.comparison())
        return node

    def expr(self):
        node = self.bool_term()
        while self.peek().type == 'OR':
            self.consume('OR')
            node = BinOpNode(node, 'OR', self.bool_term())
        return node

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
            body.append(self.parse_single_statement())

        self.consume('ENDIF')
        self.consume('SEMI')
        return IfStatementNode(condition, body)

    def parse_single_statement(self):
        t = self.peek().type
        t2 = self.peek(1).type

        if t == 'SKIBIDI':
            return self.var_declaration()
        if t == 'ECHO':
            return self.echo_statement()
        if t == 'IF':
            return self.if_statement()
        if t == 'ID' and t2 == 'ASSIGN':
            return self.assignment_statement()

        raise Exception(f"Unexpected token {t}")

    def parse_program(self):
        program = []
        while self.token_index < len(self.tokens):
            program.append(self.parse_single_statement())
        return program


class Interpreter:
    def __init__(self, program_ast):
        self.program_ast = program_ast
        self.symbol_table = {}

    def visit(self, node):
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method:
            return method(node)
        raise Exception(f"No visit method for {type(node).__name__}")

    def visit_NumberNode(self, node):
        return node.value

    def visit_StringNode(self, node):
        return node.value

    def visit_IdentifierNode(self, node):
        if node.name not in self.symbol_table:
            raise Exception(f"Variable '{node.name}' not declared")
        return self.symbol_table[node.name]

    def visit_BinOpNode(self, node):
        # Short-circuit logic
        if node.op == 'AND':
            left = self.visit(node.left)
            return left and self.visit(node.right)
        if node.op == 'OR':
            left = self.visit(node.left)
            return left or self.visit(node.right)

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
                raise Exception("Division by zero")
            return left / right

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

        raise Exception(f"Unknown operator {node.op}")

    def visit_VarDeclNode(self, node):
        self.symbol_table[node.var_name] = self.visit(node.value_expr)

    def visit_AssignmentNode(self, node):
        if node.var_name not in self.symbol_table:
            raise Exception(f"Variable '{node.var_name}' not declared")
        self.symbol_table[node.var_name] = self.visit(node.value_expr)

    def visit_IfStatementNode(self, node):
        condition = self.visit(node.condition)
        if condition:
            for stmt in node.then_body:
                self.run(stmt)

    def visit_Echo(self, node):
        _, expr = node
        value = self.visit(expr)
        print("Sklang Output:", value)

    def run(self, stmt):
        if isinstance(stmt, tuple) and stmt[0] == 'ECHO':
            self.visit_Echo(stmt)
        else:
            self.visit(stmt)

    def execute(self):
        for stmt in self.program_ast:
            self.run(stmt)
