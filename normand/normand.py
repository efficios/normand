# The MIT License (MIT)
#
# Copyright (c) 2023 Philippe Proulx <eeppeliteloop@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# This module is the portable Normand processor. It offers both the
# parse() function and the command-line tool (run the module itself)
# without external dependencies except a `typing` module for Python 3.4.
#
# Feel free to copy this module file to your own project to use Normand.
#
# Upstream repository: <https://github.com/efficios/normand>.

__author__ = "Philippe Proulx"
__version__ = "0.5.0"
__all__ = [
    "ByteOrder",
    "parse",
    "ParseError",
    "ParseResult",
    "TextLoc",
    "SymbolsT",
    "__author__",
    "__version__",
]

import re
import abc
import ast
import sys
import enum
import math
import struct
from typing import (
    Any,
    Set,
    Dict,
    List,
    Tuple,
    Union,
    Pattern,
    Callable,
    NoReturn,
    Optional,
)


# Text location (line and column numbers).
class TextLoc:
    @classmethod
    def _create(cls, line_no: int, col_no: int):
        self = cls.__new__(cls)
        self._init(line_no, col_no)
        return self

    def __init__(*args, **kwargs):  # type: ignore
        raise NotImplementedError

    def _init(self, line_no: int, col_no: int):
        self._line_no = line_no
        self._col_no = col_no

    # Line number.
    @property
    def line_no(self):
        return self._line_no

    # Column number.
    @property
    def col_no(self):
        return self._col_no

    def __repr__(self):
        return "TextLoc({}, {})".format(self._line_no, self._col_no)


# Any item.
class _Item:
    def __init__(self, text_loc: TextLoc):
        self._text_loc = text_loc

    # Source text location.
    @property
    def text_loc(self):
        return self._text_loc


# Scalar item.
class _ScalarItem(_Item):
    # Returns the size, in bytes, of this item.
    @property
    @abc.abstractmethod
    def size(self) -> int:
        ...


# A repeatable item.
class _RepableItem:
    pass


# Single byte.
class _Byte(_ScalarItem, _RepableItem):
    def __init__(self, val: int, text_loc: TextLoc):
        super().__init__(text_loc)
        self._val = val

    # Byte value.
    @property
    def val(self):
        return self._val

    @property
    def size(self):
        return 1

    def __repr__(self):
        return "_Byte({}, {})".format(hex(self._val), self._text_loc)


# String.
class _Str(_ScalarItem, _RepableItem):
    def __init__(self, data: bytes, text_loc: TextLoc):
        super().__init__(text_loc)
        self._data = data

    # Encoded bytes.
    @property
    def data(self):
        return self._data

    @property
    def size(self):
        return len(self._data)

    def __repr__(self):
        return "_Str({}, {})".format(repr(self._data), self._text_loc)


# Byte order.
@enum.unique
class ByteOrder(enum.Enum):
    # Big endian.
    BE = "be"

    # Little endian.
    LE = "le"


# Byte order setting.
class _SetBo(_Item):
    def __init__(self, bo: ByteOrder, text_loc: TextLoc):
        super().__init__(text_loc)
        self._bo = bo

    @property
    def bo(self):
        return self._bo

    def __repr__(self):
        return "_SetBo({}, {})".format(repr(self._bo), self._text_loc)


# Label.
class _Label(_Item):
    def __init__(self, name: str, text_loc: TextLoc):
        super().__init__(text_loc)
        self._name = name

    # Label name.
    @property
    def name(self):
        return self._name

    def __repr__(self):
        return "_Label({}, {})".format(repr(self._name), self._text_loc)


# Offset setting.
class _SetOffset(_Item):
    def __init__(self, val: int, text_loc: TextLoc):
        super().__init__(text_loc)
        self._val = val

    # Offset value.
    @property
    def val(self):
        return self._val

    def __repr__(self):
        return "_SetOffset({}, {})".format(repr(self._val), self._text_loc)


# Mixin of containing an AST expression and its string.
class _ExprMixin:
    def __init__(self, expr_str: str, expr: ast.Expression):
        self._expr_str = expr_str
        self._expr = expr

    # Expression string.
    @property
    def expr_str(self):
        return self._expr_str

    # Expression node to evaluate.
    @property
    def expr(self):
        return self._expr


# Variable assignment.
class _VarAssign(_Item, _ExprMixin):
    def __init__(
        self, name: str, expr_str: str, expr: ast.Expression, text_loc: TextLoc
    ):
        super().__init__(text_loc)
        _ExprMixin.__init__(self, expr_str, expr)
        self._name = name

    # Name.
    @property
    def name(self):
        return self._name

    def __repr__(self):
        return "_VarAssign({}, {}, {}, {})".format(
            repr(self._name), repr(self._expr_str), repr(self._expr), self._text_loc
        )


# Fixed-length integer, possibly needing more than one byte.
class _FlInt(_ScalarItem, _RepableItem, _ExprMixin):
    def __init__(
        self, expr_str: str, expr: ast.Expression, len: int, text_loc: TextLoc
    ):
        super().__init__(text_loc)
        _ExprMixin.__init__(self, expr_str, expr)
        self._len = len

    # Length (bits).
    @property
    def len(self):
        return self._len

    @property
    def size(self):
        return self._len // 8

    def __repr__(self):
        return "_FlInt({}, {}, {}, {})".format(
            repr(self._expr_str), repr(self._expr), repr(self._len), self._text_loc
        )


# LEB128 integer.
class _Leb128Int(_Item, _RepableItem, _ExprMixin):
    def __init__(self, expr_str: str, expr: ast.Expression, text_loc: TextLoc):
        super().__init__(text_loc)
        _ExprMixin.__init__(self, expr_str, expr)

    def __repr__(self):
        return "{}({}, {}, {})".format(
            self.__class__.__name__,
            repr(self._expr_str),
            repr(self._expr),
            self._text_loc,
        )


# Unsigned LEB128 integer.
class _ULeb128Int(_Leb128Int, _RepableItem, _ExprMixin):
    pass


# Signed LEB128 integer.
class _SLeb128Int(_Leb128Int, _RepableItem, _ExprMixin):
    pass


# Group of items.
class _Group(_Item, _RepableItem):
    def __init__(self, items: List[_Item], text_loc: TextLoc):
        super().__init__(text_loc)
        self._items = items

    # Contained items.
    @property
    def items(self):
        return self._items

    def __repr__(self):
        return "_Group({}, {})".format(repr(self._items), self._text_loc)


# Repetition item.
class _Rep(_Item, _ExprMixin):
    def __init__(
        self, item: _Item, expr_str: str, expr: ast.Expression, text_loc: TextLoc
    ):
        super().__init__(text_loc)
        _ExprMixin.__init__(self, expr_str, expr)
        self._item = item

    # Item to repeat.
    @property
    def item(self):
        return self._item

    def __repr__(self):
        return "_Rep({}, {}, {}, {})".format(
            repr(self._item), repr(self._expr_str), repr(self._expr), self._text_loc
        )


# Expression item type.
_ExprItemT = Union[_FlInt, _Leb128Int, _VarAssign, _Rep]


# A parsing error containing a message and a text location.
class ParseError(RuntimeError):
    @classmethod
    def _create(cls, msg: str, text_loc: TextLoc):
        self = cls.__new__(cls)
        self._init(msg, text_loc)
        return self

    def __init__(self, *args, **kwargs):  # type: ignore
        raise NotImplementedError

    def _init(self, msg: str, text_loc: TextLoc):
        super().__init__(msg)
        self._text_loc = text_loc

    # Source text location.
    @property
    def text_loc(self):
        return self._text_loc


# Raises a parsing error, forwarding the parameters to the constructor.
def _raise_error(msg: str, text_loc: TextLoc) -> NoReturn:
    raise ParseError._create(msg, text_loc)  # pyright: ignore[reportPrivateUsage]


# Variable/label dictionary type.
SymbolsT = Dict[str, int]


# Python name pattern.
_py_name_pat = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


# Normand parser.
#
# The constructor accepts a Normand input. After building, use the `res`
# property to get the resulting main group.
class _Parser:
    # Builds a parser to parse the Normand input `normand`, parsing
    # immediately.
    def __init__(self, normand: str, variables: SymbolsT, labels: SymbolsT):
        self._normand = normand
        self._at = 0
        self._line_no = 1
        self._col_no = 1
        self._label_names = set(labels.keys())
        self._var_names = set(variables.keys())
        self._parse()

    # Result (main group).
    @property
    def res(self):
        return self._res

    # Current text location.
    @property
    def _text_loc(self):
        return TextLoc._create(  # pyright: ignore[reportPrivateUsage]
            self._line_no, self._col_no
        )

    # Returns `True` if this parser is done parsing.
    def _is_done(self):
        return self._at == len(self._normand)

    # Returns `True` if this parser isn't done parsing.
    def _isnt_done(self):
        return not self._is_done()

    # Raises a parse error, creating it using the message `msg` and the
    # current text location.
    def _raise_error(self, msg: str) -> NoReturn:
        _raise_error(msg, self._text_loc)

    # Tries to make the pattern `pat` match the current substring,
    # returning the match object and updating `self._at`,
    # `self._line_no`, and `self._col_no` on success.
    def _try_parse_pat(self, pat: Pattern[str]):
        m = pat.match(self._normand, self._at)

        if m is None:
            return

        # Skip matched string
        self._at += len(m.group(0))

        # Update line number
        self._line_no += m.group(0).count("\n")

        # Update column number
        for i in reversed(range(self._at)):
            if self._normand[i] == "\n" or i == 0:
                if i == 0:
                    self._col_no = self._at + 1
                else:
                    self._col_no = self._at - i

                break

        # Return match object
        return m

    # Expects the pattern `pat` to match the current substring,
    # returning the match object and updating `self._at`,
    # `self._line_no`, and `self._col_no` on success, or raising a parse
    # error with the message `error_msg` on error.
    def _expect_pat(self, pat: Pattern[str], error_msg: str):
        # Match
        m = self._try_parse_pat(pat)

        if m is None:
            # No match: error
            self._raise_error(error_msg)

        # Return match object
        return m

    # Pattern for _skip_ws_and_comments()
    _ws_or_syms_or_comments_pat = re.compile(
        r"(?:[\s!@/\\?&:;.,+[\]_=|-]|#[^#]*?(?:\n|#))*"
    )

    # Skips as many whitespaces, insignificant symbol characters, and
    # comments as possible.
    def _skip_ws_and_comments(self):
        self._try_parse_pat(self._ws_or_syms_or_comments_pat)

    # Pattern for _try_parse_hex_byte()
    _nibble_pat = re.compile(r"[A-Fa-f0-9]")

    # Tries to parse a hexadecimal byte, returning a byte item on
    # success.
    def _try_parse_hex_byte(self):
        begin_text_loc = self._text_loc

        # Match initial nibble
        m_high = self._try_parse_pat(self._nibble_pat)

        if m_high is None:
            # No match
            return

        # Expect another nibble
        self._skip_ws_and_comments()
        m_low = self._expect_pat(
            self._nibble_pat, "Expecting another hexadecimal nibble"
        )

        # Return item
        return _Byte(int(m_high.group(0) + m_low.group(0), 16), begin_text_loc)

    # Patterns for _try_parse_bin_byte()
    _bin_byte_bit_pat = re.compile(r"[01]")
    _bin_byte_prefix_pat = re.compile(r"%")

    # Tries to parse a binary byte, returning a byte item on success.
    def _try_parse_bin_byte(self):
        begin_text_loc = self._text_loc

        # Match prefix
        if self._try_parse_pat(self._bin_byte_prefix_pat) is None:
            # No match
            return

        # Expect eight bits
        bits = []  # type: List[str]

        for _ in range(8):
            self._skip_ws_and_comments()
            m = self._expect_pat(self._bin_byte_bit_pat, "Expecting a bit (`0` or `1`)")
            bits.append(m.group(0))

        # Return item
        return _Byte(int("".join(bits), 2), begin_text_loc)

    # Patterns for _try_parse_dec_byte()
    _dec_byte_prefix_pat = re.compile(r"\$\s*")
    _dec_byte_val_pat = re.compile(r"(?P<neg>-?)(?P<val>\d+)")

    # Tries to parse a decimal byte, returning a byte item on success.
    def _try_parse_dec_byte(self):
        begin_text_loc = self._text_loc

        # Match prefix
        if self._try_parse_pat(self._dec_byte_prefix_pat) is None:
            # No match
            return

        # Expect the value
        m = self._expect_pat(self._dec_byte_val_pat, "Expecting a decimal constant")

        # Compute value
        val = int(m.group("val")) * (-1 if m.group("neg") == "-" else 1)

        # Validate
        if val < -128 or val > 255:
            _raise_error("Invalid decimal byte value {}".format(val), begin_text_loc)

        # Two's complement
        val %= 256

        # Return item
        return _Byte(val, begin_text_loc)

    # Tries to parse a byte, returning a byte item on success.
    def _try_parse_byte(self):
        # Hexadecimal
        item = self._try_parse_hex_byte()

        if item is not None:
            return item

        # Binary
        item = self._try_parse_bin_byte()

        if item is not None:
            return item

        # Decimal
        item = self._try_parse_dec_byte()

        if item is not None:
            return item

    # Patterns for _try_parse_str()
    _str_prefix_pat = re.compile(r'(?:u(?P<len>16|32)(?P<bo>be|le))?\s*"')
    _str_suffix_pat = re.compile(r'"')
    _str_str_pat = re.compile(r'(?:(?:\\.)|[^"])*')

    # Strings corresponding to escape sequence characters
    _str_escape_seq_strs = {
        "0": "\0",
        "a": "\a",
        "b": "\b",
        "e": "\x1b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "v": "\v",
        "\\": "\\",
        '"': '"',
    }

    # Tries to parse a string, returning a string item on success.
    def _try_parse_str(self):
        begin_text_loc = self._text_loc

        # Match prefix
        m = self._try_parse_pat(self._str_prefix_pat)

        if m is None:
            # No match
            return

        # Get encoding
        encoding = "utf8"

        if m.group("len") is not None:
            encoding = "utf_{}_{}".format(m.group("len"), m.group("bo"))

        # Actual string
        m = self._expect_pat(self._str_str_pat, "Expecting a literal string")

        # Expect end of string
        self._expect_pat(self._str_suffix_pat, 'Expecting `"` (end of literal string)')

        # Replace escape sequences
        val = m.group(0)

        for ec in '0abefnrtv"\\':
            val = val.replace(r"\{}".format(ec), self._str_escape_seq_strs[ec])

        # Encode
        data = val.encode(encoding)

        # Return item
        return _Str(data, begin_text_loc)

    # Patterns for _try_parse_group()
    _group_prefix_pat = re.compile(r"\(")
    _group_suffix_pat = re.compile(r"\)")

    # Tries to parse a group, returning a group item on success.
    def _try_parse_group(self):
        begin_text_loc = self._text_loc

        # Match prefix
        if self._try_parse_pat(self._group_prefix_pat) is None:
            # No match
            return

        # Parse items
        items = self._parse_items()

        # Expect end of group
        self._skip_ws_and_comments()
        self._expect_pat(
            self._group_suffix_pat, "Expecting an item or `)` (end of group)"
        )

        # Return item
        return _Group(items, begin_text_loc)

    # Returns a stripped expression string and an AST expression node
    # from the expression string `expr_str` at text location `text_loc`.
    def _ast_expr_from_str(self, expr_str: str, text_loc: TextLoc):
        # Create an expression node from the expression string
        expr_str = expr_str.strip().replace("\n", " ")

        try:
            expr = ast.parse(expr_str, mode="eval")
        except SyntaxError:
            _raise_error(
                "Invalid expression `{}`: invalid syntax".format(expr_str),
                text_loc,
            )

        return expr_str, expr

    # Patterns for _try_parse_val_and_attr()
    _val_expr_pat = re.compile(r"([^}:]+):\s*")
    _fl_int_len_attr_pat = re.compile(r"8|16|24|32|40|48|56|64")
    _leb128_int_attr_pat = re.compile(r"(u|s)leb128")

    # Tries to parse a value and attribute (fixed length in bits or
    # `leb128`), returning a value item on success.
    def _try_parse_val_and_attr(self):
        begin_text_loc = self._text_loc

        # Match
        m_expr = self._try_parse_pat(self._val_expr_pat)

        if m_expr is None:
            # No match
            return

        # Create an expression node from the expression string
        expr_str, expr = self._ast_expr_from_str(m_expr.group(1), begin_text_loc)

        # Length?
        m_attr = self._try_parse_pat(self._fl_int_len_attr_pat)

        if m_attr is None:
            # LEB128?
            m_attr = self._try_parse_pat(self._leb128_int_attr_pat)

            if m_attr is None:
                # At this point it's invalid
                self._raise_error(
                    "Expecting a length (multiple of eight bits), `uleb128`, or `sleb128`"
                )

            # Return LEB128 integer item
            cls = _ULeb128Int if m_attr.group(1) == "u" else _SLeb128Int
            return cls(expr_str, expr, begin_text_loc)
        else:
            # Return fixed-length integer item
            return _FlInt(
                expr_str,
                expr,
                int(m_attr.group(0)),
                begin_text_loc,
            )

    # Patterns for _try_parse_val_and_attr()
    _var_assign_pat = re.compile(
        r"(?P<name>{})\s*=\s*(?P<expr>[^}}]+)".format(_py_name_pat.pattern)
    )

    # Tries to parse a variable assignment, returning a variable
    # assignment item on success.
    def _try_parse_var_assign(self):
        begin_text_loc = self._text_loc

        # Match
        m = self._try_parse_pat(self._var_assign_pat)

        if m is None:
            # No match
            return

        # Validate name
        name = m.group("name")

        if name == _icitte_name:
            _raise_error(
                "`{}` is a reserved variable name".format(_icitte_name), begin_text_loc
            )

        if name in self._label_names:
            _raise_error("Existing label named `{}`".format(name), begin_text_loc)

        # Add to known variable names
        self._var_names.add(name)

        # Create an expression node from the expression string
        expr_str, expr = self._ast_expr_from_str(m.group("expr"), begin_text_loc)

        # Return item
        return _VarAssign(
            name,
            expr_str,
            expr,
            begin_text_loc,
        )

    # Pattern for _try_parse_set_bo()
    _bo_pat = re.compile(r"[bl]e")

    # Tries to parse a byte order name, returning a byte order setting
    # item on success.
    def _try_parse_set_bo(self):
        begin_text_loc = self._text_loc

        # Match
        m = self._try_parse_pat(self._bo_pat)

        if m is None:
            # No match
            return

        # Return corresponding item
        if m.group(0) == "be":
            return _SetBo(ByteOrder.BE, begin_text_loc)
        else:
            assert m.group(0) == "le"
            return _SetBo(ByteOrder.LE, begin_text_loc)

    # Patterns for _try_parse_val_or_bo()
    _val_var_assign_set_bo_prefix_pat = re.compile(r"\{\s*")
    _val_var_assign_set_bo_suffix_pat = re.compile(r"\s*}")

    # Tries to parse a value, a variable assignment, or a byte order
    # setting, returning an item on success.
    def _try_parse_val_or_var_assign_or_set_bo(self):
        # Match prefix
        if self._try_parse_pat(self._val_var_assign_set_bo_prefix_pat) is None:
            # No match
            return

        # Variable assignment item?
        item = self._try_parse_var_assign()

        if item is None:
            # Fixed-length value item?
            item = self._try_parse_val_and_attr()

            if item is None:
                # Byte order setting item?
                item = self._try_parse_set_bo()

                if item is None:
                    # At this point it's invalid
                    self._raise_error(
                        "Expecting a fixed-length integer, a variable assignment, or a byte order setting"
                    )

        # Expect suffix
        self._expect_pat(self._val_var_assign_set_bo_suffix_pat, "Expecting `}`")
        return item

    # Pattern for _try_parse_set_offset_val() and _try_parse_rep()
    _pos_const_int_pat = re.compile(r"0[Xx][A-Fa-f0-9]+|\d+")

    # Tries to parse an offset setting value (after the initial `<`),
    # returning an offset item on success.
    def _try_parse_set_offset_val(self):
        begin_text_loc = self._text_loc

        # Match
        m = self._try_parse_pat(self._pos_const_int_pat)

        if m is None:
            # No match
            return

        # Return item
        return _SetOffset(int(m.group(0), 0), begin_text_loc)

    # Tries to parse a label name (after the initial `<`), returning a
    # label item on success.
    def _try_parse_label_name(self):
        begin_text_loc = self._text_loc

        # Match
        m = self._try_parse_pat(_py_name_pat)

        if m is None:
            # No match
            return

        # Validate
        name = m.group(0)

        if name == _icitte_name:
            _raise_error(
                "`{}` is a reserved label name".format(_icitte_name), begin_text_loc
            )

        if name in self._label_names:
            _raise_error("Duplicate label name `{}`".format(name), begin_text_loc)

        if name in self._var_names:
            _raise_error("Existing variable named `{}`".format(name), begin_text_loc)

        # Add to known label names
        self._label_names.add(name)

        # Return item
        return _Label(name, begin_text_loc)

    # Patterns for _try_parse_label_or_set_offset()
    _label_set_offset_prefix_pat = re.compile(r"<\s*")
    _label_set_offset_suffix_pat = re.compile(r"\s*>")

    # Tries to parse a label or an offset setting, returning an item on
    # success.
    def _try_parse_label_or_set_offset(self):
        # Match prefix
        if self._try_parse_pat(self._label_set_offset_prefix_pat) is None:
            # No match
            return

        # Offset setting item?
        item = self._try_parse_set_offset_val()

        if item is None:
            # Label item?
            item = self._try_parse_label_name()

            if item is None:
                # At this point it's invalid
                self._raise_error("Expecting a label name or an offset setting value")

        # Expect suffix
        self._expect_pat(self._label_set_offset_suffix_pat, "Expecting `>`")
        return item

    # Tries to parse a base item (anything except a repetition),
    # returning it on success.
    def _try_parse_base_item(self):
        # Byte item?
        item = self._try_parse_byte()

        if item is not None:
            return item

        # String item?
        item = self._try_parse_str()

        if item is not None:
            return item

        # Value, variable assignment, or byte order setting item?
        item = self._try_parse_val_or_var_assign_or_set_bo()

        if item is not None:
            return item

        # Label or offset setting item?
        item = self._try_parse_label_or_set_offset()

        if item is not None:
            return item

        # Group item?
        item = self._try_parse_group()

        if item is not None:
            return item

    # Pattern for _try_parse_rep()
    _rep_prefix_pat = re.compile(r"\*\s*")
    _rep_expr_prefix_pat = re.compile(r"\{")
    _rep_expr_pat = re.compile(r"[^}p]+")
    _rep_expr_suffix_pat = re.compile(r"\}")

    # Tries to parse a repetition, returning the expression string and
    # AST expression node on success.
    def _try_parse_rep(self):
        # Match prefix
        if self._try_parse_pat(self._rep_prefix_pat) is None:
            # No match
            return

        # Expect and return a decimal multiplier
        self._skip_ws_and_comments()

        # Integer?
        m = self._try_parse_pat(self._pos_const_int_pat)

        if m is None:
            # Expression?
            if self._try_parse_pat(self._rep_expr_prefix_pat) is None:
                # At this point it's invalid
                self._raise_error("Expecting a positive integral multiplier or `{`")

            # Expect an expression
            expr_str_loc = self._text_loc
            m = self._expect_pat(self._rep_expr_pat, "Expecting an expression")
            expr_str = self._ast_expr_from_str(m.group(0), expr_str_loc)

            # Expect `}`
            self._expect_pat(self._rep_expr_suffix_pat, "Expecting `}`")
            expr_str = m.group(0)
        else:
            expr_str_loc = self._text_loc
            expr_str = m.group(0)

        return self._ast_expr_from_str(expr_str, expr_str_loc)

    # Tries to parse an item, possibly followed by a repetition,
    # returning `True` on success.
    #
    # Appends any parsed item to `items`.
    def _try_append_item(self, items: List[_Item]):
        self._skip_ws_and_comments()

        # Parse a base item
        item = self._try_parse_base_item()

        if item is None:
            # No item
            return False

        # Parse repetition if the base item is repeatable
        if isinstance(item, _RepableItem):
            self._skip_ws_and_comments()
            rep_text_loc = self._text_loc
            rep_ret = self._try_parse_rep()

            if rep_ret is not None:
                item = _Rep(item, rep_ret[0], rep_ret[1], rep_text_loc)

        items.append(item)
        return True

    # Parses and returns items, skipping whitespaces, insignificant
    # symbols, and comments when allowed, and stopping at the first
    # unknown character.
    def _parse_items(self) -> List[_Item]:
        items = []  # type: List[_Item]

        while self._isnt_done():
            # Try to append item
            if not self._try_append_item(items):
                # Unknown at this point
                break

        return items

    # Parses the whole Normand input, setting `self._res` to the main
    # group item on success.
    def _parse(self):
        if len(self._normand.strip()) == 0:
            # Special case to make sure there's something to consume
            self._res = _Group([], self._text_loc)
            return

        # Parse first level items
        items = self._parse_items()

        # Make sure there's nothing left
        self._skip_ws_and_comments()

        if self._isnt_done():
            self._raise_error(
                "Unexpected character `{}`".format(self._normand[self._at])
            )

        # Set main group item
        self._res = _Group(items, self._text_loc)


# The return type of parse().
class ParseResult:
    @classmethod
    def _create(
        cls,
        data: bytearray,
        variables: SymbolsT,
        labels: SymbolsT,
        offset: int,
        bo: Optional[ByteOrder],
    ):
        self = cls.__new__(cls)
        self._init(data, variables, labels, offset, bo)
        return self

    def __init__(self, *args, **kwargs):  # type: ignore
        raise NotImplementedError

    def _init(
        self,
        data: bytearray,
        variables: SymbolsT,
        labels: SymbolsT,
        offset: int,
        bo: Optional[ByteOrder],
    ):
        self._data = data
        self._vars = variables
        self._labels = labels
        self._offset = offset
        self._bo = bo

    # Generated data.
    @property
    def data(self):
        return self._data

    # Dictionary of updated variable names to their last computed value.
    @property
    def variables(self):
        return self._vars

    # Dictionary of updated main group label names to their computed
    # value.
    @property
    def labels(self):
        return self._labels

    # Updated offset.
    @property
    def offset(self):
        return self._offset

    # Updated byte order.
    @property
    def byte_order(self):
        return self._bo


# Raises a parse error for the item `item`, creating it using the
# message `msg`.
def _raise_error_for_item(msg: str, item: _Item) -> NoReturn:
    _raise_error(msg, item.text_loc)


# The `ICITTE` reserved name.
_icitte_name = "ICITTE"


# Base node visitor.
#
# Calls the _visit_name() method for each name node which isn't the name
# of a call.
class _NodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self._parent_is_call = False

    def generic_visit(self, node: ast.AST):
        if type(node) is ast.Call:
            self._parent_is_call = True
        elif type(node) is ast.Name and not self._parent_is_call:
            self._visit_name(node.id)

        super().generic_visit(node)
        self._parent_is_call = False

    @abc.abstractmethod
    def _visit_name(self, name: str):
        ...


# Expression validator: validates that all the names within the
# expression are allowed.
class _ExprValidator(_NodeVisitor):
    def __init__(self, item: _ExprItemT, allowed_names: Set[str], icitte_allowed: bool):
        super().__init__()
        self._item = item
        self._allowed_names = allowed_names
        self._icitte_allowed = icitte_allowed

    def _visit_name(self, name: str):
        # Make sure the name refers to a known and reachable
        # variable/label name.
        if name == _icitte_name and not self._icitte_allowed:
            _raise_error(
                "Illegal reserved name `{}` in expression `{}`".format(
                    _icitte_name, self._item.expr_str
                ),
                self._item.text_loc,
            )
        elif name != _icitte_name and name not in self._allowed_names:
            msg = "Illegal (unknown or unreachable) variable/label name `{}` in expression `{}`".format(
                name, self._item.expr_str
            )

            allowed_names = self._allowed_names.copy()

            if self._icitte_allowed:
                allowed_names.add(_icitte_name)

            if len(allowed_names) > 0:
                allowed_names_str = ", ".join(
                    sorted(["`{}`".format(name) for name in allowed_names])
                )
                msg += "; the legal names are {{{}}}".format(allowed_names_str)

            _raise_error(
                msg,
                self._item.text_loc,
            )


# Expression visitor getting all the contained names.
class _ExprNamesVisitor(_NodeVisitor):
    def __init__(self):
        self._parent_is_call = False
        self._names = set()  # type: Set[str]

    @property
    def names(self):
        return self._names

    def _visit_name(self, name: str):
        self._names.add(name)


# Generator state.
class _GenState:
    def __init__(
        self,
        variables: SymbolsT,
        labels: SymbolsT,
        offset: int,
        bo: Optional[ByteOrder],
    ):
        self.variables = variables.copy()
        self.labels = labels.copy()
        self.offset = offset
        self.bo = bo


# Generator of data and final state from a group item.
#
# Generation happens in memory at construction time. After building, use
# the `data`, `variables`, `labels`, `offset`, and `bo` properties to
# get the resulting context.
#
# The steps of generation are:
#
# 1. Validate that each repetition and LEB128 integer expression uses
#    only reachable names and not `ICITTE`.
#
# 2. Compute and keep the effective repetition count and LEB128 integer
#    value for each repetition and LEB128 integer instance.
#
# 3. Generate bytes, updating the initial state as it goes which becomes
#    the final state after the operation.
#
#    During the generation, when handling a `_Rep` or `_Leb128Int` item,
#    we already have the effective repetition count or value of the
#    instance.
#
#    When handling a `_Group` item, first update the current labels with
#    all the immediate (not nested) labels, and then handle each
#    contained item. This gives contained item access to "future" outer
#    labels. Then remove the immediate labels from the state so that
#    outer items don't have access to inner labels.
class _Gen:
    def __init__(
        self,
        group: _Group,
        variables: SymbolsT,
        labels: SymbolsT,
        offset: int,
        bo: Optional[ByteOrder],
    ):
        self._validate_vl_exprs(group, set(variables.keys()), set(labels.keys()))
        self._vl_instance_vals = self._compute_vl_instance_vals(
            group, _GenState(variables, labels, offset, bo)
        )
        self._gen(group, _GenState(variables, labels, offset, bo))

    # Generated bytes.
    @property
    def data(self):
        return self._data

    # Updated variables.
    @property
    def variables(self):
        return self._final_state.variables

    # Updated main group labels.
    @property
    def labels(self):
        return self._final_state.labels

    # Updated offset.
    @property
    def offset(self):
        return self._final_state.offset

    # Updated byte order.
    @property
    def bo(self):
        return self._final_state.bo

    # Returns the set of used, non-called names within the AST
    # expression `expr`.
    @staticmethod
    def _names_of_expr(expr: ast.Expression):
        visitor = _ExprNamesVisitor()
        visitor.visit(expr)
        return visitor.names

    # Validates that all the repetition and LEB128 integer expressions
    # within `group` don't refer, directly or indirectly, to subsequent
    # labels.
    #
    # The strategy here is to keep a set of allowed label names, per
    # group, initialized to `allowed_label_names`, and a set of allowed
    # variable names initialized to `allowed_variable_names`.
    #
    # Then, depending on the type of `item`:
    #
    # `_Label`:
    #     Add its name to the local allowed label names: a label
    #     occurring before a repetition, and not within a nested group,
    #     is always reachable.
    #
    # `_VarAssign`:
    #     If all the names within its expression are allowed, then add
    #     its name to the allowed variable names.
    #
    #     Otherwise, remove its name from the allowed variable names (if
    #     it's in there): a variable which refers to an unreachable name
    #     is unreachable itself.
    #
    # `_Rep` and `_Leb128`:
    #     Make sure all the names within its expression are allowed.
    #
    # `_Group`:
    #     Call this function for each contained item with a _copy_ of
    #     the current allowed label names and the same current allowed
    #     variable names.
    @staticmethod
    def _validate_vl_exprs(
        item: _Item, allowed_variable_names: Set[str], allowed_label_names: Set[str]
    ):
        if type(item) is _Label:
            allowed_label_names.add(item.name)
        elif type(item) is _VarAssign:
            # Check if this variable name is allowed
            allowed = True

            for name in _Gen._names_of_expr(item.expr):
                if name not in (
                    allowed_label_names | allowed_variable_names | {_icitte_name}
                ):
                    # Not allowed
                    allowed = False
                    break

            if allowed:
                allowed_variable_names.add(item.name)
            elif item.name in allowed_variable_names:
                allowed_variable_names.remove(item.name)
        elif isinstance(item, _Leb128Int):
            # Validate the expression (`ICITTE` allowed)
            _ExprValidator(
                item, allowed_label_names | allowed_variable_names, True
            ).visit(item.expr)
        elif type(item) is _Rep:
            # Validate the expression first (`ICITTE` not allowed)
            _ExprValidator(
                item, allowed_label_names | allowed_variable_names, False
            ).visit(item.expr)

            # Validate inner item
            _Gen._validate_vl_exprs(
                item.item, allowed_variable_names, allowed_label_names
            )
        elif type(item) is _Group:
            # Copy `allowed_label_names` so that this frame cannot
            # access the nested label names.
            group_allowed_label_names = allowed_label_names.copy()

            for subitem in item.items:
                _Gen._validate_vl_exprs(
                    subitem, allowed_variable_names, group_allowed_label_names
                )

    # Evaluates the expression of `item` considering the current
    # generation state `state`.
    #
    # If `allow_icitte` is `True`, then the `ICITTE` name is available
    # for the expression to evaluate.
    @staticmethod
    def _eval_item_expr(item: _ExprItemT, state: _GenState, allow_icitte: bool):
        syms = state.labels.copy()

        # Set the `ICITTE` name to the current offset, if any
        if allow_icitte:
            syms[_icitte_name] = state.offset

        # Add the current variables
        syms.update(state.variables)

        # Validate the node and its children
        _ExprValidator(item, set(syms.keys()), True).visit(item.expr)

        # Compile and evaluate expression node
        try:
            val = eval(compile(item.expr, "", "eval"), None, syms)
        except Exception as exc:
            _raise_error_for_item(
                "Failed to evaluate expression `{}`: {}".format(item.expr_str, exc),
                item,
            )

        # Validate result
        if type(val) is not int:
            _raise_error_for_item(
                "Invalid expression `{}`: expecting result type `int`, not `{}`".format(
                    item.expr_str, type(val).__name__
                ),
                item,
            )

        return val

    # Returns the size, in bytes, required to encode the value `val`
    # with LEB128 (signed version if `is_signed` is `True`).
    @staticmethod
    def _leb128_size_for_val(val: int, is_signed: bool):
        if val < 0:
            # Equivalent upper bound.
            #
            # For example, if `val` is -128, then the full integer for
            # this number of bits would be [-128, 127].
            val = -val - 1

        # Number of bits (add one for the sign if needed)
        bits = val.bit_length() + int(is_signed)

        if bits == 0:
            bits = 1

        # Seven bits per byte
        return math.ceil(bits / 7)

    # Computes the effective value for each repetition and LEB128
    # integer instance, filling `instance_vals` (if not `None`) and
    # returning `instance_vals`.
    #
    # At this point it must be known that, for a given variable-length
    # item, its expression only contains reachable names.
    #
    # When handling a `_Rep` item, this function appends its effective
    # multiplier to `instance_vals` _before_ handling its repeated item.
    #
    # When handling a `_VarAssign` item, this function only evaluates it
    # if all its names are reachable.
    @staticmethod
    def _compute_vl_instance_vals(
        item: _Item, state: _GenState, instance_vals: Optional[List[int]] = None
    ):
        if instance_vals is None:
            instance_vals = []

        if isinstance(item, _ScalarItem):
            state.offset += item.size
        elif type(item) is _Label:
            state.labels[item.name] = state.offset
        elif type(item) is _VarAssign:
            # Check if all the names are reachable
            do_eval = True

            for name in _Gen._names_of_expr(item.expr):
                if (
                    name != _icitte_name
                    and name not in state.variables
                    and name not in state.labels
                ):
                    # A name is unknown: cannot evaluate
                    do_eval = False
                    break

            if do_eval:
                # Evaluate the expression and keep the result
                state.variables[item.name] = _Gen._eval_item_expr(item, state, True)
        elif type(item) is _SetOffset:
            state.offset = item.val
        elif isinstance(item, _Leb128Int):
            # Evaluate the expression
            val = _Gen._eval_item_expr(item, state, True)

            # Validate result
            if type(item) is _ULeb128Int and val < 0:
                _raise_error_for_item(
                    "Invalid expression `{}`: unexpected negative result {:,} for a ULEB128 encoding".format(
                        item.expr_str, val
                    ),
                    item,
                )

            # Add the evaluation result to the to variable-length item
            # instance values.
            instance_vals.append(val)

            # Update offset
            state.offset += _Gen._leb128_size_for_val(val, type(item) is _SLeb128Int)
        elif type(item) is _Rep:
            # Evaluate the expression and keep the result
            val = _Gen._eval_item_expr(item, state, False)

            # Validate result
            if val < 0:
                _raise_error_for_item(
                    "Invalid expression `{}`: unexpected negative result {:,}".format(
                        item.expr_str, val
                    ),
                    item,
                )

            # Add to repetition instance values
            instance_vals.append(val)

            # Process the repeated item `val` times
            for _ in range(val):
                _Gen._compute_vl_instance_vals(item.item, state, instance_vals)
        elif type(item) is _Group:
            prev_labels = state.labels.copy()

            # Process each item
            for subitem in item.items:
                _Gen._compute_vl_instance_vals(subitem, state, instance_vals)

            state.labels = prev_labels

        return instance_vals

    def _zero_item_size(self, item: _Item, next_vl_instance: int):
        return 0, next_vl_instance

    def _scalar_item_size(self, item: _ScalarItem, next_vl_instance: int):
        return item.size, next_vl_instance

    def _leb128_int_item_size(self, item: _Leb128Int, next_vl_instance: int):
        # Get the value from `self._vl_instance_vals` _before_
        # incrementing `next_vl_instance` to honor the order of
        # _compute_vl_instance_vals().
        return (
            self._leb128_size_for_val(
                self._vl_instance_vals[next_vl_instance], type(item) is _SLeb128Int
            ),
            next_vl_instance + 1,
        )

    def _group_item_size(self, item: _Group, next_vl_instance: int):
        size = 0

        for subitem in item.items:
            subitem_size, next_vl_instance = self._item_size(subitem, next_vl_instance)
            size += subitem_size

        return size, next_vl_instance

    def _rep_item_size(self, item: _Rep, next_vl_instance: int):
        # Get the value from `self._vl_instance_vals` _before_
        # incrementing `next_vl_instance` to honor the order of
        # _compute_vl_instance_vals().
        mul = self._vl_instance_vals[next_vl_instance]
        next_vl_instance += 1
        size = 0

        for _ in range(mul):
            iter_size, next_vl_instance = self._item_size(item.item, next_vl_instance)
            size += iter_size

        return size, next_vl_instance

    # Returns the size of `item` and the new next repetition instance.
    def _item_size(self, item: _Item, next_vl_instance: int):
        return self._item_size_funcs[type(item)](item, next_vl_instance)

    # Handles the byte item `item`.
    def _handle_byte_item(self, item: _Byte, state: _GenState, next_vl_instance: int):
        self._data.append(item.val)
        state.offset += item.size
        return next_vl_instance

    # Handles the string item `item`.
    def _handle_str_item(self, item: _Str, state: _GenState, next_vl_instance: int):
        self._data += item.data
        state.offset += item.size
        return next_vl_instance

    # Handles the byte order setting item `item`.
    def _handle_set_bo_item(
        self, item: _SetBo, state: _GenState, next_vl_instance: int
    ):
        # Update current byte order
        state.bo = item.bo
        return next_vl_instance

    # Handles the variable assignment item `item`.
    def _handle_var_assign_item(
        self, item: _VarAssign, state: _GenState, next_vl_instance: int
    ):
        # Update variable
        state.variables[item.name] = self._eval_item_expr(item, state, True)
        return next_vl_instance

    # Handles the fixed-length integer item `item`.
    def _handle_fl_int_item(
        self, item: _FlInt, state: _GenState, next_vl_instance: int
    ):
        # Compute value
        val = self._eval_item_expr(item, state, True)

        # Validate range
        if val < -(2 ** (item.len - 1)) or val > 2**item.len - 1:
            _raise_error_for_item(
                "Value {:,} is outside the {}-bit range when evaluating expression `{}` at byte offset {:,}".format(
                    val, item.len, item.expr_str, state.offset
                ),
                item,
            )

        # Encode result on 64 bits (to extend the sign bit whatever the
        # value of `item.len`).
        if state.bo is None and item.len > 8:
            _raise_error_for_item(
                "Current byte order isn't defined at first value (`{}`) to encode on more than 8 bits".format(
                    item.expr_str
                ),
                item,
            )

        data = struct.pack(
            "{}{}".format(
                ">" if state.bo in (None, ByteOrder.BE) else "<",
                "Q" if val >= 0 else "q",
            ),
            val,
        )

        # Keep only the requested length
        len_bytes = item.len // 8

        if state.bo in (None, ByteOrder.BE):
            # Big endian: keep last bytes
            data = data[-len_bytes:]
        else:
            # Little endian: keep first bytes
            assert state.bo == ByteOrder.LE
            data = data[:len_bytes]

        # Append to current bytes and update offset
        self._data += data
        state.offset += len(data)
        return next_vl_instance

    # Handles the LEB128 integer item `item`.
    def _handle_leb128_int_item(
        self, item: _Leb128Int, state: _GenState, next_vl_instance: int
    ):
        # Get the precomputed value
        val = self._vl_instance_vals[next_vl_instance]

        # Size in bytes
        size = self._leb128_size_for_val(val, type(item) is _SLeb128Int)

        # For each byte
        for _ in range(size):
            # Seven LSBs, MSB of the byte set (continue)
            self._data.append((val & 0x7F) | 0x80)
            val >>= 7

        # Clear MSB of last byte (stop)
        self._data[-1] &= ~0x80

        # Consumed this instance
        return next_vl_instance + 1

    # Handles the group item `item`, only removing the immediate labels
    # from `state.labels` if `remove_immediate_labels` is `True`.
    def _handle_group_item(
        self,
        item: _Group,
        state: _GenState,
        next_vl_instance: int,
        remove_immediate_labels: bool = True,
    ):
        # Compute the values of the immediate (not nested) labels. Those
        # labels are reachable by any expression within the group.
        offset = state.offset
        immediate_label_names = set()  # type: Set[str]
        tmp_next_vl_instance = next_vl_instance

        for subitem in item.items:
            if type(subitem) is _SetOffset:
                # Update offset
                offset = subitem.val
            elif type(subitem) is _Label:
                # New immediate label
                state.labels[subitem.name] = offset
                immediate_label_names.add(subitem.name)

            subitem_size, tmp_next_vl_instance = self._item_size(
                subitem, tmp_next_vl_instance
            )
            offset += subitem_size

        # Handle each item now with the actual state
        for subitem in item.items:
            next_vl_instance = self._handle_item(subitem, state, next_vl_instance)

        # Remove immediate labels if required so that outer items won't
        # reach inner labels.
        if remove_immediate_labels:
            for name in immediate_label_names:
                del state.labels[name]

        return next_vl_instance

    # Handles the repetition item `item`.
    def _handle_rep_item(self, item: _Rep, state: _GenState, next_vl_instance: int):
        # Get the precomputed repetition count
        mul = self._vl_instance_vals[next_vl_instance]

        # Consumed this instance
        next_vl_instance += 1

        for _ in range(mul):
            next_vl_instance = self._handle_item(item.item, state, next_vl_instance)

        return next_vl_instance

    # Handles the offset setting item `item`.
    def _handle_set_offset_item(
        self, item: _SetOffset, state: _GenState, next_vl_instance: int
    ):
        state.offset = item.val
        return next_vl_instance

    # Handles the label item `item`.
    def _handle_label_item(self, item: _Label, state: _GenState, next_vl_instance: int):
        return next_vl_instance

    # Handles the item `item`, returning the updated next repetition
    # instance.
    def _handle_item(self, item: _Item, state: _GenState, next_vl_instance: int):
        return self._item_handlers[type(item)](item, state, next_vl_instance)

    # Generates the data (`self._data`) and final state
    # (`self._final_state`) from `group` and the initial state `state`.
    def _gen(self, group: _Group, state: _GenState):
        # Initial state
        self._data = bytearray()

        # Item handlers
        self._item_handlers = {
            _Byte: self._handle_byte_item,
            _FlInt: self._handle_fl_int_item,
            _Group: self._handle_group_item,
            _Label: self._handle_label_item,
            _Rep: self._handle_rep_item,
            _SetBo: self._handle_set_bo_item,
            _SetOffset: self._handle_set_offset_item,
            _SLeb128Int: self._handle_leb128_int_item,
            _Str: self._handle_str_item,
            _ULeb128Int: self._handle_leb128_int_item,
            _VarAssign: self._handle_var_assign_item,
        }  # type: Dict[type, Callable[[Any, _GenState, int], int]]

        # Item size getters
        self._item_size_funcs = {
            _Byte: self._scalar_item_size,
            _FlInt: self._scalar_item_size,
            _Group: self._group_item_size,
            _Label: self._zero_item_size,
            _Rep: self._rep_item_size,
            _SetBo: self._zero_item_size,
            _SetOffset: self._zero_item_size,
            _SLeb128Int: self._leb128_int_item_size,
            _Str: self._scalar_item_size,
            _ULeb128Int: self._leb128_int_item_size,
            _VarAssign: self._zero_item_size,
        }  # type: Dict[type, Callable[[Any, int], Tuple[int, int]]]

        # Handle the group item, _not_ removing the immediate labels
        # because the `labels` property offers them.
        self._handle_group_item(group, state, 0, False)

        # This is actually the final state
        self._final_state = state


# Returns a `ParseResult` instance containing the bytes encoded by the
# input string `normand`.
#
# `init_variables` is a dictionary of initial variable names (valid
# Python names) to integral values. A variable name must not be the
# reserved name `ICITTE`.
#
# `init_labels` is a dictionary of initial label names (valid Python
# names) to integral values. A label name must not be the reserved name
# `ICITTE`.
#
# `init_offset` is the initial offset.
#
# `init_byte_order` is the initial byte order.
#
# Raises `ParseError` on any parsing error.
def parse(
    normand: str,
    init_variables: Optional[SymbolsT] = None,
    init_labels: Optional[SymbolsT] = None,
    init_offset: int = 0,
    init_byte_order: Optional[ByteOrder] = None,
):
    if init_variables is None:
        init_variables = {}

    if init_labels is None:
        init_labels = {}

    gen = _Gen(
        _Parser(normand, init_variables, init_labels).res,
        init_variables,
        init_labels,
        init_offset,
        init_byte_order,
    )
    return ParseResult._create(  # pyright: ignore[reportPrivateUsage]
        gen.data, gen.variables, gen.labels, gen.offset, gen.bo
    )


# Parses the command-line arguments.
def _parse_cli_args():
    import argparse

    # Build parser
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--offset",
        metavar="OFFSET",
        action="store",
        type=int,
        default=0,
        help="initial offset (positive)",
    )
    ap.add_argument(
        "-b",
        "--byte-order",
        metavar="BO",
        choices=["be", "le"],
        type=str,
        help="initial byte order (`be` or `le`)",
    )
    ap.add_argument(
        "--var",
        metavar="NAME=VAL",
        action="append",
        help="add an initial variable (may be repeated)",
    )
    ap.add_argument(
        "-l",
        "--label",
        metavar="NAME=VAL",
        action="append",
        help="add an initial label (may be repeated)",
    )
    ap.add_argument(
        "--version", action="version", version="Normand {}".format(__version__)
    )
    ap.add_argument(
        "path",
        metavar="PATH",
        action="store",
        nargs="?",
        help="input path (none means standard input)",
    )

    # Parse
    return ap.parse_args()


# Raises a command-line error with the message `msg`.
def _raise_cli_error(msg: str) -> NoReturn:
    raise RuntimeError("Command-line error: {}".format(msg))


# Returns a dictionary of string to integers from the list of strings
# `args` containing `NAME=VAL` entries.
def _dict_from_arg(args: Optional[List[str]]):
    d = {}  # type: Dict[str, int]

    if args is None:
        return d

    for arg in args:
        m = re.match(r"({})=(\d+)$".format(_py_name_pat.pattern), arg)

        if m is None:
            _raise_cli_error("Invalid assignment {}".format(arg))

        d[m.group(1)] = int(m.group(2))

    return d


# CLI entry point without exception handling.
def _try_run_cli():
    import os.path

    # Parse arguments
    args = _parse_cli_args()

    # Read input
    if args.path is None:
        normand = sys.stdin.read()
    else:
        with open(args.path) as f:
            normand = f.read()

    # Variables and labels
    variables = _dict_from_arg(args.var)
    labels = _dict_from_arg(args.label)

    # Validate offset
    if args.offset < 0:
        _raise_cli_error("Invalid negative offset {}")

    # Validate and set byte order
    bo = None  # type: Optional[ByteOrder]

    if args.byte_order is not None:
        if args.byte_order == "be":
            bo = ByteOrder.BE
        else:
            assert args.byte_order == "le"
            bo = ByteOrder.LE

    # Parse
    try:
        res = parse(normand, variables, labels, args.offset, bo)
    except ParseError as exc:
        prefix = ""

        if args.path is not None:
            prefix = "{}:".format(os.path.abspath(args.path))

        _fail(
            "{}{}:{} - {}".format(
                prefix, exc.text_loc.line_no, exc.text_loc.col_no, str(exc)
            )
        )

    # Print
    sys.stdout.buffer.write(res.data)


# Prints the exception message `msg` and exits with status 1.
def _fail(msg: str) -> NoReturn:
    if not msg.endswith("."):
        msg += "."

    print(msg, file=sys.stderr)
    sys.exit(1)


# CLI entry point.
def _run_cli():
    try:
        _try_run_cli()
    except Exception as exc:
        _fail(str(exc))


if __name__ == "__main__":
    _run_cli()
