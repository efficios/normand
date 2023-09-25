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

__author__ = "Philippe Proulx"
__version__ = "0.1.0"
__all__ = [
    "ByteOrder",
    "parse",
    "ParseError",
    "ParseResult",
    "TextLoc",
    "VarsT",
    "__author__",
    "__version__",
]

import re
import abc
import ast
import sys
import enum
import struct
from typing import Any, Dict, List, Union, Pattern, Callable, NoReturn, Optional


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


# Any item.
class _Item:
    def __init__(self, text_loc: TextLoc):
        self._text_loc = text_loc

    # Source text location.
    @property
    def text_loc(self):
        return self._text_loc

    # Returns the size, in bytes, of this item.
    @property
    @abc.abstractmethod
    def size(self) -> int:
        ...


# A repeatable item.
class _RepableItem(_Item):
    pass


# Single byte.
class _Byte(_RepableItem):
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
class _Str(_RepableItem):
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


# Byte order.
class _Bo(_Item):
    def __init__(self, bo: ByteOrder):
        self._bo = bo

    @property
    def bo(self):
        return self._bo

    @property
    def size(self):
        return 0


# Label.
class _Label(_Item):
    def __init__(self, name: str, text_loc: TextLoc):
        super().__init__(text_loc)
        self._name = name

    # Label name.
    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return 0

    def __repr__(self):
        return "_Label({}, {})".format(repr(self._name), self._text_loc)


# Offset.
class _Offset(_Item):
    def __init__(self, val: int, text_loc: TextLoc):
        super().__init__(text_loc)
        self._val = val

    # Offset value.
    @property
    def val(self):
        return self._val

    @property
    def size(self):
        return 0

    def __repr__(self):
        return "_Offset({}, {})".format(repr(self._val), self._text_loc)


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


# Variable.
class _Var(_Item, _ExprMixin):
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

    @property
    def size(self):
        return 0

    def __repr__(self):
        return "_Var({}, {}, {}, {})".format(
            repr(self._name), repr(self._expr_str), repr(self._expr), self._text_loc
        )


# Value, possibly needing more than one byte.
class _Val(_RepableItem, _ExprMixin):
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
        return "_Val({}, {}, {}, {})".format(
            repr(self._expr_str), repr(self._expr), repr(self._len), self._text_loc
        )


# Expression item type.
_ExprItemT = Union[_Val, _Var]


# Group of items.
class _Group(_RepableItem):
    def __init__(self, items: List[_Item], text_loc: TextLoc):
        super().__init__(text_loc)
        self._items = items
        self._size = sum([item.size for item in self._items])

    # Contained items.
    @property
    def items(self):
        return self._items

    @property
    def size(self):
        return self._size

    def __repr__(self):
        return "_Group({}, {})".format(repr(self._items), self._text_loc)


# Repetition item.
class _Rep(_Item):
    def __init__(self, item: _RepableItem, mul: int, text_loc: TextLoc):
        super().__init__(text_loc)
        self._item = item
        self._mul = mul

    # Item to repeat.
    @property
    def item(self):
        return self._item

    # Repetition multiplier.
    @property
    def mul(self):
        return self._mul

    @property
    def size(self):
        return self._item.size * self._mul

    def __repr__(self):
        return "_Rep({}, {}, {})".format(
            repr(self._item), repr(self._mul), self._text_loc
        )


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


# Variable (and label) dictionary type.
VarsT = Dict[str, int]


# Python name pattern.
_py_name_pat = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


# Normand parser.
#
# The constructor accepts a Normand input. After building, use the `res`
# property to get the resulting main group.
class _Parser:
    # Builds a parser to parse the Normand input `normand`, parsing
    # immediately.
    def __init__(self, normand: str, variables: VarsT, labels: VarsT):
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
        return _Byte(int(m_high.group(0) + m_low.group(0), 16), self._text_loc)

    # Patterns for _try_parse_bin_byte()
    _bin_byte_bit_pat = re.compile(r"[01]")
    _bin_byte_prefix_pat = re.compile(r"%")

    # Tries to parse a binary byte, returning a byte item on success.
    def _try_parse_bin_byte(self):
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
        return _Byte(int("".join(bits), 2), self._text_loc)

    # Patterns for _try_parse_dec_byte()
    _dec_byte_prefix_pat = re.compile(r"\$\s*")
    _dec_byte_val_pat = re.compile(r"(?P<neg>-?)(?P<val>\d+)")

    # Tries to parse a decimal byte, returning a byte item on success.
    def _try_parse_dec_byte(self):
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
            self._raise_error("Invalid decimal byte value {}".format(val))

        # Two's complement
        val = val % 256

        # Return item
        return _Byte(val, self._text_loc)

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
        return _Str(data, self._text_loc)

    # Patterns for _try_parse_group()
    _group_prefix_pat = re.compile(r"\(")
    _group_suffix_pat = re.compile(r"\)")

    # Tries to parse a group, returning a group item on success.
    def _try_parse_group(self):
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
        return _Group(items, self._text_loc)

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

    # Patterns for _try_parse_val_and_len()
    _val_expr_pat = re.compile(r"([^}:]+):")
    _val_len_pat = re.compile(r"\s*(8|16|24|32|40|48|56|64)")

    # Tries to parse a value and length, returning a value item on
    # success.
    def _try_parse_val_and_len(self):
        begin_text_loc = self._text_loc

        # Match
        m_expr = self._try_parse_pat(self._val_expr_pat)

        if m_expr is None:
            # No match
            return

        # Expect a length
        m_len = self._expect_pat(
            self._val_len_pat, "Expecting a length (multiple of eight bits)"
        )

        # Create an expression node from the expression string
        expr_str, expr = self._ast_expr_from_str(m_expr.group(1), begin_text_loc)

        # Return item
        return _Val(
            expr_str,
            expr,
            int(m_len.group(1)),
            self._text_loc,
        )

    # Patterns for _try_parse_val_and_len()
    _var_pat = re.compile(
        r"(?P<name>{})\s*=\s*(?P<expr>[^}}]+)".format(_py_name_pat.pattern)
    )

    # Tries to parse a variable, returning a variable item on success.
    def _try_parse_var(self):
        begin_text_loc = self._text_loc

        # Match
        m = self._try_parse_pat(self._var_pat)

        if m is None:
            # No match
            return

        # Validate name
        name = m.group("name")

        if name == _icitte_name:
            self._raise_error("`{}` is a reserved variable name".format(_icitte_name))

        if name in self._label_names:
            self._raise_error("Existing label named `{}`".format(name))

        # Add to known variable names
        self._var_names.add(name)

        # Create an expression node from the expression string
        expr_str, expr = self._ast_expr_from_str(m.group("expr"), begin_text_loc)

        # Return item
        return _Var(
            name,
            expr_str,
            expr,
            self._text_loc,
        )

    # Pattern for _try_parse_bo_name()
    _bo_pat = re.compile(r"[bl]e")

    # Tries to parse a byte order name, returning a byte order item on
    # success.
    def _try_parse_bo_name(self):
        # Match
        m = self._try_parse_pat(self._bo_pat)

        if m is None:
            # No match
            return

        # Return corresponding item
        if m.group(0) == "be":
            return _Bo(ByteOrder.BE)
        else:
            assert m.group(0) == "le"
            return _Bo(ByteOrder.LE)

    # Patterns for _try_parse_val_or_bo()
    _val_var_bo_prefix_pat = re.compile(r"\{\s*")
    _val_var_bo_suffix_pat = re.compile(r"\s*}")

    # Tries to parse a value, a variable, or a byte order, returning an
    # item on success.
    def _try_parse_val_or_var_or_bo(self):
        # Match prefix
        if self._try_parse_pat(self._val_var_bo_prefix_pat) is None:
            # No match
            return

        # Variable item?
        item = self._try_parse_var()

        if item is None:
            # Value item?
            item = self._try_parse_val_and_len()

            if item is None:
                # Byte order item?
                item = self._try_parse_bo_name()

                if item is None:
                    # At this point it's invalid
                    self._raise_error("Expecting a value, a variable, or a byte order")

        # Expect suffix
        self._expect_pat(self._val_var_bo_suffix_pat, "Expecting `}`")
        return item

    # Pattern for _try_parse_offset_val() and _try_parse_rep()
    _pos_const_int_pat = re.compile(r"0[Xx][A-Fa-f0-9]+|\d+")

    # Tries to parse an offset value (after the initial `<`), returning
    # an offset item on success.
    def _try_parse_offset_val(self):
        # Match
        m = self._try_parse_pat(self._pos_const_int_pat)

        if m is None:
            # No match
            return

        # Return item
        return _Offset(int(m.group(0), 0), self._text_loc)

    # Tries to parse a label name (after the initial `<`), returning a
    # label item on success.
    def _try_parse_label_name(self):
        # Match
        m = self._try_parse_pat(_py_name_pat)

        if m is None:
            # No match
            return

        # Validate
        name = m.group(0)

        if name == _icitte_name:
            self._raise_error("`{}` is a reserved label name".format(_icitte_name))

        if name in self._label_names:
            self._raise_error("Duplicate label name `{}`".format(name))

        if name in self._var_names:
            self._raise_error("Existing variable named `{}`".format(name))

        # Add to known label names
        self._label_names.add(name)

        # Return item
        return _Label(name, self._text_loc)

    # Patterns for _try_parse_label_or_offset()
    _label_offset_prefix_pat = re.compile(r"<\s*")
    _label_offset_suffix_pat = re.compile(r"\s*>")

    # Tries to parse a label or an offset, returning an item on success.
    def _try_parse_label_or_offset(self):
        # Match prefix
        if self._try_parse_pat(self._label_offset_prefix_pat) is None:
            # No match
            return

        # Offset item?
        item = self._try_parse_offset_val()

        if item is None:
            # Label item?
            item = self._try_parse_label_name()

            if item is None:
                # At this point it's invalid
                self._raise_error("Expecting a label name or an offset value")

        # Expect suffix
        self._expect_pat(self._label_offset_suffix_pat, "Expecting `>`")
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

        # Value, variable, or byte order item?
        item = self._try_parse_val_or_var_or_bo()

        if item is not None:
            return item

        # Label or offset item?
        item = self._try_parse_label_or_offset()

        if item is not None:
            return item

        # Group item?
        item = self._try_parse_group()

        if item is not None:
            return item

    # Pattern for _try_parse_rep()
    _rep_prefix_pat = re.compile(r"\*\s*")

    # Tries to parse a repetition, returning the multiplier on success,
    # or 1 otherwise.
    def _try_parse_rep(self):
        self._skip_ws_and_comments()

        # Match prefix
        if self._try_parse_pat(self._rep_prefix_pat) is None:
            # No match
            return 1

        # Expect and return a decimal multiplier
        self._skip_ws_and_comments()
        m = self._expect_pat(
            self._pos_const_int_pat, "Expecting a positive integral multiplier"
        )
        return int(m.group(0), 0)

    # Tries to parse a repeatable item followed or not by a repetition,
    # returning an item on success.
    def _try_parse_item(self):
        self._skip_ws_and_comments()

        # Parse a base item
        item = self._try_parse_base_item()

        if item is None:
            # No item
            return

        # Parse repetition if the base item is repeatable
        if isinstance(item, _RepableItem):
            rep = self._try_parse_rep()

            if rep == 0:
                # No item
                return
            elif rep > 1:
                # Convert to repetition item
                item = _Rep(item, rep, self._text_loc)

        return item

    # Parses and returns items, skipping whitespaces, insignificant
    # symbols, and comments when allowed, and stopping at the first
    # unknown character.
    def _parse_items(self) -> List[_Item]:
        items = []  # type: List[_Item]

        while self._isnt_done():
            # Try to parse item
            item = self._try_parse_item()

            if item is not None:
                # Append new item
                items.append(item)
                continue

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
        variables: VarsT,
        labels: VarsT,
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
        variables: VarsT,
        labels: VarsT,
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


# Value expression validator.
class _ExprValidator(ast.NodeVisitor):
    def __init__(self, item: _ExprItemT, syms: VarsT):
        self._item = item
        self._syms = syms
        self._parent_is_call = False

    def generic_visit(self, node: ast.AST):
        if type(node) is ast.Call:
            self._parent_is_call = True
        elif type(node) is ast.Name and not self._parent_is_call:
            # Make sure the name refers to a known label name
            if node.id != _icitte_name and node.id not in self._syms:
                _raise_error(
                    "Unknown variable/label name `{}` in expression `{}`".format(
                        node.id, self._item.expr_str
                    ),
                    self._item.text_loc,
                )

        # TODO: Restrict the set of allowed node types

        super().generic_visit(node)
        self._parent_is_call = False


# Keeper of labels for a given group instance.
#
# A group instance is one iteration of a given group.
class _GroupInstanceLabels:
    def __init__(self):
        self._instance_labels = {}  # type: Dict[_Group, Dict[int, VarsT]]

    # Assigns the labels `labels` to a new instance of `group`.
    def add(self, group: _Group, labels: VarsT):
        if group not in self._instance_labels:
            self._instance_labels[group] = {}

        spec_instance_labels = self._instance_labels[group]
        spec_instance_labels[len(spec_instance_labels)] = labels.copy()

    # Returns the labels (not a copy) of the instance `instance_index`
    # of the group `group`.
    def labels(self, group: _Group, instance_index: int):
        return self._instance_labels[group][instance_index]


# Generator of data and labels from a group item.
#
# Generation happens in memory at construction time. After building, use
# the `data`, `variables`, `labels`, `offset`, and `bo` properties to
# get the resulting context.
class _Gen:
    def __init__(
        self,
        group: _Group,
        variables: VarsT,
        labels: VarsT,
        offset: int,
        bo: Optional[ByteOrder],
    ):
        self._group_instance_labels = _GroupInstanceLabels()
        self._resolve_labels(group, offset, labels.copy())
        self._vars = variables.copy()
        self._offset = offset
        self._bo = bo
        self._main_group = group
        self._gen()

    # Generated bytes.
    @property
    def data(self):
        return self._data

    # Updated variables.
    @property
    def variables(self):
        return self._vars

    # Updated main group labels.
    @property
    def labels(self):
        return self._group_instance_labels.labels(self._main_group, 0)

    # Updated offset.
    @property
    def offset(self):
        return self._offset

    # Updated byte order.
    @property
    def bo(self):
        return self._bo

    # Fills `self._group_instance_labels` with the labels for each group
    # instance in `item`, starting at current offset `offset` with the
    # current labels `labels`.
    #
    # Returns the new current offset.
    def _resolve_labels(self, item: _Item, offset: int, labels: VarsT) -> int:
        if type(item) is _Group:
            # First pass: compute immediate labels of this instance
            group_labels = labels.copy()
            group_offset = offset

            for subitem in item.items:
                if type(subitem) is _Offset:
                    group_offset = subitem.val
                elif type(subitem) is _Label:
                    assert subitem.name not in group_labels
                    group_labels[subitem.name] = group_offset
                else:
                    group_offset += subitem.size

            # Add to group instance labels
            self._group_instance_labels.add(item, group_labels)

            # Second pass: handle each item
            for subitem in item.items:
                offset = self._resolve_labels(subitem, offset, group_labels)
        elif type(item) is _Rep:
            for _ in range(item.mul):
                offset = self._resolve_labels(item.item, offset, labels)
        elif type(item) is _Offset:
            offset = item.val
        else:
            offset += item.size

        return offset

    def _handle_byte_item(self, item: _Byte):
        self._data.append(item.val)
        self._offset += item.size

    def _handle_str_item(self, item: _Str):
        self._data += item.data
        self._offset += item.size

    def _handle_bo_item(self, item: _Bo):
        self._bo = item.bo

    def _eval_expr(self, item: _ExprItemT):
        # Get the labels of the current group instance as the initial
        # symbols (copied because we're adding stuff).
        assert self._cur_group is not None
        syms = self._group_instance_labels.labels(
            self._cur_group, self._group_instance_indexes[self._cur_group]
        ).copy()

        # Set the `ICITTE` name to the current offset (before encoding)
        syms[_icitte_name] = self._offset

        # Add the current variables
        syms.update(self._vars)

        # Validate the node and its children
        _ExprValidator(item, syms).visit(item.expr)

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
                "Invalid expression `{}`: unexpected result type `{}`".format(
                    item.expr_str, type(val).__name__
                ),
                item,
            )

        return val

    def _handle_var_item(self, item: _Var):
        # Update variable
        self._vars[item.name] = self._eval_expr(item)

    def _handle_val_item(self, item: _Val):
        # Compute value
        val = self._eval_expr(item)

        # Validate range
        if val < -(2 ** (item.len - 1)) or val > 2**item.len - 1:
            _raise_error_for_item(
                "Value {:,} is outside the {}-bit range when evaluating expression `{}` at byte offset {:,}".format(
                    val, item.len, item.expr_str, self._offset
                ),
                item,
            )

        # Encode result on 64 bits (to extend the sign bit whatever the
        # value of `item.len`).
        if self._bo is None and item.len > 8:
            _raise_error_for_item(
                "Current byte order isn't defined at first value (`{}`) to encode on more than 8 bits".format(
                    item.expr_str
                ),
                item,
            )

        data = struct.pack(
            "{}{}".format(
                ">" if self._bo in (None, ByteOrder.BE) else "<",
                "Q" if val >= 0 else "q",
            ),
            val,
        )

        # Keep only the requested length
        len_bytes = item.len // 8

        if self._bo in (None, ByteOrder.BE):
            # Big endian: keep last bytes
            data = data[-len_bytes:]
        else:
            # Little endian: keep first bytes
            assert self._bo == ByteOrder.LE
            data = data[:len_bytes]

        # Append to current bytes and update offset
        self._data += data
        self._offset += len(data)

    def _handle_group_item(self, item: _Group):
        # Update the instance index of `item`
        if item not in self._group_instance_indexes:
            self._group_instance_indexes[item] = 0
        else:
            self._group_instance_indexes[item] += 1

        # Changed current group
        old_cur_group = self._cur_group
        self._cur_group = item

        # Handle each item
        for subitem in item.items:
            self._handle_item(subitem)

        # Restore current group
        self._cur_group = old_cur_group

    def _handle_rep_item(self, item: _Rep):
        for _ in range(item.mul):
            self._handle_item(item.item)

    def _handle_offset_item(self, item: _Offset):
        self._offset = item.val

    def _handle_item(self, item: _Item):
        if type(item) in self._item_handlers:
            self._item_handlers[type(item)](item)

    def _gen(self):
        # Initial state
        self._data = bytearray()
        self._group_instance_indexes = {}  # type: Dict[_Group, int]
        self._cur_group = None

        # Item handlers
        self._item_handlers = {
            _Byte: self._handle_byte_item,
            _Str: self._handle_str_item,
            _Bo: self._handle_bo_item,
            _Val: self._handle_val_item,
            _Var: self._handle_var_item,
            _Group: self._handle_group_item,
            _Rep: self._handle_rep_item,
            _Offset: self._handle_offset_item,
        }  # type: Dict[type, Callable[[Any], None]]

        # Handle the group item
        self._handle_item(self._main_group)


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
    init_variables: Optional[VarsT] = None,
    init_labels: Optional[VarsT] = None,
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
