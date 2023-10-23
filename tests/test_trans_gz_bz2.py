# SPDX-FileCopyrightText: 2023 Philippe Proulx <eeppeliteloop@gmail.com>
# SPDX-License-Identifier: MIT

import bz2
import gzip
import typing

import normand


def _test_comp(type: str, dec_func: typing.Callable[[bytes], bytes]):
    data = b"bonjour tout le monde \x00\x23\x42 \x17" + b"\x7b" * 500
    ntext = "!t {} {} !end".format(type, data.hex())
    res = normand.parse(ntext)
    assert dec_func(res.data) == data


def test_gz():
    _test_comp("gz", gzip.decompress)


def test_bz2():
    _test_comp("bz2", bz2.decompress)
