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

import re

import pytest

import normand


def pytest_collect_file(parent, file_path):
    ext = ".nt"

    if file_path.suffix != ext:
        # Not a Normand test file: cancel
        return

    test_name = "test-{}".format(file_path.name.replace(ext, ""))

    # Create the file node
    if file_path.name.startswith("fail-"):
        return _NormandTestFileFail.from_parent(parent, path=file_path, name=test_name)
    elif file_path.name.startswith("pass-"):
        return _NormandTestFilePass.from_parent(parent, path=file_path, name=test_name)
    else:
        # `.nt` file isn't a test case
        return


def _split_nt_file(path):
    normand_lines = []
    output_lines = []
    cur_lines = normand_lines

    with open(path) as f:
        for line in f:
            if line.rstrip() == "---" and len(output_lines) == 0:
                cur_lines = output_lines
                continue

            cur_lines.append(line)

    return "".join(normand_lines), "".join(output_lines).strip()


class _NormandTestItem(pytest.Item):
    def runtest(self):
        self._runtest(*_split_nt_file(self.path))

    def reportinfo(self):
        return self.path, None, self.name


class _NormandTestItemFail(_NormandTestItem):
    def _runtest(self, normand_text, output):
        with pytest.raises(normand.ParseError) as exc_info:
            normand.parse(normand_text)

        exc = exc_info.value
        expected_msg = ""

        for msg in reversed(exc.messages):
            expected_msg += "{}:{} - {}\n".format(
                msg.text_location.line_no, msg.text_location.col_no, msg.text
            )

        assert output.strip() == expected_msg.strip()


class _NormandTestItemPass(_NormandTestItem):
    @staticmethod
    def _data_from_output(output):
        hex_bytes = re.split(r"\s+", output.strip())
        return bytearray([int(b, 16) for b in hex_bytes])

    def _runtest(self, normand_text, output):
        assert normand.parse(normand_text).data == self._data_from_output(output)


class _NormandTestFile(pytest.File):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self._name = name

    def collect(self):
        # Yield a single item
        yield self._item_cls.from_parent(self, name=self._name)


class _NormandTestFileFail(_NormandTestFile):
    _item_cls = _NormandTestItemFail


class _NormandTestFilePass(_NormandTestFile):
    _item_cls = _NormandTestItemPass
